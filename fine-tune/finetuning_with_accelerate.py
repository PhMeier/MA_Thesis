import transformers
from transformers import AutoTokenizer, BartForSequenceClassification, AutoModelForSequenceClassification
from transformers import BartForConditionalGeneration
from datasets import load_dataset
import datasets
import numpy as np
from datasets import load_metric
import os
import wandb
from transformers import TrainingArguments, Trainer
from accelerate import Accelerator
from torchtext.datasets import MultiNLI

import torch
import torch.nn.functional as F
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader

import torch
import matplotlib.pyplot as plt
from utils import model_saver
from utils.model_saver import save_model


plt.style.use('ggplot')





save_best_model = model_saver.SaveBestModel() #SaveBestModel()
save_directories = {"cl": "/workspace/students/meier/MA/SOTA_Bart",
                    "bw":"/pfs/work7/workspace/scratch/hd_rk435-checkpointz/amrbart_mnli"}
save_path = save_directories["bw"]


os.environ["WANDB_DIR"] = os.getcwd()
os.environ["WANDB_CONFIG_DIR"] = os.getcwd()
#wandb.login()
wandb.login(key="64ee15f5b6c99dab799defc339afa0cad48b159b")
wandb.init()



def encode(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length')#, max_length="max_length")

def compute_metrics(p): #eval_pred):
    metric_acc = datasets.load_metric("accuracy")
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = {}
    result["accuracy"] = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]
    return result




# Dataset
dataset_train_split = load_dataset("glue", "mnli", split='train') #, download_mode="force_redownload")
dataset_val_split = load_dataset("glue", "mnli", split='validation_matched')
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
tokenized_datasets_t = dataset_train_split.map(encode, batched=True)
tokenized_datasets_v = dataset_val_split.map(encode, batched=True)

tokenized_datasets_t = tokenized_datasets_t.rename_column("label", "labels")
tokenized_datasets_v = tokenized_datasets_v.rename_column("label", "labels")
tokenized_datasets_t = tokenized_datasets_t.remove_columns(["idx"])
tokenized_datasets_v = tokenized_datasets_v.remove_columns(["idx"])
tokenized_datasets_t = tokenized_datasets_t.remove_columns(["premise"])
tokenized_datasets_v = tokenized_datasets_v.remove_columns(["premise"])
tokenized_datasets_t = tokenized_datasets_t.remove_columns(["hypothesis"])
tokenized_datasets_v = tokenized_datasets_v.remove_columns(["hypothesis"])
tokenized_datasets_t.set_format("torch")
tokenized_datasets_v.set_format("torch")


small_train_dataset = tokenized_datasets_t.shuffle(seed=42)#.select(range(2))
small_eval_dataset = tokenized_datasets_v.shuffle(seed=42)#.select(range(2))


train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=16)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=16)


accelerator = Accelerator(fp16=True)

model = BartForSequenceClassification.from_pretrained("facebook/bart-large")
#model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large", num_labels=3)
learning_rate = 5e-5
optim = transformers.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-08, weight_decay=0.01)
train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(train_dataloader, eval_dataloader, model, optim)

epochs = 10
num_training_steps = epochs * len(train_dataloader)
num_training_steps = 30680

lr_scheduler = transformers.get_polynomial_decay_schedule_with_warmup(optim, num_warmup_steps=1858, num_training_steps=num_training_steps)

from tqdm.auto import tqdm

log_intervall = 500
#wandb.watch(model, log_freq=log_intervall)

# Training Loop
progress_bar = tqdm(range(num_training_steps))

criterion = nn.CrossEntropyLoss()


model.train()
for epoch in range(epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss["loss"]
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    metric = load_metric("accuracy")
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.loss["logits"]
        predictions = torch.argmax(logits, dim=1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    acc = metric.compute()
    acc = acc["accuracy"]
    print(acc)
    save_best_model(acc, epoch, model, optimizer, criterion, save_path)
    #save_best_model

save_model(epochs, model, optimizer, criterion, save_path)