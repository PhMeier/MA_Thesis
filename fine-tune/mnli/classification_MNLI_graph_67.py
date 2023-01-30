import transformers
from transformers import AutoTokenizer, BartForSequenceClassification
from transformers import BartForConditionalGeneration
from datasets import load_dataset
import datasets
import numpy as np
from datasets import load_metric
#from pynvml import *
import os
import wandb
import pandas as pd
import datasets
from datasets import Dataset


save_directories = {"cl": "/workspace/students/meier/MA/AMR_Bart", "bw":"/pfs/work7/workspace/scratch/hd_rk435-checkpointz/bart_mnli_only_graph_67"}

def encode(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length')#, max_length="max_length")


os.environ["WANDB_DIR"] = os.getcwd()
os.environ["WANDB_CONFIG_DIR"] = os.getcwd()
#wandb.login()
wandb.login(key="64ee15f5b6c99dab799defc339afa0cad48b159b")
#wandb.run.name="BW-AMRBART-4Gpus"

paths = {"train_data_bw": "/home/hd/hd_hd/hd_rk435/data/mnli_amr/MNLI_train_amr.csv",
         "test_data_bw": "/home/hd/hd_hd/hd_rk435/data/mnli_amr/MNLI_dev_matched_amr.csv",
         "train_data_cl": "/home/students/meier/MA/data/mnli_amr/MNLI_amr.csv",
         "test_data_cl": "/home/students/meier/MA/data/mnli_amr/MNLI_dev_matched_amr.csv",
         "train": "../data/MNLI_filtered/MNLI_filtered/new_train.tsv",
         "test": "../data/MNLI_filtered/MNLI_filtered/new_dev_matched.tsv"}

platform = "bw"


model = BartForSequenceClassification.from_pretrained("xfbai/AMRBART-large")
df_train = pd.read_csv(paths["train_data_" + platform])
df_val = pd.read_csv(paths["test_data_" + platform])

dataset_train_split = Dataset.from_pandas(df_train)
dataset_val_split = Dataset.from_pandas(df_val)

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
num_added_toks = tokenizer.add_tokens(['<g>'], special_tokens=True) ##This line is updated
num_added_toks = tokenizer.add_tokens(['</g>'], special_tokens=True)
model.resize_token_embeddings(len(tokenizer))

dataset_train_split = dataset_train_split.map(encode, batched=True)
dataset_val_split = dataset_val_split.map(encode, batched=True)

small_train_dataset = dataset_train_split.shuffle(seed=42)#.select(range(10))
small_eval_dataset = dataset_val_split.shuffle(seed=42)#.select(range(10))



"""
def tokenize_function(examples):
    return tokenizer(examples["premise"], padding="max_length", max_length=512, truncation=True) # man_length

def tokenize_function_hyp(examples):
    return tokenizer(examples["hypothesis"], padding="max_length", max_length=512, truncation=True)
"""





metric = load_metric("accuracy")


def compute_metrics(p):  # eval_pred):
    metric_acc = datasets.load_metric("accuracy")
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    print(preds)
    print(preds.shape)
    #preds = np.argmax(preds, axis=0)
    result = {}
    print("Preds after argmax: \n ", preds)
    result["accuracy"] = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]
    return result


def preprocess_logits(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    print(logits)
    return logits.argmax(dim=1) #-1)


from transformers import TrainingArguments, Trainer

learning_rate = 5e-5
optim = transformers.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-08, weight_decay=0.01)

training_args = TrainingArguments(evaluation_strategy="epoch", per_device_train_batch_size=16,
                                  gradient_accumulation_steps=8, logging_steps=50, per_device_eval_batch_size=2,
                                  eval_accumulation_steps=10, num_train_epochs=10, report_to="wandb",
                                  output_dir=save_directories[platform], gradient_checkpointing=True, fp16=True,
                                  save_total_limit=10, load_best_model_at_end=True, save_strategy="epoch", seed=67) # disable wandb
#preprocess_logits_for_metrics=preprocess_logits,
#compute_metrics=compute_metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    preprocess_logits_for_metrics=preprocess_logits,
    compute_metrics=compute_metrics,
    optimizers=(optim, transformers.get_polynomial_decay_schedule_with_warmup(optim,
                                                                              num_warmup_steps=1858,
                                                                              num_training_steps=30680)),

)
trainer.train()
#print_summary(result)

