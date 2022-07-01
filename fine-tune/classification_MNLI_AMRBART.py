import transformers
from transformers import AutoTokenizer, BartForSequenceClassification
from transformers import BartForConditionalGeneration
from datasets import load_dataset
import datasets
import numpy as np
from datasets import load_metric
from pynvml import *
import os
import wandb

save_directories = {"cl": "/workspace/students/meier/MA/AMR_Bart", "bw":"/pfs/work7/workspace/scratch/hd_rk435-checkpointz/amrbart_mnli"}


#"""
os.environ["WANDB_DIR"] = os.getcwd()
os.environ["WANDB_CONFIG_DIR"] = os.getcwd()
#wandb.login()
wandb.login(key="64ee15f5b6c99dab799defc339afa0cad48b159b")
#"""


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


dataset_train = load_dataset("glue", "mnli", split='train') #, download_mode="force_redownload")
dataset_val = load_dataset("glue", "mnli", split='validation_matched')
#dataset = load_dataset("glue", "mnli")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
# model = BartForConditionalGeneration.from_pretrained("xfbai/AMRBART-large")
model = BartForSequenceClassification.from_pretrained("xfbai/AMRBART-large")
#model = BartForSequenceClassification.from_pretrained("facebook/bart-large")
print("Model Loaded")


print_gpu_utilization()

"""
def tokenize_function(examples):
    return tokenizer(examples["premise"], padding="max_length", max_length=512, truncation=True) # man_length

def tokenize_function_hyp(examples):
    return tokenizer(examples["hypothesis"], padding="max_length", max_length=512, truncation=True)
"""

def encode(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length')#, max_length="max_length")


tokenized_datasets_t = dataset_train.map(encode, batched=True)
tokenized_datasets_v = dataset_val.map(encode, batched=True)
#tokenized_datasets = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
small_train_dataset = tokenized_datasets_t.shuffle(seed=42)#.select(range(10))
small_eval_dataset = tokenized_datasets_v.shuffle(seed=42)#.select(range(10))
#small_test_dataset = tokenized_datasets["test_matched"].shuffle(seed=42).select(range(1000))
#print(type(small_train_dataset))

print(small_train_dataset[0])
print(small_eval_dataset[-1])

metric = load_metric("accuracy")



def compute_metrics(p): #eval_pred):
    metric_acc = datasets.load_metric("accuracy")
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = {}
    result["accuracy"] = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]
    return result
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels)
    """

from transformers import TrainingArguments, Trainer

learning_rate = 5e-5
optim = transformers.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-08, weight_decay=0.01)

training_args = TrainingArguments(evaluation_strategy="epoch", per_device_train_batch_size=16,
                                  gradient_accumulation_steps=8, logging_steps=50, per_device_eval_batch_size=4,
                                  eval_accumulation_steps=8, num_train_epochs=10, report_to="wandb", 
                                  output_dir=save_directories["cl"], gradient_checkpointing=True, fp16=True, save_total_limit=5, load_best_model_at_end=True, save_strategy="epoch") # disable wandb
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
    optimizers=(optim, transformers.get_polynomial_decay_schedule_with_warmup(optim,
                                                                              num_warmup_steps=1858,
                                                                              num_training_steps=30680)),
)
trainer.train()
#print_summary(result)

