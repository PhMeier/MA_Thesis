import transformers
from transformers import AutoTokenizer, BartForSequenceClassification
from transformers import BartForConditionalGeneration
from datasets import load_dataset
import numpy as np
from datasets import load_metric
from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


dataset = load_dataset("glue", "mnli") #, download_mode="force_redownload")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
# model = BartForConditionalGeneration.from_pretrained("xfbai/AMRBART-large")
model = BartForSequenceClassification.from_pretrained("xfbai/AMRBART-large")
mdel = BartForSequenceClassification.from_pretrained("facebook/bart-large")
print("Model Loaded")
print_gpu_utilization()

"""
def tokenize_function(examples):
    return tokenizer(examples["premise"], padding="max_length", max_length=512, truncation=True) # man_length

def tokenize_function_hyp(examples):
    return tokenizer(examples["hypothesis"], padding="max_length", max_length=512, truncation=True)
"""

def encode(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length', max_length=512)


tokenized_datasets = dataset.map(encode, batched=True)
tokenized_datasets = dataset.map(encode, batched=True)
#tokenized_datasets = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)#.select(range(1000))
small_eval_dataset = tokenized_datasets["validation_matched"].shuffle(seed=42)#.select(range(1000))
small_test_dataset = tokenized_datasets["test_matched"].shuffle(seed=42).select(range(1000))

print(small_train_dataset[0])
print(small_eval_dataset[-1])

metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", report_to="none", per_device_train_batch_size=2, 
                                  gradient_accumulation_steps=64, fp16=True, logging_steps=50, per_device_eval_batch_size=1, eval_accumulation_steps=10) # disable wandb
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
print_summary(result)
