"""
14.08: Training BART with tags

Checklist:
- save directory 
- path
"""
save_directories = {"cl": "/workspace/students/meier/MA/AMRBART_epoch_tags_text",
                    "bw":"/pfs/work7/workspace/scratch/hd_rk435-checkpointz/amrbart_mnli_text_tags_17"}
platform = "bw"

import transformers
from transformers import AutoTokenizer, BartForSequenceClassification
from transformers import BartForConditionalGeneration
from datasets import load_dataset
import datasets
import numpy as np
from datasets import load_metric

import os
import wandb



#"""
os.environ["WANDB_DIR"] = os.getcwd()
os.environ["WANDB_CONFIG_DIR"] = os.getcwd()
#wandb.login()
wandb.login(key="64ee15f5b6c99dab799defc339afa0cad48b159b")
#"""


def add_tag_premise(s):
    #s = "<t> " + s
    s["premise"] = "<t> " + s["premise"]
    return s

def add_tag_hypothesis(s):
    s["hypothesis"] = s["hypothesis"] + " </t>"
    return s



dataset_train = load_dataset("glue", "mnli", split='train') #, download_mode="force_redownload")
dataset_val = load_dataset("glue", "mnli", split='validation_matched')
#dataset = load_dataset("glue", "mnli")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
tokenizer.add_tokens(['<t>'], special_tokens=True)
tokenizer.add_tokens(['</t>'], special_tokens=True)
model = BartForSequenceClassification.from_pretrained("xfbai/AMRBART-large")
model.resize_token_embeddings(len(tokenizer))
print("Model Loaded")




"""
def tokenize_function(examples):
    return tokenizer(examples["premise"], padding="max_length", max_length=512, truncation=True) # man_length
def tokenize_function_hyp(examples):
    return tokenizer(examples["hypothesis"], padding="max_length", max_length=512, truncation=True)
"""

def encode(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length')#, max_length="max_length")

# add the tags
updated_train = dataset_train.map(add_tag_premise)
updated_train = dataset_train.map(add_tag_hypothesis)

updated_val = dataset_val.map(add_tag_premise)
updated_val = dataset_val.map(add_tag_hypothesis)

tokenized_datasets_t = updated_train.map(encode, batched=True)
tokenized_datasets_v = updated_val.map(encode, batched=True)
#tokenized_datasets = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
small_train_dataset = tokenized_datasets_t.shuffle(seed=42)#.select(range(10))
small_eval_dataset = tokenized_datasets_v.shuffle(seed=42)#.select(range(10))
#small_test_dataset = tokenized_datasets["test_matched"].shuffle(seed=42).select(range(1000))
#print(type(small_train_dataset))

print(small_train_dataset[0])
print(small_eval_dataset[-1])

metric = load_metric("accuracy")


def compute_metrics(p):  # eval_pred):
    metric_acc = datasets.load_metric("accuracy")
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    #print("Preds: \n", preds)
    #preds = np.argmax(preds, axis=1)
    result = {}
    result["accuracy"] = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]
    return result


def preprocess_logits(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=1) #-1)


from transformers import TrainingArguments, Trainer

learning_rate = 5e-5
optim = transformers.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-08, weight_decay=0.01)

training_args = TrainingArguments(evaluation_strategy="epoch", per_device_train_batch_size=16,
                                  gradient_accumulation_steps=8, logging_steps=50, per_device_eval_batch_size=4,
                                  eval_accumulation_steps=8, num_train_epochs=10, report_to="wandb", output_dir=save_directories[platform],
                                  gradient_checkpointing=True, fp16=True, save_strategy="epoch", save_total_limit=8, load_best_model_at_end=True, seed=17) # disable wandb

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits,
    optimizers=(optim, transformers.get_polynomial_decay_schedule_with_warmup(optim,
                                                                              num_warmup_steps=1858,
                                                                              num_training_steps=30680)),
)
trainer.train()
#print_summary(result)

