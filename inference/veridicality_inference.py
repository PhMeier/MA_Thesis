import evaluate
import torch
from datasets import load_dataset, load_metric
from evaluate import evaluator
from transformers import AutoTokenizer, pipeline, Trainer
from torch.utils.data import DataLoader
import datasets
import numpy as np
from transformers import AutoTokenizer, BartForSequenceClassification
from transformers import pipeline


tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

def encode(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length')#, max_length="max_length")


def compute_metrics(p): #eval_pred):
    metric_acc = datasets.load_metric("accuracy")
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = {}
    result["accuracy"] = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]
    return result



if __name__ == "__main__":
    path = "/workspace/students/meier/MA/SOTA_Bart/best/checkpoint-12000/" #"../checkpoint-12000/"
    #model = torch.load(path+"pytorch_model.bin", map_location=torch.device('cpu'))
    model = BartForSequenceClassification.from_pretrained(path, local_files_only=True)
    dataset_test_split = load_dataset("csv", data_files="/home/students/meier/MA/MA_Thesis/preprocess/verb_verid_normal.csv")
    tokenized_datasets_test = dataset_test_split.rename_column("signature", "label")
    tokenized_datasets_test = tokenized_datasets_test.rename_column("sentence", "premise")
    tokenized_datasets_test = tokenized_datasets_test.rename_column("complement", "hypothesis")
    tokenized_datasets_test = tokenized_datasets_test.map(encode, batched=True) 
    trainer = Trainer(model=model, tokenizer=tokenizer, compute_metrics=compute_metrics, eval_dataset=tokenized_datasets_test)
    trainer.evaluate()
