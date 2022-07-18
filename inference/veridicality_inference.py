import evaluate
import torch
from datasets import load_dataset, load_metric
from evaluate import evaluator
from transformers import AutoTokenizer, pipeline, Trainer
from torch.utils.data import DataLoader
import datasets
import numpy as np
from transformers import AutoTokenizer, BartForSequenceClassification
from transformers import pipeline, TrainingArguments
#import numpy as np
#np.set_printoptions(threshold=np.inf)
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")


def encode(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True,
                     padding='max_length')  # , max_length="max_length")


def compute_metrics(p):  # eval_pred):
    metric_acc = datasets.load_metric("accuracy")
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = {}
    result["accuracy"] = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]
    return result


if __name__ == "__main__":
    paths = {"cl_data": "/home/students/meier/MA/MA_Thesis/preprocess/full_verb_veridicality.csv",
             "cl_model": "/workspace/students/meier/MA/SOTA_Bart/best/checkpoint-12000/",
             "tow_model": "../checkpoint-12000/",
             "tow_data": "C:/Users/Meier/Projekte/MA_Thesis/preprocess/verb_verid_nor.csv"}

    # /workspace/students/meier/MA/SOTA_Bart/best
    path = "../checkpoint-12000/"  # "../checkpoint-12000/"
    # model = torch.load(path+"pytorch_model.bin", map_location=torch.device('cpu'))
    model = BartForSequenceClassification.from_pretrained(paths["cl_model"], local_files_only=True)
    dataset_test_split = load_dataset("csv", data_files={"test": paths["cl_data"]})
    #tokenized_datasets_test = dataset_test_split.rename_column("signature", "label")
    #tokenized_datasets_test = tokenized_datasets_test.rename_column("sentence", "premise")
    #tokenized_datasets_test = tokenized_datasets_test.rename_column("complement", "hypothesis")
    tokenized_datasets_test = dataset_test_split.map(encode, batched=True)
    dataset_test_split = dataset_test_split.select(10)
    targs = TrainingArguments(eval_accumulation_steps=10, per_device_eval_batch_size=2, output_dir="./")
    trainer = Trainer(model=model, tokenizer=tokenizer, compute_metrics=compute_metrics, args=targs)
    # trainer.evaluate()
    model.eval()
    res = trainer.predict(tokenized_datasets_test["test"])
    print(res)

    #print(res.label_ids.reshape(107, 14).tolist())
    pd.DataFrame(res.label_ids).to_csv("results.csv")
    print(res.metrics)

