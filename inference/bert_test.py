import datasets
import numpy as np
from datasets import load_dataset
from transformers import BertForSequenceClassification, AutoTokenizer, Trainer

paths = {"cl_data": "/home/students/meier/MA/MA_Thesis/preprocess/verb_verid_nor.csv",
         "cl_model": "/workspace/students/meier/MA/SOTA_Bart/best/checkpoint-12000/",
         "tow_model": "../checkpoint-12000/",
         "tow_data": "C:/Users/Meier/Projekte/MA_Thesis/preprocess/verb_verid_nor.csv",
         "laptop_data": "C:/Users/phMei/PycharmProjects/MA_Thesis/preprocess/verb_verid_nor.csv"}


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
    model = BertForSequenceClassification.from_pretrained("ishan/bert-base-uncased-mnli")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset_test_split = load_dataset("csv", data_files={"test": paths["laptop_data"]})
    tokenized_datasets_test = dataset_test_split.map(encode, batched=True)
    trainer = Trainer(model=model, tokenizer=tokenizer, compute_metrics=compute_metrics)
    model.eval()
    res = trainer.predict(tokenized_datasets_test["test"])


