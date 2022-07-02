import evaluate
import torch
from datasets import load_dataset, load_metric
from evaluate import evaluator
from transformers import AutoTokenizer, pipeline
from torch.utils.data import DataLoader
import datasets
import numpy as np
from transformers import AutoTokenizer, BartForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

def encode(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length')#, max_length="max_length")

"""
def compute_metrics(p): #eval_pred):
    metric_acc = datasets.load_metric("accuracy")
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = {}
    result["accuracy"] = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]
    return result


def evaluate(model, test_data):
    #model.eval()
    metric = load_metric("accuracy")
    for batch in test_data:
        outputs = model(**batch)
        logits = outputs.loss["logits"]
        predictions = torch.argmax(logits, dim=1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    acc = metric.compute()
    acc = acc["accuracy"]
    print(acc)
"""


if __name__ == "__main__":
    path = "/workspace/students/meier/MA/SOTA_Bart/best/checkpoint-12000/" #"../checkpoint-12000/"
    #model = torch.load(path+"pytorch_model.bin", map_location=torch.device('cpu'))
    model = BartForSequenceClassification.from_pretrained(path, local_files_only=True)
    dataset_test_split = load_dataset("glue", "mnli", split='test_matched')
    #tokenized_datasets_test = dataset_test_split.map(encode, batched=True) 
    
    dataset_test_split = dataset_test_split.rename_column("premise", "text")
    dataset_test_split = dataset_test_split.rename_column("hypothesis", "text")


    pipe = pipeline("sentiment-analysis", model=model, device=0, tokenizer=tokenizer)
    metric = evaluate.load("accuracy")
    eval1 = evaluator("sentiment-analysis")

    results = eval1.compute(model_or_pipeline=pipe, data=dataset_test_split, metric=metric,
                           label_mapping={"Contradiction": -1, "Neutral": 0, "Entailment":1}, input_column="text")

    print(results)
    """
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    tokenized_datasets_t = dataset_test_split.map(encode, batched=True)
    tokenized_datasets_t = tokenized_datasets_t.rename_column("label", "labels")
    tokenized_datasets_t = tokenized_datasets_t.remove_columns(["idx"])
    tokenized_datasets_t = tokenized_datasets_t.remove_columns(["premise"])
    tokenized_datasets_t = tokenized_datasets_t.remove_columns(["hypothesis"])
    tokenized_datasets_t.set_format("torch")
    eval_dataloader = DataLoader(tokenized_datasets_t, batch_size=1)
    evaluate(model, eval_dataloader)
    """



