import datasets
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import BertForSequenceClassification, AutoTokenizer, Trainer, AutoModelForSequenceClassification, \
    TrainingArguments

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
    #preds = np.argmax(preds, axis=1)
    result = {}
    result["accuracy"] = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]
    return result


def preprocess_logits(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=1)


if __name__ == "__main__":

    paths = {"cl_data": "/home/students/meier/MA/MA_Thesis/preprocess/verb_verid_neg.csv", #full_verb_veridicality.csv",
             "cl_model": "/workspace/students/meier/MA/BERT_mnli_filtered_larger_batch_size/checkpoint-4551", #"/workspace/students/meier/MA/BERT_mnli_filtered/checkpoint-6070", #BART_veridicality_text/checkpoint-15175/", #"/workspace/students/meier/MA/SOTA_Bart/best/checkpoint-12000/",
             "tow_model": "../checkpoint-12000/",
             "tow_data": "C:/Users/Meier/Projekte/MA_Thesis/preprocess/verb_verid_neg.csv",
             "cl_model_graph": "/workspace/students/meier/MA/amrbart_mnli_filtered_only_graph/checkpoint-2277",
             "cl_data_graph_pos": "/home/students/meier/MA/MA_Thesis/preprocess/veridicality_positive_test_graph.csv",
             "cl_data_graph_neg": "/home/students/meier/MA/MA_Thesis/preprocess/veridicality_negated_test_graph.csv"}


    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)



    dataset_test_split = load_dataset("csv", data_files={"test": paths["cl_data_graph_pos"]})

    tokenized_datasets_test = dataset_test_split.map(encode, batched=True)
    targs = TrainingArguments(eval_accumulation_steps=10, per_device_eval_batch_size=8, output_dir="./")
    trainer = Trainer(model=model, tokenizer=tokenizer, args=targs, preprocess_logits_for_metrics=preprocess_logits, compute_metrics=compute_metrics)
    # trainer.evaluate()
    model.eval()
    res = trainer.predict(tokenized_datasets_test["test"])

    #res = trainer.predict(dataset_test_split["test"])

    print(res)
    print(res.label_ids)
    #print(res.label_ids.reshape(107, 14).tolist())
    pd.DataFrame(res.predictions).to_csv("/home/students/meier/MA/results/BERT_veridicality_pos_6070.csv") #"results_mnli_matched_bartLarge.csv")
    print(res.metrics)




    """
    model = BertForSequenceClassification.from_pretrained("ishan/bert-base-uncased-mnli")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset_test_split = load_dataset("csv", data_files={"test": paths["laptop_data"]})
    tokenized_datasets_test = dataset_test_split.map(encode, batched=True)
    trainer = Trainer(model=model, tokenizer=tokenizer, compute_metrics=compute_metrics)
    model.eval()
    res = trainer.predict(tokenized_datasets_test["test"])
    """

