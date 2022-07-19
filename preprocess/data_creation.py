"""
Parsing text data to AMR format by using AMRBART.

"""
import datasets
import numpy as np
from datasets import load_dataset
from transformers import Trainer
from transformers import AutoTokenizer, BartForConditionalGeneration
#import jsonlines
import json

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

def encode(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True,
                     padding='max_length')  # , max_length="max_length")


if __name__ == "__main__":
    #model = BartForConditionalGeneration.from_pretrained("xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing")
    mnli_train = load_dataset("glue", "mnli", split='train')
    data_premise = []
    data_hypothesis = []
    for prem, hypo in zip(mnli_train["premise"], mnli_train["hypothesis"]):
        data_premise.append({"src": prem, "tgt": ""})
        data_hypothesis.append({"src": hypo, "tgt": ""})
    #print(data_premise)
    print(data_hypothesis)
    #tokenized_datasets_test = dataset_test_split.map(encode, batched=True)
    #print(tokenized_datasets_test)
    with open("premise_train.jsonl", "w", encoding="utf-8") as f:
        for line in data_premise:
            f.write(json.dumps(line) + "\n")
    with open("hypothesis_train.jsonl", "w", encoding="utf-8") as f:
        for line in data_hypothesis:
            f.write(json.dumps(line) + "\n")
