"""
Parsing text data to AMR format by using AMRBART.

Data needs to be separated into premise and hypothesis

"""
from datasets import load_dataset
# import jsonlines
import json


def routine_for_normal_mnli():
    mnli_train = load_dataset("glue", "mnli", split='train')
    data_premise = []
    data_hypothesis = []
    for prem, hypo in zip(mnli_train["premise"], mnli_train["hypothesis"]):
        data_premise.append({"src": prem, "tgt": ""})
        data_hypothesis.append({"src": hypo, "tgt": ""})
    # print(data_premise)
    print(data_hypothesis)
    # tokenized_datasets_test = dataset_test_split.map(encode, batched=True)
    # print(tokenized_datasets_test)
    with open("premise_train.jsonl", "w", encoding="utf-8") as f:
        for line in data_premise:
            f.write(json.dumps(line) + "\n")
    with open("hypothesis_train.jsonl", "w", encoding="utf-8") as f:
        for line in data_hypothesis:
            f.write(json.dumps(line) + "\n")


def read_data(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append(line.split("\t"))
    return data


def routine_for_filtered_mnli(data):
    data_premise = []
    data_hypothesis = []
    for line in data[1:]:
        data_premise.append({"src": line[8], "tgt": ""})
        data_hypothesis.append({"src": line[9], "tgt": ""})
    with open("premise_train_filtered.jsonl", "w+", encoding="utf-8") as f:
        for line in data_premise:
            f.write(json.dumps(line) + "\n")
    with open("hypothesis_train_filtered.jsonl", "w+", encoding="utf-8") as f:
        for line in data_hypothesis:
            f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    dataset_train = load_dataset("glue", "mnli", split='train')
    # print(dataset_train["idx"])
    mnli_data_filtered = read_data("/home/students/meier/MA/MNLI_filtered/MNLI_filtered/new_train.tsv")
    routine_for_filtered_mnli(mnli_data_filtered)
