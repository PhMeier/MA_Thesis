"""
This script is used to investigate the MNLI data in order to find overlaps with the verb veridicality dataset.

- 14.08: Test if manipulation of datasets is possible, write code to add tags.
Manipulate datasets in huggingface
https://huggingface.co/docs/datasets/process
@author: Philipp Meier
"""
from datasets import load_dataset
import datasets
from collections import Counter


def add_tag_premise(s):
    #s = "<t> " + s
    s["premise"] = "<t> " + s["premise"]
    return s

def add_tag_hypothesis(s):
    s["hypothesis"] = s["hypothesis"] + " </t>"
    return s



if __name__ == "__main__":
    dataset_train = load_dataset("glue", "mnli", split='train')  # , download_mode="force_redownload")
    dataset_val = load_dataset("glue", "mnli", split='validation_matched')
    c = Counter(dataset_val["label"])
    print(c)
    #print(dataset_train["hypothesis"])
    query = "I started to slink away"
    #map(add_tag, dataset_train["premise"])
    #print(dataset_train["premise"])
    #updated_train = dataset_train.map(lambda prem: {"premise": "<t> "+ prem["premise"]}) #add_tag, dataset_train["premise"])
    #updated_train = dataset_train.map(add_tag_premise)
    #updated_train = dataset_train.map(add_tag_hypothesis)
    #print(updated_train["premise"])
    #print(updated_train["hypothesis"])
    """
    for item, item2 in zip(dataset_train["premise"], dataset_train["hypothesis"]):
        if query in item:
            print(item, item2)
    for item, item2 in zip(dataset_val["premise"], dataset_val["hypothesis"]):
        if query in item:
            print(item, item2)
    """