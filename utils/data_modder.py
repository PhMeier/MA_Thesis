"""
This script is used to investigate the MNLI data in order to find overlaps with the verb veridicality dataset.

@author: Philipp Meier
"""
from datasets import load_dataset
import datasets





if __name__ == "__main__":
    dataset_train = load_dataset("glue", "mnli", split='train')  # , download_mode="force_redownload")
    dataset_val = load_dataset("glue", "mnli", split='validation_matched')
    #print(dataset_train["hypothesis"])
    query = "I started to slink away"
    for item, item2 in zip(dataset_train["premise"], dataset_train["hypothesis"]):
        if query in item:
            print(item, item2)
    for item, item2 in zip(dataset_val["premise"], dataset_val["hypothesis"]):
        if query in item:
            print(item, item2)