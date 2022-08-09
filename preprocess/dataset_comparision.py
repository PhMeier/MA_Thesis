"""
Script to identify ids which are not in the original MNLI dataset.

The complete MNLI data is now in AMR format. We need the indexes of the instances,
which are not in the modified dataset in order to remove them from the original MNLI data in
AMR format.
"""
from datasets import load_dataset
import datasets

def read_data(filename):
    data = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            key = line.split("\t")[7]
            #print(key)
            data[key] = line.split("\t")
    return data



if __name__ == "__main__":
    dataset_train = load_dataset("glue", "mnli", split='train')
    #print(dataset_train["idx"])
    mnli_data = read_data("../data/original_MNLI/multinli_1.0_train.txt")
    print(len(mnli_data.keys()))