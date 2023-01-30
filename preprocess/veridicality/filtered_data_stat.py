"""
Function to count labels in the MNLI Filtered dataset
"""

import csv
from collections import Counter

def compute_label_stats(data):
    labels = []
    for line in data[1:]:
        labels.append(line[-1])
    c = Counter(labels)
    print(c)

if __name__ == "__main__":
    path = "/data/MNLI_filtered/MNLI_filtered/new_dev_matched.tsv"
    data = list(csv.reader(open(path, 'r', encoding="utf-8"), delimiter='\t'))
    compute_label_stats(data)
    #print(data)
