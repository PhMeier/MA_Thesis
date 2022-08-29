"""
In this script the joint data (text and graph) is created.
"""

import pandas as pd

def process_premise(filename):
    data_premise = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if len(line) > 4:
                # print(line)
                fline = line.strip().split(" ", 1)[1].split("<pad>")[0]
                fline = fline.replace("<s>", "<g>")
                fline = fline.replace("</AMR>", "")
                data_premise.append(fline)
    return data_premise

def combine_data(filename):
    data = []
    df = pd.read_csv(filename, index_col=False)
    print(df)
    """
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append()
    """


# Veridicality

def procedure_for_mnli_filtered_dev():
    dev_set = "../data/MNLI_filtered/MNLI_filtered/new_dev_matched_with_tags.csv" #"/home/students/meier/MA/MNLI_filtered/MNLI_filtered/new_dev_matched.tsv"
    premise_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_filtered_premise_dev_matched/dev-nodes.json"
    hypo_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_filtered_hypothesis_dev_matched/dev-nodes.json"
    combine_data(dev_set)






if __name__ == "__main__":
    # Procedures for train, dev and test data MNLI filtered
    procedure_for_mnli_filtered_dev()