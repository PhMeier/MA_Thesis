"""
Add text marker to .tsv files and hugginface data

"""
import pandas as pd
import csv

def read_tsv(file):
    data = []
    tsv_file = open(file, encoding="utf-8")
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    for line in read_tsv:
        data.append(line)
    print(data[0])
    return data


def add_tags(data):
    for i in range(len(data)):
        if i > 0:
            prem = "<t> " + data[i][8]
            hypo = data[i][9] + " </t>"
            data[i][8] = prem
            data[i][9] = hypo
    return data


def write_out_file(filename,data):
    print(data[0])

    new_file = filename.split(".tsv")[0]
    df = pd.DataFrame(data[1:], columns=data[0])
    df.to_csv(new_file+"_with_tags.csv", encoding="utf-8", index=False)
    print(df)
    """
    with open(new_file+"_with_tags.tsv", "w+", encoding="utf-8") as f:
        for line in data:
            f.write(line)
    """

def routine_for_tsv(filename):
    data = read_tsv(filename)
    data_with_tags = add_tags(data)
    write_out_file(filename, data_with_tags)




if __name__ == "__main__":
    paths = {"train_data_bw": "/home/hd/hd_hd/hd_rk435/MNLI_filtered/MNLI_filtered/new_train.tsv",
             "val_data_bw": "/home/hd/hd_hd/hd_rk435/MNLI_filtered/MNLI_filtered/new_dev_matched.tsv",
             "train_data_cl": "/home/students/meier/MA/MNLI_filtered/MNLI_filtered/new_train.tsv",
             "test_data_cl": "/home/students/meier/MA/MNLI_filtered/MNLI_filtered/new_dev_matched.tsv",
             "train": "../data/MNLI_filtered/MNLI_filtered/new_train.tsv",
             "test": "../data/MNLI_filtered/MNLI_filtered/new_dev_matched.tsv"}
    routine_for_tsv(paths["train"])
    df_train = pd.read_csv("../data/MNLI_filtered/MNLI_filtered/new_dev_matched_with_tags.csv")
    print(df_train.columns.values)
    print(df_train["gold_label"])