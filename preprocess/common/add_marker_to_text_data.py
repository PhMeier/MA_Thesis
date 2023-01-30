"""
Add text marker to .tsv files and hugginface data.
Outputs a csv file

"""
import pandas
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

def add_tags_veridicality(data):
    for i in range(len(data)):
        if i > 0:
            sentence = "<t> " + data[i][3]
            neg_sent = "<t> " + data[i][4]
            complement = data[i][5] + " </t>"
            data[i][3] = sentence
            data[i][4] = neg_sent
            data[i][5] = complement
    return data


def routine_for_tsv(filename):
    data = read_tsv(filename)
    data_with_tags = add_tags(data)
    write_out_file(filename, data_with_tags)


def routine_for_veridicality_data(filename):
    data = read_tsv(filename)
    data_with_tags = add_tags_veridicality(data)
    write_out_file(filename, data_with_tags)


def routine_for_splitted_files(filename):
    data = []
    filen = open(filename, encoding="utf-8")
    read_csv = csv.reader(filen)
    print(read_csv)
    for line in read_csv:
        data.append(line)
    filen.close()
    #print(data)
    res = add_tags_to_csv(data)
    with open("verb_verid_neg_with_tags.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(res)


def add_tags_to_csv(data):
    for line in data[1:]:
        line[1] = "<t> " + line[1] + " </t>"
        line[2] = "<t> " + line[2] + " </t>"
        #print(line[1])+
    return data
    #print(data)


def add_tags_for_data_mnli_test(data):
    premise = []
    hypo = []
    for line in data[1:]:
        line[5] = "<t> " + line[5] + " </t>"
        line[6] = "<t> " + line[6] + " </t>"
        premise.append(line[5])
        hypo.append(line[6])
    return premise, hypo



def test_data_mnli(filename):
    data = []

    with open(filename, encoding="utf-8") as f:
        for line in f:
            data.append(line.strip().split("\t"))
    premise, hypo = add_tags_for_data_mnli_test(data)
    df = pd.DataFrame({"premise": premise, "hypothesis": hypo})
    df.to_csv("multinli_0.9_test_matched_unlabeled_mod_with_tags.csv", index=True, header=["premise", "hypothesis"])





if __name__ == "__main__":
    paths = {"train_data_bw": "/home/hd/hd_hd/hd_rk435/MNLI_filtered/MNLI_filtered/new_train.tsv",
             "val_data_bw": "/home/hd/hd_hd/hd_rk435/MNLI_filtered/MNLI_filtered/new_dev_matched.tsv",
             "train_data_cl": "/home/students/meier/MA/MNLI_filtered/MNLI_filtered/new_train.tsv",
             "test_data_cl": "/home/students/meier/MA/MNLI_filtered/MNLI_filtered/new_dev_matched.tsv",
             "train": "../data/MNLI_filtered/MNLI_filtered/new_train.tsv",
             "test": "../data/MNLI_filtered/MNLI_filtered/new_dev_matched.tsv",
             "test_veridicality": "../data/verb_veridicality_evaluation.tsv",
             "verid_pos": "verb_verid_nor.csv",
             "verid_neg": "verb_verid_neg.csv",
             "test_mnli": "multinli_0.9_test_matched_unlabeled_mod.csv"}

    test_data_mnli(paths["test_mnli"])
    #routine_for_splitted_files(paths["verid_neg"])


    #routine_for_veridicality_data(paths["test_veridicality"])
    #routine_for_tsv(paths["train"])
    #df_train = pd.read_csv("../data/MNLI_filtered/MNLI_filtered/new_dev_matched_with_tags.csv")
    #print(df_train.columns.values)
    #print(df_train["gold_label"])