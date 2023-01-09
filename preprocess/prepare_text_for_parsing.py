"""
Text data is prepared in order to be parsed by AMRBART.
Needs to be in jsonl format with "src" and "tgt" keys.

Data needs to be separated into premise and hypothesis.

"""
from datasets import load_dataset
# import jsonlines
import json
import pandas as pd

def routine_for_normal_mnli(splitname):
    mnli_train = load_dataset("glue", "mnli", split=splitname)
    data_premise = []
    data_hypothesis = []
    for prem, hypo in zip(mnli_train["premise"], mnli_train["hypothesis"]):
        data_premise.append({"src": prem, "tgt": ""})
        data_hypothesis.append({"src": hypo, "tgt": ""})
    # print(data_premise)
    print(data_hypothesis)
    # tokenized_datasets_test = dataset_test_split.map(encode, batched=True)
    # print(tokenized_datasets_test)
    with open("premise_"+splitname+".jsonl", "w", encoding="utf-8") as f:
        for line in data_premise:
            f.write(json.dumps(line) + "\n")
    with open("hypothesis_"+splitname+".jsonl", "w", encoding="utf-8") as f:
        for line in data_hypothesis:
            f.write(json.dumps(line) + "\n")


def read_data(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append(line.split("\t"))
    return data

def read_csv(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append(line.split(","))
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

# ROUTINES FOR DIFFERENT DATA SPLITS AND TYPES

def routine_for_dev_data(data):
    data_premise = []
    data_hypothesis = []
    for line in data[1:]:
        data_premise.append({"src": line[8], "tgt": ""})
        data_hypothesis.append({"src": line[9], "tgt": ""})
    with open("premise_dev_matched.jsonl", "w+", encoding="utf-8") as f:
        for line in data_premise:
            f.write(json.dumps(line) + "\n")
    with open("hypothesis_dev_matched.jsonl", "w+", encoding="utf-8") as f:
        for line in data_hypothesis:
            f.write(json.dumps(line) + "\n")

def routine_for_mnli_test_data(data):
    data_premise = []
    data_hypothesis = []
    for line in data[1:]:
        data_premise.append({"src": line[5], "tgt": ""})
        data_hypothesis.append({"src": line[6], "tgt": ""})
    with open("premise_test_matched.jsonl", "w+", encoding="utf-8") as f:
        for line in data_premise:
            f.write(json.dumps(line) + "\n")
    with open("hypothesis_test_matched.jsonl", "w+", encoding="utf-8") as f:
        for line in data_hypothesis:
            f.write(json.dumps(line) + "\n")


def routine_for_veridicality_test_data(data):
    sentence = []
    negated_sentence = []
    complement = []
    for line in data[1:]:
        sentence.append({"src": line[3], "tgt": ""})
        negated_sentence.append({"src": line[4], "tgt": ""})
        complement.append({"src": line[5], "tgt": ""})
    with open("sentence_veridicality.jsonl", "w+", encoding="utf-8") as f:
        for line in sentence:
            f.write(json.dumps(line) + "\n")
    with open("negated_sentence_veridicality.jsonl", "w+", encoding="utf-8") as f:
        for line in negated_sentence:
            f.write(json.dumps(line) + "\n")
    with open("complement_veridicality.jsonl", "w+", encoding="utf-8") as f:
        for line in complement:
            f.write(json.dumps(line) + "\n")


def routine_yanaka_train(data):
    data_premise = []
    data_hypothesis = []
    for line in data[1:]:
        data_premise.append({"src": line[8], "tgt": ""})
        data_hypothesis.append({"src": line[9], "tgt": ""})
    with open("premise_train_yanaka.jsonl", "w+", encoding="utf-8") as f:
        for line in data_premise:
            f.write(json.dumps(line) + "\n")
    with open("hypothesis_train_yanaka.jsonl", "w+", encoding="utf-8") as f:
        for line in data_hypothesis:
            f.write(json.dumps(line) + "\n")


def routine_yanaka_dev(data):
    data_premise = []
    data_hypothesis = []
    for line in data[1:]:
        data_premise.append({"src": line[8], "tgt": ""})
        data_hypothesis.append({"src": line[9], "tgt": ""})
    with open("premise_dev_yanaka.jsonl", "w+", encoding="utf-8") as f:
        for line in data_premise:
            f.write(json.dumps(line) + "\n")
    with open("hypothesis_dev_yanaka.jsonl", "w+", encoding="utf-8") as f:
        for line in data_hypothesis:
            f.write(json.dumps(line) + "\n")


def routine_sick_extracted(data):
    data_premise = []
    data_hypothesis = []
    for line in data[1:]:
        print(line)
        data_premise.append({"src": line[8], "tgt": ""})
        data_hypothesis.append({"src": line[9], "tgt": ""})
    with open("premise_sick_extracted.jsonl", "w+", encoding="utf-8") as f:
        for line in data_premise:
            f.write(json.dumps(line) + "\n")
    with open("hypothesis_sick_extracted.jsonl", "w+", encoding="utf-8") as f:
        for line in data_hypothesis:
            f.write(json.dumps(line) + "\n")


def write_to_jsonl(filename, data):
    with open(filename, "w+", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def routine_sick_2(pos, neg):
    prem_pos = pos["f(s1)"].to_list()
    prem_neg = neg["f(s1)"].to_list()
    s2_pos = pos["s2"].to_list()
    hypo_pos = pos["s1"].to_list()
    hypo_neg = neg["s1"].to_list()
    s2_neg = neg["s2"].to_list()
    pos_premise = []
    pos_hypothesis = []
    neg_premise = []
    neg_hypothesis = []
    s2_pos_list = []
    s2_neg_list = []
    for p_prem, n_prem, p_hypo, n_hypo, s2p, s2n in zip(prem_pos,prem_neg, hypo_pos, hypo_neg, s2_pos, s2_neg):
        pos_premise.append({"src": p_prem, "tgt": ""})
        pos_hypothesis.append({"src": p_hypo, "tgt": ""})
        neg_premise.append({"src": n_prem, "tgt": ""})
        neg_hypothesis.append({"src": n_hypo, "tgt": ""})
        s2_pos_list.append({"src": s2p, "tgt": ""})
        s2_neg_list.append({"src": s2n, "tgt": ""})
    write_to_jsonl("pos_premise_step2.jsonl", pos_premise)
    write_to_jsonl("pos_hypothesis_step2.jsonl", pos_hypothesis)
    write_to_jsonl("neg_premise_step2.jsonl", neg_premise)
    write_to_jsonl("neg_hypothesis_step2.jsonl", neg_hypothesis)
    write_to_jsonl("sick/pos_s2.jsonl", s2_pos_list)
    write_to_jsonl("sick/neg_s2.jsonl", s2_neg_list)


if __name__ == "__main__":
    sick_pos = pd.read_csv("./sick/pos_env_complete_sick_new.csv")
    sick_neg = pd.read_csv("./sick/neg_env_complete_sick_new.csv")
    routine_sick_2(sick_pos, sick_neg)
    #dataset_train = load_dataset("glue", "mnli", split='train')
    # print(dataset_train["idx"])

    #mnli_data_filtered = read_data("home/students/meier/MA/MNLI_filtered/MNLI_filtered/new_train.tsv")
    #mnli_data_filtered = read_data("/home/students/meier/MA/MNLI_filtered/MNLI_filtered/new_dev_matched.tsv")
    #mnli_data = read_data("../data/original_MNLI/multinli_1.0/multinli_1.0/multinli_1.0_dev_matched.txt")
    #routine_for_filtered_mnli(mnli_data_filtered)
    #routine_for_dev_data(mnli_data_filtered)

    # Yanaka data Routine
    # Train
    """
    data = read_data("C:/Users/phMei/Projekte/transitivity/naturalistic/train.tsv")
    routine_yanaka_train(data)
    data = read_data("C:/Users/phMei/Projekte/transitivity/naturalistic/dev_matched.tsv")
    routine_yanaka_dev(data)
    """

    #routine_for_normal_mnli("validation_mismatched")



    # test data
    """
    test_data = read_data("/home/students/meier/MA/multinli_0.9_test_matched_unlabeled_mod.csv")
    routine_for_mnli_test_data(test_data)
    veridicality_data = read_data("/home/students/meier/MA/MA_Thesis/data/verb_veridicality_evaluation.tsv")
    routine_for_veridicality_test_data(veridicality_data)
    """
