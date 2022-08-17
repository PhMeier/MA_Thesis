"""
Parsed AMR data is here prepared to be used as input for AMRBART

Outputs a csv pandas frame

"""
import json
import pandas as pd
from datasets import Dataset, load_dataset


def process_premise(filename):
    data_premise = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if len(line) > 4:
                #print(line)
                fline = line.strip().split(" ", 1)[1].split("<pad>")[0]
                fline = fline.replace("<s>", "<g>")
                fline = fline.replace("</AMR>", "")
                data_premise.append(fline)
    return data_premise


def process_hypothesis(filename):
    data_hypo = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if len(line) > 4:
                #print(line)
                fline = line.strip().split(" ", 1)[1].split("<pad>")[0]
                fline = fline.replace("<s>", "")
                fline = fline.replace("</AMR>", "</g>")
                #print(fline)
                data_hypo.append(fline)
    return data_hypo


def extract_label(filename):
    data = []
    indexes = []
    num_to_label = {"entailment": 0, "neutral": 1, "contradiction": 2}
    with open(filename, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if "gold_label" not in line:
                label = line.split("\t")[0]
                if label != "-":
                #print(line.split("\t")[0])
                    data.append(num_to_label[label])
                else:
                    indexes.append(index)
    return data, indexes

def extract_label_filtered_data(filename):
    data, indexes = [], []
    num_to_label = {"entailment": 0, "neutral": 1, "contradiction": 2}
    with open(filename, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if "gold_label" not in line:
                label = line.split("\t")[11]
                if label != "-":
                #print(line.split("\t")[0])
                    data.append(num_to_label[label])
                else:
                    indexes.append(index)
    return data, indexes


def remove_from_list(data, index):
    for idx in reversed(index):
        data.pop(idx)
    return data


def train_procedure():
    print("######## TRAIN PROCEDURE ########")
    training_labels = "/home/students/meier/MA/data/MNLI/multinli_1.0/multinli_1.0_train.txt"
    premise_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_premise.json"
    hypo_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_hypothesis.json"

    labels, index =extract_label(training_labels)
    print(labels)
    print(len(labels))
    premise = process_premise(premise_json)
    hypo = process_hypothesis(hypo_json)
    print(len(premise))
    print(len(hypo))

    #print(premise)
    final_data = {"premise": premise,
                  "hypothesis": hypo,
                  "label": labels}
    df = pd.DataFrame(final_data)

    print(df)
    df.to_csv("MNLI_train_amr.csv")


def dev_procedure():
    print("######## TEST PROCEDURE ########")
    dev_labels = "/home/students/meier/MA/data/MNLI/multinli_1.0/multinli_1.0_dev_matched.txt"
    premise_json = "v"
    hypo_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_hypothesis_dev_matched/dev_matched_hypo.json"

    dataset_val = load_dataset("glue", "mnli", split='validation_matched')
    print(dataset_val["label"][111])
    print(dataset_val["premise"][111])


    labels, index =extract_label(dev_labels)
    print(labels)
    print(len(labels))
    premise = process_premise(premise_json)
    hypo = process_hypothesis(hypo_json)
    premise = remove_from_list(premise, index)
    hypo = remove_from_list(hypo, index)
    print(len(premise))
    print(len(hypo))

    #print(premise)
    final_data = {"premise": premise,
                  "hypothesis": hypo,
                  "label": labels}
    df = pd.DataFrame(final_data)

    print(df)

    df.to_csv("MNLI_dev_matched_amr.csv")



def train_procedure_for_filtered_data():
    print("######## TRAIN PROCEDURE FILTERED ########")
    training_labels = "/home/students/meier/MA/MNLI_filtered/MNLI_filtered/new_train.tsv"
    premise_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_filtered_premise.json"
    hypo_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_filtered_hypothesis.json"

    labels, index = extract_label_filtered_data(training_labels)
    print(labels)
    print(len(labels))
    premise = process_premise(premise_json)
    hypo = process_hypothesis(hypo_json)
    print(len(premise))
    print(len(hypo))

    #print(premise)
    final_data = {"premise": premise,
                  "hypothesis": hypo,
                  "label": labels}
    df = pd.DataFrame(final_data)

    print(df)
    df.to_csv("/home/students/meier/MA/MNLI_filtered/MNLI_filtered/MNLI_filtered_train_amr.csv")



def dev_procedure_for_filtered_data():
    print("######## TEST PROCEDURE ########")
    dev_labels = "/home/students/meier/MA/MNLI_filtered/MNLI_filtered/new_dev_matched.tsv"
    premise_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_filtered_premise_dev_matched/dev-nodes.json"
    hypo_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_filtered_hypothesis_dev_matched/dev-nodes.json"

    dataset_val = load_dataset("glue", "mnli", split='validation_matched')
    print(dataset_val["label"][111])
    print(dataset_val["premise"][111])


    labels, index =extract_label(dev_labels)
    print(labels)
    print(len(labels))
    premise = process_premise(premise_json)
    hypo = process_hypothesis(hypo_json)
    premise = remove_from_list(premise, index)
    hypo = remove_from_list(hypo, index)
    print(len(premise))
    print(len(hypo))

    #print(premise)
    final_data = {"premise": premise,
                  "hypothesis": hypo,
                  "label": labels}
    df = pd.DataFrame(final_data)

    print(df)
    df.to_csv("MNLI_filtered_dev_matched_amr.csv")




if __name__ == "__main__":
    #train_procedure()
    #dev_procedure()
    train_procedure_for_filtered_data()
    dev_procedure_for_filtered_data()
