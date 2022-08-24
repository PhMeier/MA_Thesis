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
                # print(line)
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
                # print(line)
                fline = line.strip().split(" ", 1)[1].split("<pad>")[0]
                fline = fline.replace("<s>", "")
                fline = fline.replace("</AMR>", "</g>")
                # print(fline)
                data_hypo.append(fline)
    return data_hypo


def process_sentence(filename):
    data_sentence= []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if len(line) > 4:
                # print(line)
                fline = line.strip().split(" ", 1)[1].split("<pad>")[0]
                fline = fline.replace("<s>", "<g>")
                fline = fline.replace("</AMR>", "")
                data_sentence.append(fline)
    return data_sentence


def process_complement(filename):
    data_hypo = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if len(line) > 4:
                # print(line)
                fline = line.strip().split(" ", 1)[1].split("<pad>")[0]
                fline = fline.replace("<s>", "")
                fline = fline.replace("</AMR>", "</g>")
                # print(fline)
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
                    # print(line.split("\t")[0])
                    data.append(num_to_label[label])
                else:
                    indexes.append(index)
    return data, indexes


def extract_label_filtered_data(filename):
    data, indexes = [], []
    num_to_label = {"entailment\n": 0, "neutral\n": 1, "contradiction\n": 2, "entailment": 0, "neutral": 1,
                    "contradiction": 2}
    with open(filename, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if "signature" not in line:
                label = line.split("\t")[11]
                if label != "-":
                    # print(line.split("\t")[0])
                    data.append(num_to_label[label])
                else:
                    indexes.append(index)
    return data, indexes


def extract_label_veridicality_test_data(filename):
    labels_pos, labels_neg = [], []
    num_to_label = {"+\n": 0, "o\n": 1, "-\n": 2, "+": 0, "o": 1, "-": 2}
    with open(filename, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if "signature" not in line:
                label = line.split("\t")[14]
                label_pos = label.split("/")[0]
                print("label: ", label)

                label_neg = label.split("/")[1]
                labels_pos.append(label_pos)
                labels_neg.append(label_neg)
    return labels_pos, labels_neg


def remove_from_list(data, index):
    for idx in reversed(index):
        data.pop(idx)
    return data


def train_procedure():
    print("######## TRAIN PROCEDURE ########")
    training_labels = "/home/students/meier/MA/data/MNLI/multinli_1.0/multinli_1.0_train.txt"
    premise_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_premise.json"
    hypo_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_hypothesis.json"

    labels, index = extract_label(training_labels)
    print(labels)
    print(len(labels))
    premise = process_premise(premise_json)
    hypo = process_hypothesis(hypo_json)
    print(len(premise))
    print(len(hypo))

    # print(premise)
    final_data = {"premise": premise,
                  "hypothesis": hypo,
                  "label": labels}
    df = pd.DataFrame(final_data)

    print(df)
    df.to_csv("MNLI_train_amr.csv")


def dev_procedure():
    print("######## DEV PROCEDURE ########")
    dev_labels = "/home/students/meier/MA/data/MNLI/multinli_1.0/multinli_1.0_dev_matched.txt"
    premise_json = "v"
    hypo_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_hypothesis_dev_matched/dev_matched_hypo.json"

    dataset_val = load_dataset("glue", "mnli", split='validation_matched')
    print(dataset_val["label"][111])
    print(dataset_val["premise"][111])

    labels, index = extract_label(dev_labels)
    print(labels)
    print(len(labels))
    premise = process_premise(premise_json)
    hypo = process_hypothesis(hypo_json)
    premise = remove_from_list(premise, index)
    hypo = remove_from_list(hypo, index)
    print(len(premise))
    print(len(hypo))

    # print(premise)
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

    # print(premise)
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

    labels, index = extract_label_filtered_data(dev_labels)
    print(labels)
    print(len(labels))
    premise = process_premise(premise_json)
    hypo = process_hypothesis(hypo_json)
    premise = remove_from_list(premise, index)
    hypo = remove_from_list(hypo, index)
    print(len(premise))
    print(len(hypo))

    # print(premise)
    final_data = {"premise": premise,
                  "hypothesis": hypo,
                  "label": labels}
    df = pd.DataFrame(final_data)

    print(df)
    df.to_csv("MNLI_filtered_dev_matched_amr.csv")


def procedure_for_mnli_test_data():
    premise_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_premise_test/dev-nodes.json"
    hypo_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_hypothesis_test/dev-nodes.json"


    premise = process_premise(premise_json)
    hypo = process_hypothesis(hypo_json)

    final_data = {"premise": premise,
                  "hypothesis": hypo}
    df = pd.DataFrame(final_data)

    print(df)
    df.to_csv("MNLI_test_set_kaggle_graph.csv")




def procedure_for_veridicality_test_set():
    sentence_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/veridicality_sentence_test/dev-nodes.json"
    negated_sentence_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/veridicality_negated_sentence_test/dev-nodes.json"
    complement_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/veridicality_complement_test/dev-nodes.json"
    labels = "/home/students/meier/MA/MA_Thesis/data/verb_veridicality_evaluation.tsv"

    labels_pos, labels_neg = extract_label_veridicality_test_data(labels)

    sentence = process_sentence(sentence_json)
    negated_sentence = process_sentence(negated_sentence_json)
    complement = process_complement(complement_json)

    final_data_pos = {"premise": sentence,
                  "hypothesis": complement,
                  "label": labels_pos}

    final_data_neg = {"premise": negated_sentence,
                  "hypothesis": complement,
                  "label": labels_neg}

    df = pd.DataFrame(final_data_pos)
    df = pd.DataFrame(final_data_neg)

    df.to_csv("veridicality_positive_test_graph.csv")
    df.to_csv("veridicality_negated_test_graph.csv")



if __name__ == "__main__":
    # train_procedure()
    # dev_procedure()
    #train_procedure_for_filtered_data()
    #dev_procedure_for_filtered_data()
    procedure_for_mnli_test_data()
    procedure_for_veridicality_test_set()
