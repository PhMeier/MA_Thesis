"""
Main step of creating the joint data.
1) Remove tags
2) create_joint_data
In this script the joint data (text and graph) is created.
We need the text data
"""

import pandas as pd
import itertools
from datasets import Dataset, load_dataset


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


def process_premise_graph(filename):
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


def process_hypothesis_graph(filename):
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





def combine_data(filename):
    data = []
    df = pd.read_csv(filename, index_col=False)
    print(df)
    """
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append()
    """


def combine_lists_premise(text_data, graph_data):
    result = []
    for t, g in zip(text_data, graph_data):
        t = "<t> " + t + " </t>"
        result.append(t + " " + g)
    return result


def combine_lists_hypothesis(text_data, graph_data):
    result = []
    for t, g in zip(text_data, graph_data):
        t = "<t> "+ t + "</t>"
        g = g.replace("<g>", "")
        g = g + "</g>"
        result.append(g + " " + t)
    return result



def combine_lists_hypothesis_verid(text_data, graph_data):
    result = []
    for t, g in zip(text_data, graph_data):
        t = "<t> "+ t + "</t>"
        g = g.replace("<g>", "")
        result.append(g + " " + t)
    return result


def get_text_premise_and_hypo(filename, premise_g, hypo_g):
    df = pd.read_csv(filename, index_col=False)
    #print(df["sentence1"])
    df["sentence1"] = df["sentence1"].map(lambda x:x+" </t>")
    df["sentence2"] = df["sentence2"].map(lambda x:"<t> " + x)
    premise_text = df["sentence1"].to_list()
    hypo_text = df["sentence2"].to_list()
    #print(premise_text)
    #print(hypo_text)
    new_prem = combine_lists_premise(premise_text, premise_g)
    new_hypo = combine_lists_hypothesis_verid(hypo_text, hypo_g)
    #print(new_prem[1])
    #print(new_hypo[1])
    labels = df["gold_label"].to_list()
    final_data = {"premise": new_prem,
                  "hypothesis": new_hypo,
                  "label": labels}
    df = pd.DataFrame(final_data)
    return df

# for test data
def extract_label_veridicality_test_data(filename):
    labels_pos, labels_neg = [], []
    sentences, neg_sentences, complements = [],[],[]
    num_to_label = {"+\n": 0, "o\n": 1, "-\n": 2, "+": 0, "o": 1, "-": 2}
    with open(filename, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if "signature" not in line:
                sentence = line.split("\t")[3]
                neg_sentence = line.split("\t")[4]
                compl = line.split("\t")[5]
                sentence = "<t> " + sentence
                neg_sentence = "<t> " + neg_sentence
                compl = "<t> " + compl + " </t>"
                label = line.split("\t")[14]
                label_pos = label.split("/")[0]
                label_neg = label.split("/")[1]
                label_pos = num_to_label[label.split("/")[0]]
                label_neg = num_to_label[label.split("/")[1]]
                #print("LAbel pos: ", label_pos)
                #print("Label neg: ", label_neg)
                labels_pos.append(label_pos)
                labels_neg.append(label_neg)
                sentences.append(sentence)
                neg_sentences.append(neg_sentence)
                complements.append(compl)
    return labels_pos, labels_neg, sentences, neg_sentences, complements


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



# --------------------------------------------------------------------

# Veridicality

def procedure_for_mnli_filtered_dev():
    dev_set = "../data/MNLI_filtered/MNLI_filtered/new_dev_matched_with_tags.csv" #"/home/students/meier/MA/MNLI_filtered/MNLI_filtered/new_dev_matched.tsv"
    premise_json = "../data/MNLI_filtered/MNLI_filtered/dev_matched_premise.json"#"/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_filtered_premise_dev_matched/dev-nodes.json"
    hypo_json ="../data/MNLI_filtered/MNLI_filtered/dev_matched_hypo.json"#"/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_filtered_hypothesis_dev_matched/dev-nodes.json"
    #combine_data(dev_set)
    premise = process_premise_graph(premise_json)
    hypo = process_hypothesis_graph(hypo_json)
    #print(premise)
    df = get_text_premise_and_hypo(dev_set, premise, hypo)
    df.to_csv("/home/students/meier/MA/MNLI_filtered/MNLI_filtered/new_dev_matched_joint_input.csv")


def procedure_for_mnli_filtered_data_train():
    training_data = "/home/students/meier/MA/MNLI_filtered/MNLI_filtered/new_train_with_tags.csv"
    premise_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_filtered_premise.json"
    hypo_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_filtered_hypothesis.json"
    #combine_data(dev_set)
    premise = process_premise_graph(premise_json)
    hypo = process_hypothesis_graph(hypo_json)
    #print(premise)
    df = get_text_premise_and_hypo(training_data, premise, hypo)
    df.to_csv("/home/students/meier/MA/MNLI_filtered/MNLI_filtered/new_train_matched_joint_input.csv")


def procedure_veridicality_test_data():
    sentence_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/veridicality_sentence_test/dev-nodes.json"
    negated_sentence_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/veridicality_negated_sentence_test/dev-nodes.json"
    complement_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/veridicality_complement_test/dev-nodes.json"
    labels = "/home/students/meier/MA/MA_Thesis/data/verb_veridicality_evaluation.tsv"

    labels_pos, labels_neg, sentences, neg_sentences, complements = extract_label_veridicality_test_data(labels)
    sentence = process_sentence(sentence_json)
    negated_sentence = process_sentence(negated_sentence_json)
    complement = process_complement(complement_json)

    sentences_combined = combine_lists_premise(sentences, sentence)
    neg_sentences_combined = combine_lists_premise(neg_sentences, negated_sentence)
    complements_combined = combine_lists_hypothesis_verid(complements, complement)

    final_data_pos = {"premise": sentences_combined,
                  "hypothesis": complements_combined,
                  "label": labels_pos}

    final_data_neg = {"premise": neg_sentences_combined,
                  "hypothesis": complements_combined,
                  "label": labels_neg}

    df = pd.DataFrame(final_data_pos)
    df2 = pd.DataFrame(final_data_neg)

    df.to_csv("veridicality_positive_test_joint_input.csv")
    df2.to_csv("veridicality_negated_test_joint_input.csv")







# normal MNLI


def extract_label(filename):
    data = []
    indexes = []
    premise = []
    hypo = []
    num_to_label = {"entailment": 0, "neutral": 1, "contradiction": 2}
    with open(filename, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if "gold_label" not in line:
                label = line.split("\t")[0]
                premise.append(line.split("\t")[5])
                hypo.append(line.split("\t")[6])
                if label != "-":
                    # print(line.split("\t")[0])
                    data.append(num_to_label[label])
                else:
                    indexes.append(index)
    return data, indexes, premise, hypo


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

#def process_hypo_joint_generation()


def train_procedure():
    print("######## TRAIN PROCEDURE ########")
    training_labels = "/home/students/meier/MA/data/MNLI/multinli_1.0/multinli_1.0_train.txt"
    premise_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_premise.json"
    hypo_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_hypothesis.json"

    labels, index, premise_text, hypo_text = extract_label(training_labels)
    #print(labels)
    #print(len(labels))
    premise_g = process_premise(premise_json)
    hypo_g = process_hypothesis(hypo_json)
    #print(len(premise))
    #print(len(hypo))
    premise = combine_lists_premise(premise_text, premise_g)
    hypo = combine_lists_hypothesis_verid(hypo_text, hypo_g)

    # print(premise)
    final_data = {"premise": premise,
                  "hypothesis": hypo,
                  "label": labels}
    df = pd.DataFrame(final_data)

    #print(df)
    df.to_csv("MNLI_train_joint_input.csv")


def remove_from_list(data, index):
    for idx in reversed(index):
        data.pop(idx)
    return data


def dev_procedure():
    print("######## DEV PROCEDURE ########")
    dev_labels = "/home/students/meier/MA/data/MNLI/multinli_1.0/multinli_1.0_dev_mismatched.txt"
    premise_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_premise_validation_mismatched/mnli_premise_val_mismatched.json"
    hypo_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_hypothesis_validation_mismatched/mnli_hypo_val_mismatched.json"
    dataset_val = load_dataset("glue", "mnli", split='validation_mismatched')
    labels, index, premise_text, hypo_text = extract_label(dev_labels)
    #print(labels)
    print(len(labels))
    premise_g = process_premise(premise_json)
    hypo_g = process_hypothesis(hypo_json)
    premise_g = remove_from_list(premise_g, index)
    hypo_g = remove_from_list(hypo_g, index)
    premise_text = remove_from_list(premise_text, index)
    hypo_text = remove_from_list(hypo_text, index)
    #print(len(premise))
    #print(len(hypo))
    premise = combine_lists_premise(premise_text, premise_g)
    hypo = combine_lists_hypothesis_verid(hypo_text, hypo_g)

    # print(premise)
    final_data = {"premise": premise,
                  "hypothesis": hypo,
                  "label": labels}
    df = pd.DataFrame(final_data)

    #print(df)
    df.to_csv("MNLI_dev_matched_joint_input.csv")


def test_procedure():
    test_data = "/home/students/meier/MA/data/MNLI/multinli_1.0/multinli_0.9_test_matched_unlabeled_mod.csv"
    premise_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_premise_test/dev-nodes.json"
    hypo_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_hypothesis_test/dev-nodes.json"



    premise = process_premise(premise_json)
    hypo = process_hypothesis(hypo_json)

    df = get_text_premise_and_hypo_test_data(test_data, premise, hypo)

    print(df)
    df.to_csv("MNLI_test_set_kaggle_joint.csv")



def get_text_premise_and_hypo_test_data(filename, premise_g, hypo_g):
    df = pd.read_csv(filename, sep="\t", index_col=False)
    #print(df["sentence1"])

    #df["sentence1"] = df["sentence1"].map(lambda x:"<t> " + x + " </t>")
    #df["sentence2"] = df["sentence2"].map(lambda x: "<t> " + x + " </t>")

    premise_text = df["sentence1"].to_list()
    hypo_text = df["sentence2"].to_list()
    #print(premise_text)
    #print(hypo_text)
    new_prem = combine_lists_premise(premise_text, premise_g)
    new_hypo = combine_lists_hypothesis_verid(hypo_text, hypo_g)
    #print(new_prem[1])
    #print(new_hypo[1])
    final_data = {"premise": new_prem,
                  "hypothesis": new_hypo
                  }
    df = pd.DataFrame(final_data)
    return df


def get_premise_and_hypothesis(filename):
    prem = []
    hypo = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if "index" not in line:
                #print(line)
                #print(line.split("\t"))
                p = "<t> " + line.split("\t")[8] + " </t>"
                h = "<t> " + line.split("\t")[9] + " </t>"
                prem.append(p)
                hypo.append(h)
    return prem, hypo


# 05.10.22
def yanaka_procedure():
    # get text
    train_text = "C:/Users/phMei/Projekte/transitivity/naturalistic/train.tsv"
    dev_text = "C:/Users/phMei/Projekte/transitivity/naturalistic/dev_matched.tsv"

    # get graph
    train_graph = pd.read_csv("yanaka_train_graph.csv")
    dev_graph = pd.read_csv("yanaka_dev_graph.csv")

    train_graph_prem = train_graph["premise"].to_list()
    train_graph_hypo = train_graph["hypothesis"].to_list()
    train_labels = train_graph["label"].to_list()

    dev_graph_prem = dev_graph["premise"].to_list()
    dev_graph_hypo = dev_graph["hypothesis"].to_list()
    dev_labels = dev_graph["label"].to_list()

    train_prem, train_hypo = get_premise_and_hypothesis(train_text)
    dev_prem, dev_hypo = get_premise_and_hypothesis(dev_text)


    combined_train_prem = [i + " " + j for i,j in zip(train_prem, train_graph_prem)]
    combined_train_hypo = [i + " " + j for i,j in zip(train_graph_hypo, train_hypo)]
    #ombined_train_prem = combine_data(train_prem, train_graph_prem)
    #combined_train_hypo = combine_data(train_hypo, train_graph_hypo)

    combined_dev_prem = [i + " " + j for i,j in zip(dev_prem, dev_graph_prem)]
    combined_dev_hypo = [i + " " + j for i,j in zip(dev_graph_hypo, dev_hypo)]

    #combined_dev_prem = combine_data(dev_prem, dev_graph_prem)
    #combined_dev_hypo = combine_data(dev_hypo, dev_graph_hypo)

    final_data_train = {"premise": combined_train_prem,
                  "hypothesis": combined_train_hypo,
                  "label": train_labels}
    df_train = pd.DataFrame(final_data_train)
    df_train.to_csv("yanaka_train_joint.csv")

    final_dev_train = {"premise": combined_dev_prem,
                  "hypothesis": combined_dev_hypo,
                  "label": dev_labels}
    df_dev = pd.DataFrame(final_dev_train)
    df_dev.to_csv("yanaka_dev_joint.csv")



    final_train_text = {"premise": train_prem,
                  "hypothesis": train_hypo,
                  "label": train_labels}
    df_train = pd.DataFrame(final_train_text)
    df_train.to_csv("yanaka_train_text_tags.csv")


    final_dev_text = {"premise": dev_prem,
                  "hypothesis": dev_hypo,
                  "label": dev_labels}
    df_dev = pd.DataFrame(final_dev_text)
    df_dev.to_csv("yanaka_dev_text_tags.csv")



def combine_data(text, graph):
    res = []
    for t, g in zip(text, graph):
        res.append(t + " " + g)
    return res



def mnli_mismatched_procedure():
    print("######## DEV PROCEDURE ########")
    dev_labels = "/home/students/meier/MA/data/MNLI/multinli_1.0/multinli_1.0_dev_matched.txt"
    premise_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_premise_dev_matched/dev_matched_premise.json"
    hypo_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_hypothesis_dev_matched/dev_matched_hypo.json"
    dataset_val = load_dataset("glue", "mnli", split='validation_mismatched')

    labels, index, premise_text, hypo_text = extract_label(dev_labels)
    labels = dataset_val["label"]
    #print(labels)
    print(len(labels))
    premise_g = process_premise(premise_json)
    hypo_g = process_hypothesis(hypo_json)
    #premise_g = remove_from_list(premise_g, index)
    #hypo_g = remove_from_list(hypo_g, index)
    #premise_text = remove_from_list(premise_text, index)
    #hypo_text = remove_from_list(hypo_text, index)
    #print(len(premise))
    #print(len(hypo))
    premise = combine_lists_premise(premise_text, premise_g)
    hypo = combine_lists_hypothesis(hypo_text, hypo_g)

    # print(premise)
    final_data = {"premise": premise,
                  "hypothesis": hypo,
                  "label": labels}
    df = pd.DataFrame(final_data)

    #print(df)
    df.to_csv("MNLI_dev_mismatched_joint_input.csv")


def synthetic_data():
    text_data = pd.read_csv("/results/extracted_synthetic_generation_data_no_tags.csv")
    labels = text_data["label"]
    premise_text = text_data["premise"]
    hypo_text = text_data["hypothesis"]
    premise_json = "C:/Users/phMei/PycharmProjects/MA_Thesis/data/parsed_data/synthetic_prem.json"
    hypo_json = "C:/Users/phMei/PycharmProjects/MA_Thesis/data/parsed_data/synthetic_hypo.json"
    premise_g = process_premise(premise_json)
    hypo_g = process_hypothesis(hypo_json)
    premise = combine_lists_premise(premise_text, premise_g)
    hypo = combine_lists_hypothesis(hypo_text, hypo_g)

    final_data = {"premise": premise,
                  "hypothesis": hypo,
                  "label": labels}
    df = pd.DataFrame(final_data)

    df.to_csv("synthetic_data_joint.csv")


if __name__ == "__main__":
    synthetic_data()
    # Procedures for train, dev and test data MNLI filtered
    #procedure_for_mnli_filtered_dev()

    #procedure_for_mnli_filtered_data_train()
    #procedure_veridicality_test_data()
    #train_procedure()
    #dev_procedure()

    #test_procedure()
    #yanaka_procedure()