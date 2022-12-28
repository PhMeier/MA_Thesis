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
                label_neg = label.split("/")[1]
                label_pos = num_to_label[label.split("/")[0]]
                label_neg = num_to_label[label.split("/")[1]]
                #print("LAbel pos: ", label_pos)
                #print("Label neg: ", label_neg)
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


def dev_mismatched_procedure():
    print("######## DEV PROCEDURE ########")
    dev_labels = "/home/students/meier/MA/data/MNLI/multinli_1.0/multinli_1.0_dev_mismatched.txt"
    premise_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_premise_validation_mismatched/mnli_premise_val_mismatched.json"
    hypo_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_hypothesis_validation_mismatched/mnli_hypothesis_dev_matched/mnli_hypo_val_mismatched.json"

    dataset_val = load_dataset("glue", "mnli", split='validation_mismatched')
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

    df.to_csv("MNLI_dev_mismatched_amr.csv")


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
    df2 = pd.DataFrame(final_data_neg)

    df.to_csv("veridicality_positive_test_graph.csv")
    df2.to_csv("veridicality_negated_test_graph.csv")


def extract_label(label_file):
    num_to_label = {"entailment": 0, "neutral": 1, "contradiction": 2}
    data = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            if "gold_label" not in line:
                label = line.split("\t")[0].strip()
                data.append(num_to_label[label])
    return data




def procedure_for_yanaka_data():
    premise_train = "/home/students/meier/MA/AMRBART/fine-tune/outputs/yanaka_premise_train_yanaka/dev-nodes.json"
    hypo_train = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_hypothesis_train_yanaka/dev-nodes.json"
    premise_dev = "/home/students/meier/MA/AMRBART/fine-tune/outputs/yanaka_hypothesis_dev_yanaka/dev-nodes.json"
    hypo_dev = "/home/students/meier/MA/AMRBART/fine-tune/outputs/yanaka_hypothesis_dev_yanaka/dev-nodes.json"
    labels_train = "/home/students/meier/MA/transitivity/naturalistic/train.tsv" # idx 11
    labels_dev = "/home/students/meier/MA/transitivity/naturalistic/dev_matched.tsv" # idx 11

    #premise_train = "C:/Users/phMei/PycharmProjects/MA_Thesis/dev-nodes.json"
    
    train_labels = extract_label(labels_train)
    dev_labels = extract_label(labels_dev)

    premise_train = process_premise(premise_train)
    hypo_train = process_hypothesis(hypo_train)

    premise_dev = process_premise(premise_dev)
    hypo_dev = process_hypothesis(hypo_dev)

    final_data_train = {"premise": premise_train,
                  "hypothesis": hypo_train,
                  "label": train_labels}
    df_train = pd.DataFrame(final_data_train)


    final_data_dev = {"premise": premise_dev,
                  "hypothesis": hypo_dev,
                  "label": dev_labels}
    df_dev = pd.DataFrame(final_data_dev)

    df_train.to_csv("yanaka_train_graph.csv")
    df_dev.to_csv("yanaka_dev_graph.csv")


def procedure_create_graph_frame_mnli_validation_mismatched():
    print("######## DEV PROCEDURE ########")
    dev_labels = "/home/students/meier/MA/data/MNLI/multinli_1.0/multinli_1.0_dev_mismatched.txt"
    premise_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_premise_validation_mismatched/mnli_premise_val_mismatched.json"
    hypo_json = "/home/students/meier/MA/AMRBART/fine-tune/outputs/mnli_hypothesis_validation_mismatched/mnli_hypo_val_mismatched.json"

    dataset_val = load_dataset("glue", "mnli", split='validation_mismatched')
    print(dataset_val["label"][111])
    print(dataset_val["premise"][111])

    labels = dataset_val["label"] #, index = extract_label(dev_labels)

    print(labels)
    print(len(labels))
    premise = process_premise(premise_json)
    hypo = process_hypothesis(hypo_json)
    #premise = remove_from_list(premise, index)
    #hypo = remove_from_list(hypo, index)
    print(len(premise))
    print(len(hypo))

    # print(premise)
    final_data = {"premise": premise,
                  "hypothesis": hypo,
                  "label": labels}
    df = pd.DataFrame(final_data)

    print(df)

    df.to_csv("MNLI_dev_mismatched_amr.csv")

def procedure_extracted_sick():
    """
    This function directly creates joint data, since pure graph data is not needed anymore for the Transitivity task
    :return:
    """
    print("######## SICK Extracted ########")
    label_dict = {"entailment": 0, "neutral": 1, "contradiction": 2}
    extracted_sick = pd.read_csv("sick/extracted_sick_instances.csv")
    premise_json = "sick/premise_sick_extracted.json"
    hypothesis_json = "sick/hypothesis_sick_extracted.json"

    premise_graph = process_premise(premise_json)
    hypo_graph = process_hypothesis(hypothesis_json)

    text_prem = extracted_sick["sentence_B_original"].to_list()
    text_hypo = extracted_sick["sentence_A_dataset"].to_list()
    labels = extracted_sick["SemEval_set"].to_list()
    labels = [label_dict[i] for i in labels]

    text_prem = ["<t> " + i + " </t>" for i in text_prem]
    text_hypo = ["<t> " + i + " </t>" for i in text_hypo]

    premise_joint = [i + " " + j for i,j in zip(text_prem, premise_graph)]
    hypo_joint = [i + " " + j for i,j in zip(hypo_graph, text_hypo)]
    print(premise_joint[0])
    print(hypo_joint[0])
    data_joint = {"premise":premise_joint, "hypothesis": hypo_joint, "label": labels}
    df = pd.DataFrame(data_joint, columns=["premise", "hypothesis", "label"])
    print(df.head())
    df.to_csv("extracted_sick_joint.csv", index=True)





if __name__ == "__main__":
    procedure_extracted_sick()
    #procedure_create_graph_frame_mnli_validation_mismatched()
    # train_procedure()
    # dev_procedure()
    #train_procedure_for_filtered_data()
    #dev_procedure_for_filtered_data()
    #procedure_for_mnli_test_data()
    #procedure_for_veridicality_test_set()
    #procedure_for_yanaka_data()
