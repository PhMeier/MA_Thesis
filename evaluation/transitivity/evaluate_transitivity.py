"""
This script evalutes the results generated from the transitivity steps.
Step 1 inference in ./inference/transitivity/sick_extracted_step_1.py
"""
import sys
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, f1_score


def calc_metrics_step1():
    """

    :return:
    """
    path_to_predictions = "../../results/transitivity/step1/sick_amrbart_text_42.csv"
    gold_label = "../../results/transitivity/step1/extracted_sick_with_tags.csv"
    predictions = pd.read_csv(path_to_predictions)
    gold = pd.read_csv(gold_label)
    y_pred = predictions["label"].tolist()
    y_true = gold["label"].tolist()
    print(y_pred)
    print(y_true)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    print("{}{:.3f}".format("Accuracy: ", accuracy))
    print("{}{:.3f}".format("Precision: ", precision))
    print("{}{:.3f}".format("Recall: ", recall))
    print(classification_report(y_true, y_pred, zero_division=0))
    print(classification_report(y_true, y_pred, zero_division=0, labels=[0,1]))


def calc_metrics(y_true, y_pred, outputfile):
    """
    Subfuction of calc_metrics_overall.
    Calculates accuracy, precision, recall and F1.
    :param y_true:
    :param y_pred:
    :param outputfile:
    :return:
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print("{}{:.3f}".format("Accuracy: ", accuracy))
    print("{}{:.3f}".format("Precision: ", precision))
    print("{}{:.3f}".format("Recall: ", recall))
    print("{}{:.3f}".format("F1: ", f1))
    print(classification_report(y_true, y_pred, zero_division=0))
    cl_repo = classification_report(y_true, y_pred, zero_division=0)
    with open(outputfile, "w", encoding="utf-8") as f:
        f.write("Accuracy: " + str(accuracy) + "\n")
        f.write("Precision: " + str(precision) + "\n")
        f.write("Recall: " + str(recall) + "\n")
        f.write(cl_repo)


def calc_metrics_overall(filename, path, gold_label_path):
    """
    Calculates the score globally.
    :param filename: Name of the output file
    :param path: Path to the predictions
    :param gold_label_path: Path to the gold labels
    :return:
    """
    outputfile = path + filename.split(".csv")[0]+"_results.txt"
    gold_label = gold_label_path #"../../preprocess/sick/neg_step2_only_label.csv"
    print(filename)
    predictions = pd.read_csv(path+filename)
    gold = pd.read_csv(gold_label)
    y_pred = predictions["label"].tolist()
    y_true = gold["label"].tolist()
    calc_metrics(y_true, y_pred, outputfile)


def calc_metric_per_signature(filename, path, gold_label_path, signature):
    """
    Calculates the metrics for a defined signature like Neutral.Entailment.Neutral.
    :param filename: Name of the output file
    :param path: Path to the predictions
    :param gold_label_path: Path to the gold labels
    :param signature: Signature which should be evaluated
    :return:
    """
    print(filename)
    outputfile = path + filename.split(".csv")[0]+"_" + signature + "_results.txt"
    gold_label = gold_label_path #"../../preprocess/sick/neg_step2_only_label.csv"
    predictions = pd.read_csv(path+filename)
    gold = pd.read_csv(gold_label)
    gold["index"] = gold.index #range(0, len(gold))
    x_extracted = gold.loc[gold["complete_signature"] == signature] # get rows with signature
    x_extracted_index_list = x_extracted["index"].tolist()
    y_pred_idx = predictions.loc[predictions.index[x_extracted_index_list]] #predictions["label"].tolist()
    y_true = x_extracted["label"].tolist()
    y_pred = y_pred_idx["label"].tolist()

    print(len(y_true))
    print(len(y_pred))
    print(y_true)
    print(y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = classification_report(y_true, y_pred, zero_division=0)
    print("{}{:.3f}".format("Accuracy: ", accuracy))
    print("{}{:.3f}".format("Precision: ", precision))
    print("{}{:.3f}".format("Recall: ", recall))
    print("{}{:.3f}".format("F1: ", f1))
    print(classification_report(y_true, y_pred, zero_division=0))

    with open(outputfile, "w", encoding="utf-8") as f:
        f.write("Accuracy: " + str(accuracy) + "\n")
        f.write("Precision: " + str(precision) + "\n")
        f.write("Recall: " + str(recall) + "\n")
        f.write(cm)

    #print(x)


if __name__ == "__main__":
    signature = "Entailment.Neutral.Neutral"
    path = "../../results/transitivity/step2/"
    gold_label_path_step2 = "../../preprocess/sick/step2_data/complete_text_step2.csv" #step2_only_label.csv" #pos_step2_only_label.csv" #step_3_text_extractions_pos.csv"
    #gold_label_path = "../../preprocess/pos_step3_only_label.csv"
    gold_label_path_step3 = "../../preprocess/common/step_3_text_extractions_common.csv"
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".csv")]
    print(onlyfiles)
    for fi in onlyfiles:
        if "comp" in fi: # comm for step 3
            #calc_metrics_overall(fi, path, gold_label_path_step3)
            calc_metric_per_signature(fi, path, gold_label_path_step2, signature)
    #calc_metrics_step2_pos()





