"""
- Gold Labels reinladen
- Predictions reinladen
- Über Indices gewünschte Instanzen betrachten
- Evaluieren

"""
import numpy as np
import pandas
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import csv




def get_by_indices(sig, key, indices, prediction, gold_label, data):
    """
    Write the missclassified instances out by key and line!
    :param indices:
    :param prediction:
    :param gold_label:
    :param data:
    :return:
    """
    with open("fine_grained_analysis_" + key + ".csv", "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        for i in range(len(prediction)):
            if prediction[i] != gold_label[i]:
                idx = indices[i]
                #print(indices[i])
                if key == "positive":
                    output_string = [data[idx][0], sig, data[idx][3], data[idx][5],str(prediction[i]), str(gold_label[i])]
                    writer.writerow(output_string)
                    #print(data[idx][0], data[idx][3], data[idx][5], data[idx][-1])
                else:
                    output_string = [data[idx][0], sig, data[idx][3], data[idx][4],str(prediction[i]), str(gold_label[i])]
                    writer.writerow(output_string)



def return_indices(content, count, signature):
    index = []
    for num, line in enumerate(content[1:]):
        data_sig = line[14]
        if data_sig == signature:
            index.append(num)
    assert len(index) == count
    print(len(index))
    return index




def get_rows_by_index(data, index):
    labels = []
    for idx in index:
        #idx=idx-1
        #print(df_true.iloc[idx])
        #print(data.iloc[idx])
        #print(data.iloc[idx]["label"])
        labels.append(data.iloc[idx]["label"])
    return labels


if __name__ == "__main__":
    platform = "cl"
    paths = {"local" :"../data/verb_veridicality_evaluation.tsv", "cl": "/home/students/meier/MA/verb_veridicality/verb_veridicality_evaluation.tsv" }
    f = paths[platform]
    results = sys.argv[1]
    key_pos_or_neg = sys.argv[2]
    outputfile = sys.argv[3]

    content = []
    with open(f, "r", encoding="utf-8") as f:
        for line in f:
            content.append(line.split("\t"))

    plus_plus = return_indices(content, 212, signature="+/+\n")
    plus_minus = return_indices(content, 100, signature="+/-\n")
    minus_plus = return_indices(content, 25, signature="-/+\n")
    neutral_plus = return_indices(content, 63, signature="o/+\n")
    neutral_minus = return_indices(content, 28, signature="o/-\n")
    minus_neutral = return_indices(content, 55, signature="-/o\n")
    plus_neutral = return_indices(content, 80, signature="+/o\n")
    neutral_neutral = return_indices(content, 935, signature="o/o\n")

    indices_dict = {"plus_plus": plus_plus, "plus_minus": plus_minus, "minus_plus": minus_plus, "neutraL_plus": neutral_plus,
                    "neutral_minus": neutral_minus, "minus_neutral": minus_neutral, "plus_neutral": plus_neutral,
                    "neutral_neutral": neutral_neutral}

    plus_plus = neutral_neutral


    positive = "/home/students/meier/MA/MA_Thesis/preprocess/verb_verid_nor.csv"
    negative = "/home/students/meier/MA/MA_Thesis/preprocess/verb_verid_neg.csv"

    #positive = "../preprocess/verb_verid_nor.csv"
    #negative = "../preprocess/verb_verid_neg.csv"
    """
    results = "AMRBART_veridicality_pos_graph_only_2277.csv"
    results_neg = "AMRBART_veridicality_neg_graph_only_2277.csv"
    results = "AMRBART_veridicality_nor_results_3036.csv"
    results_neg = "AMRBART_veridicality_neg_results_3036.csv"
    results_neg = "Bart_veridicality_neg_results_15175.csv"
    results = "Bart_veridicality_nor_results_15175.csv"
    """
    pos_or_neg = {"pos": positive, "neg": negative}
    file = pos_or_neg[key_pos_or_neg]
    results = results
    #positive = positive
    #content = []
    df_true = pd.read_csv(file, index_col=False)
    print(df_true.columns)
    df_pred = pd.read_csv(results, index_col=False)
    print(df_true.iloc[12]["label"])



    #print(confusion_matrix(y_true=df_true["label"].tolist(),y_pred=df_pred["label"].tolist()))

    #normalised_cm =

    matrix = confusion_matrix(y_true=df_true["label"].tolist(), y_pred=df_pred["label"].tolist())
    y =matrix.diagonal()/matrix.sum(axis=1)
    y_2 =matrix.diagonal()/matrix.sum(axis=0)
    print("Axis1: ", y)
    print("Axis 0: ", y_2)

    normalised_cm = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(normalised_cm, annot=True, fmt=".3f", xticklabels = ["entailment", "neutral", "contradiction"], yticklabels=["entailment", "neutral", "contradiction"])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    #plt.show(block=False)
    plt.savefig(outputfile + "_confusion_matrix.png")

    cmd = ConfusionMatrixDisplay(matrix, display_labels=["entailment", "neutral", "contradiction"])

    cmd.plot()
    #plt.show()
    plt.savefig(outputfile + "_percentage.png")
    #plt.savefig(outputfile+".png")


    #"""
    header = ["Index", "Signature", "Sentence", "Complement", "Prediction", "Gold Label"]
    f = open(key_pos_or_neg + "_" + outputfile + ".csv", "w+", newline='', encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(header)
    f.close()

    for key, values in indices_dict.items():
        true_labels = get_rows_by_index(df_true, values)
        #print(true_labels)
        preds = get_rows_by_index(df_pred, values)
        #print(preds)
        #print(len(preds))
        print(key)
        print("Accuracy: {}".format(accuracy_score(true_labels, preds)*100))

        #get_by_indices(key, key_pos_or_neg, values, preds, true_labels, content)

        #print("Precision: {}".format(precision_score(true_labels, preds, average="macro") * 100))
        #print("F1 Score: {}".format(f1_score(true_labels, preds, labels=[0, 1, 2], average="micro")*100))
        #print("Recall Score: {}".format(recall_score(true_labels, preds, labels=[0, 1, 2], average="micro") * 100))
        #print("Gold label: \n", true_labels)
        #print("Predictions: \n", preds)
        print(precision_recall_fscore_support(true_labels, preds))
        print("\n")

    #"""


