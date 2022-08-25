"""
- Gold Labels reinladen
- Predictions reinladen
- Über Indices gewünschte Instanzen betrachten
- Evaluieren

"""


import pandas
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

def return_indices(content, count, signature):
    index = []
    for num, line in enumerate(content):
        if line[14] == signature:
            index.append(num)
    assert len(index) == count
    print(len(index))
    return index


if __name__ == "__main__":
    f = "../data/verb_veridicality_evaluation.tsv"
    content = []
    with open(f, "r", encoding="utf-8") as f:
        for line in f:
            content.append(line.split("\t"))



def get_rows_by_index(data, index):
    labels = []
    for idx in index:
        idx=idx-1
        #print(df_true.iloc[idx])
        labels.append(data.iloc[idx]["label"])
    return labels


if __name__ == "__main__":
    f = "../data/verb_veridicality_evaluation.tsv"
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

    positive = "../preprocess/verb_verid_nor.csv"
    negative = "../preprocess/verb_verid_neg.csv"
    results = "AMRBART_veridicality_pos_graph_only_2277.csv"
    results_neg = "AMRBART_veridicality_neg_graph_only_2277.csv"
    results = "AMRBART_veridicality_nor_results_3036.csv"
    results_neg = "AMRBART_veridicality_neg_results_3036.csv"
    results_neg = "Bart_veridicality_neg_results_15175.csv"
    results = "Bart_veridicality_nor_results_15175.csv"

    results = results
    positive = positive
    content = []
    df_true = pd.read_csv(positive, index_col=False)
    print(df_true.columns)
    df_pred = pd.read_csv(results, index_col=False)
    print(df_true.iloc[12]["label"])
    print(confusion_matrix(y_true=df_true["label"].tolist(),y_pred=df_pred["label"].tolist()))
    matrix = confusion_matrix(y_true=df_true["label"].tolist(), y_pred=df_pred["label"].tolist())
    y =matrix.diagonal()/matrix.sum(axis=0)
    print(y)

    cmd = ConfusionMatrixDisplay(matrix, display_labels=["entailment", "neutral", "contradiction"])
    cmd.plot()
    plt.show()


    #"""
    for key, values in indices_dict.items():
        true_labels = get_rows_by_index(df_true, values)
        #print(true_labels)
        preds = get_rows_by_index(df_pred, values)
        #print(preds)
        #print(len(preds))
        print(key)
        print("Accuracy: {}".format(accuracy_score(true_labels, preds)*100))
        #print("Precision: {}".format(precision_score(true_labels, preds, average="macro") * 100))
        #print("F1 Score: {}".format(f1_score(true_labels, preds, labels=[0, 1, 2], average="micro")*100))
        #print("Recall Score: {}".format(recall_score(true_labels, preds, labels=[0, 1, 2], average="micro") * 100))
        #print("Gold label: \n", true_labels)
        #print("Predictions: \n", preds)
        #print(precision_recall_fscore_support(true_labels, preds))
        print("\n")
    #"""


