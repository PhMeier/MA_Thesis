
"""
- Read in results (Pos or neg)
- Get True labels
- Calculate

"""
import sys

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score, precision_recall_fscore_support, \
    classification_report, accuracy_score

if __name__ == "__main__":

    platform = "local"
    results = sys.argv[1]
    key_pos_or_neg = sys.argv[2]
    abc = "/home/students/meier/MA/MA_Thesis"
    positive = "../preprocess/verb_verid_nor.csv"
    negative = "../preprocess/verb_verid_neg.csv"

    pos_or_neg = {"positive": positive, "negative": negative}
    file = pos_or_neg[key_pos_or_neg]

    df_true = pd.read_csv(file, index_col=False)
    print(df_true.columns)
    df_pred = pd.read_csv(results, index_col=False)
    print(df_true.iloc[12]["label"])
    print(confusion_matrix(y_true=df_true["label"].tolist(),y_pred=df_pred["label"].tolist()))
    matrix = confusion_matrix(y_true=df_true["label"].tolist(), y_pred=df_pred["label"].tolist())
    y =matrix.diagonal()/matrix.sum(axis=1)
    y_2 =matrix.diagonal()/matrix.sum(axis=0)
    y =matrix.diagonal()/matrix.sum(axis=1)
    y_2 =matrix.diagonal()/matrix.sum(axis=0)
    print("Axis1: ", y)
    print("Axis 0: ", y_2)

    y_true = df_true["label"].tolist()
    y_pred = df_pred["label"].tolist()

    print("Accuracy: {}".format(accuracy_score(y_true, y_pred) * 100))
    print("Precision: {}".format(precision_score(y_true, y_pred, average="macro") * 100))
    print("Recall Score: {}".format(recall_score(y_true, y_pred, labels=[0, 1, 2], average="macro") * 100))
    print("F1 Score: {}".format(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro")*100))
    #print("Gold label: \n", y_true)
    #print("Predictions: \n", y_pred)
    print(classification_report(y_true, y_pred, target_names=["entailment", "neutral", "contradiction"]))
    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average="macro") * 100
    rec = recall_score(y_true, y_pred, labels=[0, 1, 2], average="macro") * 100
    f1= f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro")*100
    cl_rep = classification_report(y_true, y_pred, target_names=["entailment", "neutral", "contradiction"])

    outputfile = results.split(".csv")[0]

    with open(outputfile+".txt", "w+", encoding="utf-8") as f:
        f.write("Accuracy: " + str(acc) + "\n")
        f.write("Precision: "+ str(prec) + "\n")
        f.write("Recall: " + str(rec) + "\n")
        f.write("F1 Score: "+ str(f1) + "\n")
        f.write("Classification Report \n")
        f.write(cl_rep + "\n")


