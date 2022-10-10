import sys

import pandas
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score, precision_recall_fscore_support, \
    classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd






if __name__ == "__main__":
    # graph and text
    paths = {"graph": "/home/students/meier/MA/data/mnli_amr/MNLI_dev_matched_amr.csv",
             "joint": "/home/students/meier/MA/MA_Thesis/preprocess/MNLI_dev_matched_joint_input.csv"}

    datatype = sys.argv[1]
    predictions = sys.argv[2]

    df = pandas.read_csv(paths[datatype])
    y_true = df["label"].to_list()

    df_pred = pandas.read_csv(predictions)
    y_pred = df_pred["label"].to_list()

    print("Accuracy: {}".format(accuracy_score(y_true, y_pred) * 100))
    print("Precision: {}".format(precision_score(y_true, y_pred, average="macro") * 100))
    print("Recall Score: {}".format(recall_score(y_true, y_pred, labels=[0, 1, 2], average="macro") * 100))
    print("F1 Score: {}".format(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro")*100))


