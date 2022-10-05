"""
Global veridicality evaluation, plots are made for every csv file
"""

import sys
import numpy as np
import pandas as pd
import seaborn as sns
from os import listdir
import matplotlib.pyplot as plt
from os.path import isfile, join
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay









if __name__ == "__main__":
    platform = "cl"
    paths = {"local" :"../data/verb_veridicality_evaluation.tsv", "cl": "/home/students/meier/MA/verb_veridicality/verb_veridicality_evaluation.tsv" }
    f = paths[platform]
    key_pos_or_neg = sys.argv[1]
    path = "/home/students/meier/MA/results/veridicality_results/"
    # get all csv files with pos or neg
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) if f.endswith(".csv") if key_pos_or_neg in f]

    content = []
    with open(f, "r", encoding="utf-8") as f:
        for line in f:
            content.append(line.split("\t"))

    positive = "/home/students/meier/MA/MA_Thesis/preprocess/verb_verid_nor.csv"
    negative = "/home/students/meier/MA/MA_Thesis/preprocess/verb_verid_neg.csv"

    pos_or_neg = {"pos": positive, "neg": negative}
    fil = pos_or_neg[key_pos_or_neg]

    for result in onlyfiles:
        df_true = pd.read_csv(fil, index_col=False)
        #print(df_true.columns)
        df_pred = pd.read_csv(path+result, index_col=False)
        #print(df_true.iloc[12]["label"])
        matrix = confusion_matrix(y_true=df_true["label"].tolist(), y_pred=df_pred["label"].tolist())
        y = matrix.diagonal() / matrix.sum(axis=1)
        y_2 = matrix.diagonal() / matrix.sum(axis=0)
        print("Axis1: ", y)
        print("Axis 0: ", y_2)

        normalised_cm = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(normalised_cm, annot=True, fmt=".3f", xticklabels=["entailment", "neutral", "contradiction"],
                    yticklabels=["entailment", "neutral", "contradiction"])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')


        filename_perc = result.split(".csv")[0] + "_percentage.png"
        # plt.show(block=False)
        plt.savefig(path + filename_perc)

        cmd = ConfusionMatrixDisplay(matrix, display_labels=["entailment", "neutral", "contradiction"])

        cmd.plot()
        # plt.show()
        filename_cm = result.split(".csv")[0] + "_cm.png"
        plt.savefig(path + filename_cm)


