

import pandas as pd
from datasets import load_dataset

def convert_joint():
    graph_train = "C:/Users/phMei/cluster/MNLI_train_joint_input.csv"
    graph_val = "C:/Users/phMei/cluster/MNLI_dev_matched_joint_input.csv"
    graph_val_mism = "MNLI_dev_mismatched_joint_input.csv"
    graph_train = pd.read_csv(graph_train)
    graph_val = pd.read_csv(graph_val)
    graph_val_mism = pd.read_csv(graph_val_mism)
    #graph_train = df_train["label"] == 0
    #graph_val = df_val["label"] == 0
    mnli_train = load_dataset("glue", "mnli", split="train")
    mnli_val = load_dataset("glue", "mnli", split="validation_matched")
    mnli_val_mism = load_dataset("glue", "mnli", split="validation_mismatched")
    mnli_train = mnli_train.filter(lambda x: x["label"] == 0)
    mnli_val = mnli_val.filter(lambda x: x["label"] == 0)
    mnli_val_mism = mnli_val_mism.filter(lambda x: x["label"] == 0)
    df_train = mnli_train.to_pandas()
    df_val = mnli_val.to_pandas()
    df_val_mism = mnli_val_mism.to_pandas()

    graph_train["hypothesis"] = df_train["hypothesis"]
    graph_val["hypothesis"] = df_val["hypothesis"]
    graph_val_mism["hypothesis"] = df_val_mism["hypothesis"]

    graph_train["premise"] = graph_train["premise"].map(lambda x: x + "</g> [EOS]")
    graph_val["premise"] = graph_val["premise"].map(lambda x: x + "</g> [EOS]")
    graph_val_mism["premise"] = graph_val_mism["premise"].map(lambda x: x + "</g> [EOS]")

    graph_train["hypothesis"] = graph_train["hypothesis"].map(lambda x: "<t> " + str(x) + " </t> [EOS]")
    graph_val["hypothesis"] = graph_val["hypothesis"].map(lambda x: "<t> " + str(x) + " </t> [EOS]")
    graph_val_mism["hypothesis"] = graph_val_mism["hypothesis"].map(lambda x: "<t> " + str(x) + " </t> [EOS]")

    graph_train.drop(graph_train[graph_train.label != 0].index, inplace=True)
    graph_val.drop(graph_val[graph_val.label != 0].index, inplace=True)
    graph_val_mism.drop(graph_val_mism[graph_val_mism.label != 0].index, inplace=True)

    graph_train = graph_train.drop("label", axis=1)
    graph_val = graph_val.drop("label", axis=1)
    graph_val_mism = graph_val_mism.drop("label", axis=1)

    graph_train.to_csv("C:/Users/phMei/cluster/MNLI_train_joint_input_generation_hypothesis_is_text.csv", index=False)
    graph_val.to_csv("C:/Users/phMei/cluster/MNLI_dev_matched_joint_input_generation_hypothesis_is_text.csv", index=False)
    graph_val_mism.to_csv("C:/Users/phMei/cluster/MNLI_dev_mismatched_joint_input_generation_hypothesis_is_text.csv", index=False)
    """
    df["hypothesis"] = val_or_train["hypothesis"]
    df["premise"] = df["premise"].map(lambda x: x + " </g> [EOS]")
    df["hypothesis"] = df["hypothesis"].map(lambda x: "<t> " + x + " </t> [EOS]")
    for x in df["hypothesis"]:
        print(x)
    df.to_csv("C:/Users/phMei/cluster/MNLI_train_matched_joint_input_generation_hypothesis_is_text.csv", index=False)
    """

def convert_graph():
    graph_train = "C:/Users/phMei/cluster/MNLI_train_amr.csv"
    graph_val = "C:/Users/phMei/cluster/MNLI_dev_matched_amr.csv"
    graph_val_mis = "MNLI_dev_mismatched_amr.csv"
    graph_train = pd.read_csv(graph_train)
    graph_val = pd.read_csv(graph_val)
    graph_val_mis = pd.read_csv(graph_val_mis)
    #graph_train = df_train["label"] == 0
    #graph_val = df_val["label"] == 0
    mnli_train = load_dataset("glue", "mnli", split="train")
    mnli_val = load_dataset("glue", "mnli", split="validation_matched")
    mnli_val_mis = load_dataset("glue", "mnli", split="validation_mismatched")

    mnli_train = mnli_train.filter(lambda x: x["label"] == 0)
    mnli_val = mnli_val.filter(lambda x: x["label"] == 0)
    mnli_val_mis = mnli_val_mis.filter(lambda x: x["label"] == 0)

    df_train = mnli_train.to_pandas()
    df_val = mnli_val.to_pandas()
    df_val_mis = mnli_val_mis.to_pandas()

    graph_train["hypothesis"] = df_train["hypothesis"]
    graph_val["hypothesis"] = df_val["hypothesis"]
    graph_val_mis["hypothesis"] = df_val_mis["hypothesis"]

    graph_train["premise"] = graph_train["premise"].map(lambda x: x + "</g> [EOS]")
    graph_val["premise"] = graph_val["premise"].map(lambda x: x + "</g> [EOS]")
    graph_val_mis["premise"] = graph_val_mis["premise"].map(lambda x: x + "</g> [EOS]")

    graph_train["hypothesis"] = graph_train["hypothesis"].map(lambda x: "<t> " + str(x) + " </t> [EOS]")
    graph_val["hypothesis"] = graph_val["hypothesis"].map(lambda x: "<t> " + str(x) + " </t> [EOS]")
    graph_val_mis["hypothesis"] = graph_val_mis["hypothesis"].map(lambda x: "<t> " + str(x) + " </t> [EOS]")

    graph_train.drop(graph_train[graph_train.label != 0].index, inplace=True)
    graph_val.drop(graph_val[graph_val.label != 0].index, inplace=True)
    graph_val_mis.drop(graph_val_mis[graph_val_mis.label != 0].index, inplace=True)
    graph_train = graph_train.drop("label", axis=1)
    graph_val = graph_val.drop("label", axis=1)
    graph_val_mis = graph_val_mis.drop("label", axis=1)

    graph_train.to_csv("C:/Users/phMei/cluster/MNLI_train_amr_input_generation_hypothesis_is_text.csv", index=False)
    graph_val.to_csv("C:/Users/phMei/cluster/MNLI_dev_matched_amr_input_generation_hypothesis_is_text.csv", index=False)
    graph_val_mis.to_csv("C:/Users/phMei/cluster/MNLI_dev_mismatched_amr_input_generation_hypothesis_is_text.csv", index=False)



def create_text_tags():
    mnli_train = load_dataset("glue", "mnli", split="train")
    mnli_val = load_dataset("glue", "mnli", split="validation_matched")
    mnli_val_mism = load_dataset("glue", "mnli", split="validation_mismatched")
    mnli_train = mnli_train.filter(lambda x: x["label"] == 0)
    mnli_val = mnli_val.filter(lambda x: x["label"] == 0)
    mnli_val_mism = mnli_val_mism.filter(lambda x: x["label"] == 0)
    df_val = mnli_val.to_pandas()
    df_train = mnli_train.to_pandas()
    df_val_mism = mnli_val_mism.to_pandas()
    df_train["premise"] = df_train["premise"].map(lambda x: "<t> " + x + " [EOS]")
    df_train["hypothesis"] = df_train["hypothesis"].map(lambda x:  x + "</t>  [EOS]")
    df_val["premise"] = df_val["premise"].map(lambda x: "<t> " + x + " [EOS]")
    df_val["hypothesis"] = df_val["hypothesis"].map(lambda x: x + " </t> [EOS]")
    df_val_mism["premise"] = df_val_mism["premise"].map(lambda x: "<t> " + x + " [EOS]")
    df_val_mism["hypothesis"] = df_val_mism["hypothesis"].map(lambda x: x + " </t> [EOS]")
    df_train = df_train.drop("label", axis=1)
    df_val = df_val.drop("label", axis=1)
    df_val_mism = df_val_mism.drop("label", axis=1)
    df_train = df_train.drop("idx", axis=1)
    df_val = df_val.drop("idx", axis=1)
    df_val_mism = df_val_mism.drop("idx", axis=1)
    df_train.to_csv("C:/Users/phMei/cluster/MNLI_train_text_tags_input_generation_hypothesis_is_text.csv", index=False)
    df_val.to_csv("C:/Users/phMei/cluster/MNLI_dev_matched_text_tags_input_generation_hypothesis_is_text.csv", index=False)
    df_val_mism.to_csv("C:/Users/phMei/cluster/MNLI_dev_mismatched_text_tags_input_generation_hypothesis_is_text.csv", index=False)

if __name__ == "__main__":
    path = "C:/Users/phMei/cluster/MNLI_dev_matched_amr.csv"

    mnli_train = load_dataset("glue", "mnli", split="train")
    mnli_val = load_dataset("glue", "mnli", split="validation_matched")

    df_val = mnli_val.to_pandas()
    df_train = mnli_train.to_pandas()
    #print(df_val["hypothesis"])

    df = pd.read_csv(path)
    #create_text_tags()
    convert_graph()
    convert_joint()

    #convert_graph(df, mnli_val)
