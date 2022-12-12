

import pandas as pd
from datasets import load_dataset

def convert_joint():
    graph_train = "C:/Users/phMei/cluster/MNLI_train_joint_input.csv"
    graph_val = "C:/Users/phMei/cluster/MNLI_dev_matched_joint_input.csv"
    graph_train = pd.read_csv(graph_train)
    graph_val = pd.read_csv(graph_val)
    #graph_train = df_train["label"] == 0
    #graph_val = df_val["label"] == 0
    mnli_train = load_dataset("glue", "mnli", split="train")
    mnli_val = load_dataset("glue", "mnli", split="validation_matched")
    #mnli_train = mnli_train.filter(lambda x: x["label"] == 0)
    #mnli_val = mnli_val.filter(lambda x: x["label"] == 0)
    df_train = mnli_train.to_pandas()
    df_val = mnli_val.to_pandas()

    graph_train["hypothesis"] = df_train["hypothesis"]
    graph_val["hypothesis"] = df_val["hypothesis"]

    graph_train["premise"] = graph_train["premise"].map(lambda x: x + "</g> [EOS]")
    graph_val["premise"] = graph_val["premise"].map(lambda x: x + "</g> [EOS]")

    graph_train["hypothesis"] = graph_train["hypothesis"].map(lambda x: "<t> " + x + " </t> [EOS]")
    graph_val["hypothesis"] = graph_val["hypothesis"].map(lambda x: "<t> " + x + " </t> [EOS]")

    graph_train.drop(graph_train[graph_train.label != 0].index, inplace=True)
    graph_val.drop(graph_val[graph_val.label != 0].index, inplace=True)
    graph_train = graph_train.drop("label", axis=1)
    graph_val = graph_val.drop("label", axis=1)

    graph_train.to_csv("C:/Users/phMei/cluster/MNLI_train_joint_input_generation_hypothesis_is_text.csv", index=False)
    graph_val.to_csv("C:/Users/phMei/cluster/MNLI_dev_matched_joint_input_generation_hypothesis_is_text.csv", index=False)
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
    graph_train = pd.read_csv(graph_train)
    graph_val = pd.read_csv(graph_val)
    #graph_train = df_train["label"] == 0
    #graph_val = df_val["label"] == 0
    mnli_train = load_dataset("glue", "mnli", split="train")
    mnli_val = load_dataset("glue", "mnli", split="validation_matched")
    #mnli_train = mnli_train.filter(lambda x: x["label"] == 0)
    #mnli_val = mnli_val.filter(lambda x: x["label"] == 0)
    df_train = mnli_train.to_pandas()
    df_val = mnli_val.to_pandas()

    graph_train["hypothesis"] = df_train["hypothesis"]
    graph_val["hypothesis"] = df_val["hypothesis"]

    graph_train["premise"] = graph_train["premise"].map(lambda x: x + "</g> [EOS]")
    graph_val["premise"] = graph_val["premise"].map(lambda x: x + "</g> [EOS]")

    graph_train["hypothesis"] = graph_train["hypothesis"].map(lambda x: "<t> " + x + " </t> [EOS]")
    graph_val["hypothesis"] = graph_val["hypothesis"].map(lambda x: "<t> " + x + " </t> [EOS]")



    graph_train.drop(graph_train[graph_train.label != 0].index, inplace=True)
    graph_val.drop(graph_val[graph_val.label != 0].index, inplace=True)
    graph_train = graph_train.drop("label", axis=1)
    graph_val = graph_val.drop("label", axis=1)

    graph_train.to_csv("C:/Users/phMei/cluster/MNLI_train_amr_input_generation_hypothesis_is_text.csv", index=False)
    graph_val.to_csv("C:/Users/phMei/cluster/MNLI_dev_matched_amr_input_generation_hypothesis_is_text.csv", index=False)



def create_text_tags():
    mnli_train = load_dataset("glue", "mnli", split="train")
    mnli_val = load_dataset("glue", "mnli", split="validation_matched")
    mnli_train = mnli_train.filter(lambda x: x["label"] == 0)
    mnli_val = mnli_val.filter(lambda x: x["label"] == 0)
    df_val = mnli_val.to_pandas()
    df_train = mnli_train.to_pandas()
    df_train["premise"] = df_train["premise"].map(lambda x: "<t> " + x + " [EOS]")
    df_train["hypothesis"] = df_train["hypothesis"].map(lambda x:  x + "</t>  [EOS]")
    df_val["premise"] = df_val["premise"].map(lambda x: "<t> " + x + " [EOS]")
    df_val["hypothesis"] = df_val["hypothesis"].map(lambda x: x + " </t> [EOS]")
    df_train = df_train.drop("label", axis=1)
    df_val = df_val.drop("label", axis=1)
    df_train = df_train.drop("idx", axis=1)
    df_val = df_val.drop("idx", axis=1)
    df_train.to_csv("C:/Users/phMei/cluster/MNLI_train_text_tags_input_generation_hypothesis_is_text.csv", index=False)
    df_val.to_csv("C:/Users/phMei/cluster/MNLI_dev_matched_text_tags_input_generation_hypothesis_is_text.csv", index=False)

if __name__ == "__main__":
    path = "C:/Users/phMei/cluster/MNLI_dev_matched_amr.csv"

    mnli_train = load_dataset("glue", "mnli", split="train")
    mnli_val = load_dataset("glue", "mnli", split="validation_matched")

    df_val = mnli_val.to_pandas()
    df_train = mnli_train.to_pandas()
    #print(df_val["hypothesis"])

    df = pd.read_csv(path)
    create_text_tags()
    #convert_graph()
    #convert_joint()

    #convert_graph(df, mnli_val)
