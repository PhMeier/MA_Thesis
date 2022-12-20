
import pandas as pd
from datasets import load_dataset
pd.set_option('display.max_columns', None)
from collections import Counter

def convert_joint2():
    joint_train = "C:/Users/phMei/cluster/MNLI_train_joint_input.csv"
    joint_val = "C:/Users/phMei/cluster/MNLI_dev_matched_joint_input.csv"
    joint_val_mism = "MNLI_dev_mismatched_joint_input.csv"

    text_train = pd.read_csv("C:/Users/phMei/cluster/MNLI_train_text_tags_input_generation_hypothesis_is_text.csv")
    text_val = pd.read_csv("C:/Users/phMei/cluster/MNLI_dev_matched_text_tags_input_generation_hypothesis_is_text.csv")
    text_val_mis = pd.read_csv("C:/Users/phMei/cluster/MNLI_dev_mismatched_text_tags_input_generation_hypothesis_is_text.csv")

    #print(text_val_mis["hypothesis"])
    # lade hier die text sachen rein und ersetze einfach die hypothesen mit text!

    joint_train = pd.read_csv(joint_train)
    joint_val = pd.read_csv(joint_val)
    joint_val_mism = pd.read_csv(joint_val_mism)

    joint_train = joint_train[joint_train.label == 0]
    joint_val = joint_val[joint_val.label == 0]

    filtered_joint_train = joint_train.loc[joint_train["label"] == 0]
    hypos_train = text_train["hypothesis"].tolist()
    prems_train = filtered_joint_train["premise"].tolist()
    prem_tagged_train = [pr + "</g> [EOS]" for pr in prems_train]
    hypo_tagged_train = ["<t> " + hy for hy in hypos_train]
    d = {"premise":prem_tagged_train, "hypothesis": hypo_tagged_train}
    df = pd.DataFrame(d, columns=["premise","hypothesis"])
    df.to_csv("generation_joint_training_data.csv")

    filtered_joint_val_matched = joint_val.loc[joint_val["label"] == 0]
    hypos_val = text_val["hypothesis"].tolist()
    prems_val = filtered_joint_val_matched["premise"].tolist()
    prem_tagged_val = [pr + "</g> [EOS]" for pr in prems_val]
    hypo_tagged_val = ["<t> " + hy for hy in hypos_val]
    d = {"premise": prem_tagged_val, "hypothesis": hypo_tagged_val}
    df = pd.DataFrame(d, columns=["premise","hypothesis"])
    df.to_csv("generation_joint_val_data.csv")


    filtered_joint_mismatched = joint_val_mism.loc[joint_val_mism["label"] == 0]
    hypos_val_mis = text_val_mis["hypothesis"].tolist()
    prems_val_mis = filtered_joint_mismatched["premise"].tolist()
    prem_tagged_val_mis = [pr + "</g> [EOS]" for pr in prems_val_mis]
    hypo_tagged_val_mis = ["<t> " + hy for hy in hypos_val_mis]
    d = {"premise": prem_tagged_val_mis, "hypothesis": hypo_tagged_val_mis}
    df = pd.DataFrame(d, columns=["premise", "hypothesis"])
    df.to_csv("generation_joint_val_mismatched_data.csv")


    """
    print(len(joint_val_mism))
    print(len(text_val_mis))
    print(joint_val_mism.shape)

    joint_train["hypothesis"] = text_train["hypothesis"]
    joint_val["hypothesis"] = text_val["hypothesis"]
    #joint_val_mism["hypothesis"] = text_val_mis["hypothesis"]

    df = pd.DataFrame({"premise":x["premise"], "hypothesis": text_val_mis["hypothesis"]})
    print(df)
    df.to_csv("text.csv")
    """
    #print(joint_val_mism)
    #print(joint_val_mism.values.tolist())

    """
    joint_train = joint_train[joint_train.label==0]
    joint_val = joint_val[joint_val.label==0]
    joint_val_mism = joint_val_mism[joint_val_mism.label==0]
    print(joint_val_mism)
    joint_val_mism["hypothesis"] = joint_val_mism["hypothesis"].str.extract(r'(<t>[\s\w\d.?\/\(\)\[\]\'!\w\-\w\",:+;$%&§€]*<\/t>)')
    #x = joint_val_mism_text_hypo.values.tolist()
    """



def convert_joint():
    graph_train = "C:/Users/phMei/cluster/MNLI_train_joint_input.csv"
    graph_val = "C:/Users/phMei/cluster/MNLI_dev_matched_joint_input.csv"
    graph_val_mism = "MNLI_dev_mismatched_joint_input.csv"
    #graph_train = pd.read_csv(graph_train)
    #graph_val = pd.read_csv(graph_val)
    #graph_val_mism = pd.read_csv(graph_val_mism)

    graph_train = load_dataset("csv", data_files=graph_train)
    graph_val = load_dataset("csv", data_files=graph_val)
    graph_val_mism = load_dataset("csv", data_files=graph_val_mism)

    # graph_train = df_train["label"] == 0
    # graph_val = df_val["label"] == 0
    mnli_train = load_dataset("glue", "mnli", split="train")
    mnli_val = load_dataset("glue", "mnli", split="validation_matched")
    mnli_val_mism = load_dataset("glue", "mnli", split="validation_mismatched")
    mnli_train = mnli_train.filter(lambda x: x["label"] == 0)
    mnli_val = mnli_val.filter(lambda x: x["label"] == 0)
    mnli_val_mism = mnli_val_mism.filter(lambda x: x["label"] == 0)

    graph_train = graph_train.filter(lambda x: x["label"] == 0)
    graph_val = graph_val.filter(lambda x: x["label"] == 0)
    graph_val_mism = graph_val_mism.filter(lambda x: x["label"] == 0)

    graph_train = graph_train.to_pandas()
    graph_val = graph_val.to_pandas()
    graph_val_mism = graph_val_mism.to_pandas()
    df_train = mnli_train.to_pandas()
    df_val = mnli_val.to_pandas()
    df_val_mism = mnli_val_mism.to_pandas()


    print(graph_train.column_names)
    #print(graph_train["train"]["premise"])



    graph_train= graph_train.map(lambda x: x["premise"] + "</g> [EOS]")
    graph_val["train"]["premise"] = graph_val["train"]["premise"].map(lambda x: x + "</g> [EOS]")
    graph_val_mism["train"]["premise"] = graph_val_mism["train"]["premise"].map(lambda x: x + "</g> [EOS]")

    graph_train["train"]["hypothesis"] = df_train["hypothesis"]
    graph_val["train"]["hypothesis"] = df_val["hypothesis"]
    graph_val_mism["train"]["hypothesis"] = df_val_mism["hypothesis"]

    graph_train["train"]["hypothesis"] = graph_train["train"]["hypothesis"].map(lambda x: "<t> " + str(x) + " </t> [EOS]")
    graph_val["train"]["hypothesis"] = graph_val["train"]["hypothesis"].map(lambda x: "<t> " + str(x) + " </t> [EOS]")
    graph_val_mism["train"]["hypothesis"] = graph_val_mism["train"]["hypothesis"].map(lambda x: "<t> " + str(x) + " </t>")

    graph_train.drop(graph_train[graph_train.label != 0].index, inplace=True)
    graph_val.drop(graph_val[graph_val.label != 0].index, inplace=True)
    graph_val_mism.drop(graph_val_mism[graph_val_mism.label != 0].index, inplace=True)

    graph_train = graph_train.drop("label", axis=1)
    graph_val = graph_val.drop("label", axis=1)
    graph_val_mism = graph_val_mism.drop("label", axis=1)

    graph_train.to_csv("C:/Users/phMei/cluster/MNLI_train_joint_input_generation_hypothesis_is_text.csv", index=False)
    graph_val.to_csv("C:/Users/phMei/cluster/MNLI_dev_matched_joint_input_generation_hypothesis_is_text.csv",
                     index=False)
    graph_val_mism.to_csv("C:/Users/phMei/cluster/MNLI_dev_mismatched_joint_input_generation_hypothesis_is_text.csv",
                          index=False)
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
    # graph_train = df_train["label"] == 0
    # graph_val = df_val["label"] == 0
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
    graph_val_mis.to_csv("C:/Users/phMei/cluster/MNLI_dev_mismatched_amr_input_generation_hypothesis_is_text.csv",
                         index=False)


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
    df_train["hypothesis"] = df_train["hypothesis"].map(lambda x: x + "</t>  [EOS]")
    df_val["premise"] = df_val["premise"].map(lambda x: "<t> " + x + " [EOS]")
    df_val["hypothesis"] = df_val["hypothesis"].map(lambda x: x + " </t> [EOS]")
    df_val_mism["premise"] = df_val_mism["premise"].map(lambda x: "<t> " + x + " [EOS]")
    df_val_mism["hypothesis"] = df_val_mism["hypothesis"].map(lambda x: x + " </t>")
    df_train = df_train.drop("label", axis=1)
    df_val = df_val.drop("label", axis=1)
    df_val_mism = df_val_mism.drop("label", axis=1)
    df_train = df_train.drop("idx", axis=1)
    df_val = df_val.drop("idx", axis=1)
    df_val_mism = df_val_mism.drop("idx", axis=1)
    df_train.to_csv("C:/Users/phMei/cluster/MNLI_train_text_tags_input_generation_hypothesis_is_text.csv", index=False)
    df_val.to_csv("C:/Users/phMei/cluster/MNLI_dev_matched_text_tags_input_generation_hypothesis_is_text.csv",
                  index=False)
    df_val_mism.to_csv("C:/Users/phMei/cluster/MNLI_dev_mismatched_text_tags_input_generation_hypothesis_is_text.csv",
                       index=False)


if __name__ == "__main__":
    path = "C:/Users/phMei/cluster/MNLI_dev_matched_amr.csv"

    mnli_train = load_dataset("glue", "mnli", split="train")
    mnli_val = load_dataset("glue", "mnli", split="validation_matched")

    df_val = mnli_val.to_pandas()
    df_train = mnli_train.to_pandas()
    # print(df_val["hypothesis"])

    df = pd.read_csv(path)
    create_text_tags()
    # convert_graph()
    #convert_joint()

    #convert_joint2()

    # convert_graph(df, mnli_val)
