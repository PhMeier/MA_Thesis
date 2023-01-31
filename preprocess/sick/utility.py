"""
Utility functions for transitivity classification.
"""

import pandas as pd


def add_tags_to_sick(filename):
    """

    :param filename: e.g step2_data/neg_env_complete_sick_new
    :return:
    """
    df = pd.read_csv(filename)
    df["f(s1)"] = df["f(s1)"].map(lambda x: "<t> " + x)
    df["s1"] = df["s1"].map(lambda x: x + "</t>")
    df.to_csv(filename+"_new_tags.csv", index=True)


def get_step3_instances():
    """
    Utility script to take the commonalities from step 2 as inference data for transitivity classification (step 3).
    """
    data = pd.read_csv("commonalities_step2_common.csv")
    idx = data["index"].to_list()
    joint = pd.read_csv("complete_joint_step3.csv")
    text = pd.read_csv("../common/complete_text_step3.csv")
    joint_ex = joint.loc[idx]
    text_ex = text.loc[idx]
    joint_ex.to_csv("step_3_joint_extractions_common.csv", index=False)
    text_ex.to_csv("step_3_text_extractions_common.csv", index=False)


if __name__ == "__main__":



