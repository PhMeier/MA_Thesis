"""
Used to find the successfull instances between three csv files for the transitivity task step 1.
"""


import pandas as pd


if __name__ == "__main__":
    # Joint
    df_17_joint = pd.read_csv("step_1_results/sick_amrbart_joint_17.csv")
    df_42_joint = pd.read_csv("step_1_results/sick_amrbart_joint_42.csv")
    df_67_joint = pd.read_csv("step_1_results/sick_amrbart_joint_67.csv")
    # Text
    df_17 = pd.read_csv("step_1_results/sick_amrbart_text_17.csv")
    df_42 = pd.read_csv("step_1_results/sick_amrbart_text_42.csv")
    df_67 = pd.read_csv("step_1_results/sick_amrbart_text_67.csv")
    truth = pd.read_csv("extracted_sick_instances_ground_truth.csv")
    # define the correct instances of 17, 42 and 67 by comparing it to the ground truth
    common_17 = pd.merge(truth, df_17, how="inner")
    common_42 = pd.merge(truth, df_42, how="inner")
    common_67 = pd.merge(truth, df_67, how="inner")

    common_17_joint = pd.merge(truth, df_17_joint, how="inner")
    common_42_joint = pd.merge(truth, df_42_joint, how="inner")
    common_67_joint = pd.merge(truth, df_67_joint, how="inner")

    common_joint_17_42 = pd.merge(common_17_joint, common_42_joint, on=["index"])
    common_17_42 = pd.merge(common_17, common_42, on=["index"])

    #merge = pd.concat([common_17, common_42, common_67])
    merge_text = pd.merge(common_17_42, common_67, on=["index"])
    merge_joint = pd.merge(common_joint_17_42, common_67_joint, on=["index"])

    final_merge = pd.merge(merge_text, merge_joint, on=["index"])
    print(final_merge)

    final_merge = final_merge.drop(columns=["label_x", "label_y", "label_x_x", "label_y_x", "label_x_y"])

    final_merge.to_csv("commonalities_all.csv", index=False)
    """
    df_17 = pd.read_csv("sick_amrbart_text_17.csv")
    df_42 = pd.read_csv("sick_amrbart_text_42.csv")
    df_67 = pd.read_csv("sick_amrbart_text_67.csv")
    truth = pd.read_csv("extracted_sick_instances_ground_truth.csv")
    # define the correct instances of 17, 42 and 67 by comparing it to the ground truth
    common_17 = pd.merge(truth, df_17, how="inner")
    common_42 = pd.merge(truth, df_42, how="inner")
    common_67 = pd.merge(truth, df_67, how="inner")
    common_17_42 = pd.merge(common_17, common_42, on=["index"])

    #merge = pd.concat([common_17, common_42, common_67])
    final_merge = pd.merge(common_17_42, common_67, on=["index"])
    print(final_merge)
    final_merge = final_merge.drop(columns=["label_x", "label_y"])

    final_merge.to_csv("commonalities.csv", index=False)
    """


