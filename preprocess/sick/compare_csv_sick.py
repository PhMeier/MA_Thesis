"""
Used to find the successfull instances between three csv files for the transitivity task step 1 and step 2.

All predictions are considered and compared to the ground truth.
Then the predictions are compared to each other and the intersection of the correct predictions are used to identify
the instances for the next steps (smallest intersecting set).
"""
import pandas as pd


def preparation_for_step2():
    """
    Load results, compare them to ground truth and create the final dataframe for step 2.
    :return: Instances for step 2
    """
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


def preparation_for_step3():
    """
    Load results, compare them to ground truth and create the final dataframe for step 3.
    :return: Instances for step 3
    """
    df_17_joint = pd.read_csv("../../results/transitivity/step2/sick_amrbart_joint_42_step2_comp.csv")
    df_42_joint = pd.read_csv("../../results/transitivity/step2/sick_amrbart_joint_17_step2_comp.csv")
    df_67_joint = pd.read_csv("../../results/transitivity/step2/sick_amrbart_joint_67_step2_comp.csv")
    # neg
    #df_17_joint_neg = pd.read_csv("step_2_results/sick_amrbart_joint_42_step2_neg.csv")
    #df_42_joint_neg = pd.read_csv("step_2_results/sick_amrbart_joint_17_step2_neg.csv")
    #df_67_joint_neg = pd.read_csv("step_2_results/sick_amrbart_joint_67_step2_neg.csv")
    # Text
    df_17 = pd.read_csv("../../results/transitivity/step2/sick_amrbart_text_17_step2_comp.csv")
    df_42 = pd.read_csv("../../results/transitivity/step2/sick_amrbart_text_42_step2_comp.csv")
    df_67 = pd.read_csv("../../results/transitivity/step2/sick_amrbart_text_67_step2_comp.csv")
    # neg
    #df_17_neg = pd.read_csv("step_2_results/sick_amrbart_text_17_step2_neg.csv")
    #df_42_neg = pd.read_csv("step_2_results/sick_amrbart_text_42_step2_neg.csv")
    #df_67_neg = pd.read_csv("step_2_results/sick_amrbart_text_67_step2_neg.csv")

    ground_truth = pd.read_csv("./step2_data/complete_step2_only_label.csv")
    #truth_neg = pd.read_csv("step2_data/neg_step2_only_label.csv")

    common_17_joint_pos = pd.merge(ground_truth, df_17_joint, how="inner")
    common_42_joint_pos = pd.merge(ground_truth, df_42_joint, how="inner")
    common_67_joint_pos = pd.merge(ground_truth, df_67_joint, how="inner")
    common_joint_17_42 = pd.merge(common_17_joint_pos, common_42_joint_pos, on=["index"])
    merge_joint = pd.merge(common_joint_17_42, common_67_joint_pos, on=["index"])

    #common_17_joint_neg = pd.merge(truth_neg, df_17_joint_neg, how="inner")
    #common_42_joint_neg = pd.merge(truth_neg, df_42_joint_neg, how="inner")
    #common_67_joint_neg = pd.merge(truth_neg, df_67_joint_neg, how="inner")
    #common_joint_17_42_neg = pd.merge(common_17_joint_neg, common_42_joint_neg, on=["index"])
    #merge_joint_neg = pd.merge(common_joint_17_42_neg, common_67_joint_neg, on=["index"])

    common_17 = pd.merge(ground_truth, df_17, how="inner")
    common_42 = pd.merge(ground_truth, df_42, how="inner")
    common_67 = pd.merge(ground_truth, df_67, how="inner")
    common_text_17_42 = pd.merge(common_17, common_42, on=["index"])
    merge_text = pd.merge(common_text_17_42, common_67, on=["index"])

    #common_17_neg = pd.merge(truth_neg, df_17_neg, how="inner")
    #common_42_neg = pd.merge(truth_neg, df_42_neg, how="inner")
    #common_67_neg = pd.merge(truth_neg, df_67_neg, how="inner")
    #common_text_17_42_neg = pd.merge(common_17_neg, common_42_neg, on=["index"])
    #merge_text_neg = pd.merge(common_text_17_42_neg, common_67_neg, on=["index"])

    # merge text and joint pos
    final_merge = pd.merge(merge_text, merge_joint, on=["index"])
    #final_merge_neg = pd.merge(merge_text_neg, merge_joint_neg, on=["index"])

    final_merge = final_merge.drop(columns=["label_x", "label_y_y", "label_x_x", "label_y_x", "label_x_y"])
    #final_merge_neg = final_merge_neg.drop(columns=["label_x", "label_y_y", "label_x_x", "label_y_x", "label_x_y"])

    final_merge.to_csv("commonalities_step2_common.csv", index=False)
    #final_merge_neg.to_csv("commonalities_step2_neg.csv", index=False)






if __name__ == "__main__":
    preparation_for_step3()

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


