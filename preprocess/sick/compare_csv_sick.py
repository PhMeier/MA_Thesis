"""
Used to find the successfull instances between three csv files for the transitivity task step 1.
"""


import pandas as pd


if __name__ == "__main__":
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


