import pandas as pd






if __name__ == "__main__":
    df_67 = pd.read_csv("sick_amrbart_text_67.csv")
    truth = pd.read_csv("extracted_sick_instances_ground_truth.csv")
    common = pd.merge(truth, df_67, how="inner")
    print(common)
    common.to_csv("sick/commonalities.csv", index=False)


