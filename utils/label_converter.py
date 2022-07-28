"""
Converts numeric labels to entailment, neutral and contradiction
"""
import pandas as pd


if __name__ == "__main__":
    "C:/Users/phMei/Downloads/multinli_0.9_test_matched_sample_submission.csv"
    new_index = [i for i in range(9847, 19643)]
    num_to_label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    df_sample = pd.read_csv("C:/Users/phMei/Downloads/multinli_0.9_test_matched_sample_submission.csv")
    df = pd.read_csv("results_mnli_matched_kaggle_bartLarge.csv")
    df = pd.read_csv("results_mnli_matched_kaggle_amrbart.csv")
    #print(df_sample.head())
    #print(df.head())
    #print(df["0"])
    df["0"] = df["0"].map(num_to_label)
    print(df)
    #df["pairID"] = df_sample["pairID"]

    #df2 = pd.DataFrame({"pairID":df_sample["pairID"], "gold_label": df["0"]})
    df2 = pd.DataFrame({"pairID": new_index, "gold_label": df["0"]})

    print(df2.head())
    #df2.to_csv("final_amrbart_results.csv", index=False)


