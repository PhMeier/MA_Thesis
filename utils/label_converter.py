"""
Converts numeric labels to entailment, neutral and contradiction
"""
import pandas as pd


if __name__ == "__main__":
    "C:/Users/phMei/Downloads/multinli_0.9_test_matched_sample_submission.csv"

    num_to_label = {"0":"entailment", "1":"neutral", "2":"contradiction"}
    df_sample = pd.read_csv("C:/Users/phMei/Downloads/multinli_0.9_test_matched_sample_submission.csv")
    df = pd.read_csv("results_mnli_matched_kaggle_bartLarge.csv")
    print(df.head())
    #df["pairID"] = df_sample["pairID"]

    df2 = pd.DataFrame({"pairID":df_sample["pairID"], "gold_label": df["gold_label"]})

    print(df2.head())
    df2.to_csv("final_bart_large_results.csv", index=False)


