import pandas as pd






if __name__ == "__main__":
    df = pd.read_csv("step2_data/neg_env_complete_sick_new.csv")
    df["f(s1)"] = df["f(s1)"].map(lambda x: "<t> " + x)
    df["s1"] = df["s1"].map(lambda x: x + "</t>")
    df.to_csv("step2_data/neg_env_complete_sick_new_tags.csv", index=True)

    """
    df = pd.read_csv("neg_env_complete_sick_new.csv")
    df["premise"] = df["premise"].map(lambda x: "<t> " + x)
    df["hypothesis"] = df["hypothesis"].map(lambda x: x + "</t>")
    df.to_csv("neg_env_complete_sick_new_tags.csv", index=True)
    """
