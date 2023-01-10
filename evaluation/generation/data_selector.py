import random
import pandas as pd



if __name__ == "__main__":
    path = "C:/Users/Meier/Projekte/MA_Thesis/results/generation/ambart_generation_text_17_1.csv"
    print("hjio")

    df = pd.read_csv(path)
    #premise,hypo,generated_hypo
    print(df)
    x = df.loc[df["generated_hypo"].str.endswith(".")==True]
    print(x)
    #x = x.sample(n=50)
    dictio = {"premise":x["premise"], "hypothesis": x["generated_hypo"]}
    df_new = pd.DataFrame(dictio)
    df_new.to_csv("extracted_amrbart_text_17_1.csv")
