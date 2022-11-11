"""
Compare the included verbs in the veridical dataset to the list of verbs in the paper. Find
the missign verbs.
"""
import pandas as pd


def read_file(filename):
    content = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            content.append(line.split()[0].strip())
    return content

if __name__ == "__main__":
    paper_verbs = read_file("../data/veridical_verbs.txt")
    df = pd.read_csv("../data/verb_veridicality_evaluation.tsv", sep="\t")
    dataset_verbs = df["verb"].to_list()

    paper_verbs_set = set(paper_verbs)
    dataset_verbs_set = set(dataset_verbs)

    diff = list(dataset_verbs_set - paper_verbs_set)
    print(diff)
    x = df.loc[df["verb"] == "have"]
    print(x["task"])
    print(len(x["task"].to_list()))

    print(len(dataset_verbs_set))
    print(len(paper_verbs_set))
    print(dataset_verbs)
    print(len(dataset_verbs))