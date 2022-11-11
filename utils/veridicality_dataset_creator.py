import pandas as pd


def get_task_and_verb(data):
    task = []
    verb = []
    sent = []
    neg_sent = []
    comp = []
    sig_pos = []
    sig_neg = []
    sig_to_number = {"+":0, "o":1, "-":2}
    for line in data:
        task.append(line[1])
        verb.append(line[2])
        sent.append(line[3])
        neg_sent.append(line[4])
        comp.append(line[5])
        sig_po = sig_to_number[line[-1].split("/")[0]]
        sig_ne = sig_to_number[line[-1].split("/")[1]]
        sig_pos.append(sig_po)
        sig_neg.append(sig_ne)
    return task, verb, sent, neg_sent, comp, sig_pos, sig_neg



if __name__ == "__main__":
    filename = "../data/verb_veridicality_evaluation.tsv"
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append(line.strip().split("\t"))
    data = data[1:]
    task, verb, sent, neg_sent, comp, sig_pos, sig_neg = get_task_and_verb(data)
    data_pos = {"task": task,
            "verb": verb,
            "premise": sent,
            "hypothesis": comp,
            "label": sig_pos}

    data_neg = {"task": task,
            "verb": verb,
            "premise": neg_sent,
            "hypothesis": comp,
            "label": sig_neg}

    df_pos = pd.DataFrame(data_pos)
    df_neg = pd.DataFrame(data_neg)

    df_pos.to_csv("veridicality_pos.csv")
    df_neg.to_csv("veridicality_neg.csv")

