"""
Data preparator to convert data in jsonlines

"""

import jsonlines
import json

def get_text(data, index):
    """
    Retrieve content from a tab separated file through a given index.
    :param data:
    :param index:
    :return:
    """
    sent = []
    for line in data:
        dictio = {}
        dictio["src"] = line[index]
        dictio["tgt"] = ""
        #sent.append(line[index])
        sent.append(dictio)
    return sent


def write_files(data, filename):
    """
    Writes out the data to a given filename.
    Example output:
    {"src": "I am 24 and a mother of a 2 and a half year old.", "tgt": ""}
    :param data:
    :param filename:
    :return:
    """
    with jsonlines.open(filename, "w") as writer:
        writer.write_all(data)

    #json_str = json.dumps(data)
    #print(json_str)
    """
    with jsonlines.open(filename, "w+", encoding="utf-8") as f:
        for line in data:
            f.write() #("{\"src\":"+line+", \tgt\": \\""}")
    """




if __name__ == "__main__":
    data = []
    with open("./data/verb_veridicality_evaluation.tsv", "r", encoding="utf-8") as f:
        for line in f:
            data.append(line.split("\t"))
    #print(data)
    sents = get_text(data[1:], 3)
    #print(len(sents))
    val = sents[:200]
    test = sents[200:400]
    train = sents[400:]
    print(train)
    #print(val[-1])
    #print(test[0])
    write_files(val, "val.jsonl")
    write_files(train, "train.jsonl")
    write_files(test, "test.jsonl")