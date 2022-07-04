"""
Preprocessing for verb_veridicality_evaluation.tsv
Header:
index	task	verb	sentence	neg_sentence	complement
turker_pos_ratings	turker_neg_ratings	bert_pos_entailment_prob	bert_pos_contradiction_prob	bert_pos_neutral_prob
bert_neg_entailment_prob	bert_neg_contradiction_prob	bert_neg_neutral_prob	signature

Interesting is verb, sentence, neg_sentence, complement and signature.
Signature needs to be translated to Entailment, Contradiction and Neutral.
Each verb has 2 signatures, this means each data instance has two "instances" after preprocessing:
- Normal sentence + Label
- Negated Sentence + Label
"""
import pandas as pd


def create_dataframe(data, negated, outputfile):
    """
    Creates a dataframe containing the data from verb_veridicality_evaluation.tsv. Can be used to create a dataframe
    with positive sentences or negated sentences
    :param data: Data containing content from verb_veridicality
    :param negated: If dataframe should contain only negated sentences or not
    :param outputfile: NAme of the outputfile
    :return:
    """

    if negated:
        output_dict = {"neg_sentence": [], "complement": [], "signature": []}
        content = loop_routine(data, output_dict, negated)
    else:
        output_dict = {"sentence":[], "complement":[], "signature":[]}
        content = loop_routine(data, output_dict, negated)
    df = pd.DataFrame(content)
    print(df.head)
    df.to_csv(outputfile, encoding="utf-8")


def loop_routine(data, output_dict, negated):
    """
    Submethod of create_dataframe. Creates a dictionary containing the important columns.
    :param data:
    :return: output dict
    """
    signature_dictionary = {"+": "0", "o": "1", "-": "2", "+\n": "0", "o\n": "1", "-\n": "2"}
    key_index = {"sentence":3, "neg_sentence":4, "complement":5} #, "signature":14}
    for line in data[1:]:
        for key in output_dict.keys():
            if key == "signature":
                output_dict["signature"].append(signature_dictionary[line[14].split("/")[1 if negated else 0]])
            else:
                output_dict[key].append(line[key_index[key]])
    return output_dict


if __name__ == "__main__":
    path = "../data/verb_veridicality_evaluation.tsv"
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(line.split("\t"))
    print(data)
    create_dataframe(data, False, "verb_verid_normal.csv")