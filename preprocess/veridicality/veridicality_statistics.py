import pandas as pd

def verb_statistics(data, signature):
    """
    Show how many instances of a verb for each signature.
    :param data:
    :param signature:
    :return:
    """
    dictionary ={}
    for inst in data:
        if inst[-1] == signature:
            #if inst[2] == "learn":
            #    print(inst)
            key = inst[2] + " " + inst[1]
            if key in dictionary:
                dictionary[key] += 1
            else:
                dictionary[key] = 1
    print("Signature: ", signature)
    print("\n".join("{}\t{}".format(k, v) for k, v in sorted(dictionary.items(), key=lambda x:x[1], reverse=True)))
    print("Count of verbs:\t {}".format(len(dictionary.keys())))
    print("Sum of all:\t {}".format(sum(dictionary.values())))
    return dictionary





if __name__ == "__main__":
    filename = "../../data/verb_veridicality_evaluation.tsv"
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append(line.strip().split("\t"))
    data = data[1:]
    verb_stats = verb_statistics(data, "o/+") #"o/o") #"o/o")

    x = []
    with open("../../utils/verbs_for_excel.txt", "r", encoding="utf-8") as f:
        for line in f:
            x.append(line.strip())
    print(x)
    for verb in x:
        if verb in verb_stats:
            print("{}".format(verb_stats[verb]*3))
        else:
            print(verb)




    #"o/o"
