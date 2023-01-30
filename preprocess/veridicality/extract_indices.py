"""
Retrieves the indices per verb signature
"""


def return_indices(content, count, signature):
    index = []
    for num, line in enumerate(content):
        if line[14] == signature:
            index.append(num)
    assert len(index) == count
    print(len(index))
    return index


if __name__ == "__main__":
    f = "../data/verb_veridicality_evaluation.tsv"
    content = []
    with open(f, "r", encoding="utf-8") as f:
        for line in f:
            content.append(line.split("\t"))
    content = content[1:]
    plus_plus = return_indices(content, 212, signature="+/+\n")
    plus_minus = return_indices(content, 100, signature="+/-\n")
    minus_plus = return_indices(content, 25, signature="-/+\n")
    neutral_plus = return_indices(content, 63, signature="o/+\n")
    neutral_minus = return_indices(content, 28, signature="o/-\n")
    minus_neutral = return_indices(content, 55, signature="-/o\n")
    plus_neutral = return_indices(content, 80, signature="+/o\n")
    neutral_neutral = return_indices(content, 935, signature="o/o\n")
