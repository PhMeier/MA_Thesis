from collections import Counter

def read_file(filename):
    di = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            di[line.strip().split("\t")[1]] = float(line.strip().split("\t")[0].replace(",", "."))
    return di


def calc(pos, neg):
    di = {}
    better_or_not = {}
    for key, value in pos.items():
        if abs(pos[key] - neg[key]) <= 5:
            di[key] = "constant"
        if pos[key] == neg[key]:
            print(key, pos[key], neg[key])
            better_or_not[key] = "equal"
        if pos[key] > neg[key]:
            better_or_not[key] = "Pos better"
        if pos[key] < neg[key]:
            better_or_not[key] = "Neg better"
        else:
            di[key] = "not constant"
    return di, better_or_not

if __name__ == "__main__":
    pos = "verb_files/pos_joint_neut_neut.txt"
    neg = "verb_files/neg_joint_neut_neut.txt"
    p = read_file(pos)
    n = read_file(neg)

    res, better_or_not= calc(p, n)
    print(res.values())
    c = Counter(res.values())
    print(c)
    c = Counter(better_or_not.values())
    print(c)
    print(*better_or_not.values(), sep="\n")