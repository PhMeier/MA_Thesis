"""
Evaluation procedures for tabulars in Google Sheet


- Identify verbs with best/worst scores
"""

def best_n_verbs(verb_score, model_type):
    print(model_type)
    sorted_verb_score  = {k: v for k, v in sorted(verb_score.items(), key=lambda item: item[1], reverse=True)}
    #print(sorted_verb_score)
    for key, value in verb_score.items():
        if value < 50.00:
            print(key, value)
    #print("\n".join("{}\t{:8.2f}".format(k, v) for k, v in sorted(verb_score.items(), key=lambda x:x[1], reverse=True)))


def read_tab_file(filename):
    verb_score = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line_ = line.strip().split("\t")
            verb = line_[0]
            percentage = line_[-1]
            verb_score[verb] = float(percentage.replace(",", "."))
    return verb_score

if __name__ == "__main__":
    ambart_text_neut_neut = read_tab_file("google_sheets_results/amrbart_text_neut-neut_neg.txt")
    #ambart_graph_neut_neut = read_tab_file("google_sheets_results/amrbart_graph_neut-neut_neg.txt")
    #ambart_joint_neut_neut = read_tab_file("google_sheets_results/amrbart_joint_neut-neut_neg.txt")
    print(ambart_text_neut_neut)
    best_n_verbs(ambart_text_neut_neut, "Text")
    #best_n_verbs(ambart_graph_neut_neut, "Graph")
    #best_n_verbs(ambart_joint_neut_neut, "Joint")