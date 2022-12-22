
"""
Changes 29/10
- Added successfull instances

4/11
- Look for specific verb/Fehler anhand der Verben quantifizieren
"""



import pandas
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import csv




def get_by_indices(sig, key, indices, prediction, gold_label, data):
    """
    Write the missclassified instances out by key and line!
    :param indices:
    :param prediction:
    :param gold_label:
    :param data:
    :return:
    """
    with open("fine_grained_analysis_" + key + ".csv", "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        for i in range(len(prediction)):
            if prediction[i] != gold_label[i]:
                idx = indices[i]
                #print(indices[i])
                if key == "positive":
                    output_string = [data[idx][0], sig, data[idx][3], data[idx][5],str(prediction[i]), str(gold_label[i])]
                    writer.writerow(output_string)
                    #print(data[idx][0], data[idx][3], data[idx][5], data[idx][-1])
                else:
                    output_string = [data[idx][0], sig, data[idx][3], data[idx][4],str(prediction[i]), str(gold_label[i])]
                    writer.writerow(output_string)


def return_indices(content, count, signature):
    index = []
    for num, line in enumerate(content[1:]):
        data_sig = line[14]
        if data_sig == signature:
            index.append(line[0])
    print(len(index))
    assert len(index) == count
    return index


def indepth_results_failed(y_true, y_pred, signature_indices):
    """
    Compare the predictions to gold labels. Find the instances which are false (or match).
    :param y_true:
    :param y_pred:
    :param signature_indices: contains the indices of the specified signature
    :return:
    """
    print(signature_indices)
    y_true_labels = y_true["label"].iloc[signature_indices].values.tolist()
    y_pred_labels = y_pred["label"].iloc[signature_indices].values.tolist()
    #print("Gold: ", y_true_labels)
    #print("Preds: ", y_pred_labels)
    # Find the indices of errors
    indices_diff = [i for i in range(len(y_pred_labels)) if y_pred_labels[i] != y_true_labels[i]]
    #print(indices_diff)
    values_indices = [signature_indices[i] for i in indices_diff]
    #print(values_indices)
    failed_instances = y_true.iloc[values_indices].values.tolist()
    #print(*failed_instances, sep="\n")
    indices_diff = set(indices_diff)
    #print("Len preds: ", len(y_pred_labels))
    #print("Len indices diff: ", len(indices_diff))
    return failed_instances, y_pred_labels, indices_diff


def indepth_results_succed(y_true, y_pred, signature_indices):
    """
    Compare the predictions to gold labels. Find the instances which are false (or match).
    :param y_true:
    :param y_pred:
    :param signature_indices:
    :return:
    """
    print(signature_indices)
    y_true_labels = y_true["label"].iloc[signature_indices].values.tolist()
    y_pred_labels = y_pred["label"].iloc[signature_indices].values.tolist()
    #print("Gold: ", y_true_labels)
    #print("Preds: ", y_pred_labels)
    # Find the indices of errors
    indices_equal = [i for i in range(len(y_pred_labels)) if y_pred_labels[i] == y_true_labels[i]]
    #print(indices_diff)
    values_indices = [signature_indices[i] for i in indices_equal]
    #print(values_indices)
    failed_instances = y_true.iloc[values_indices].values.tolist()
    #print(*failed_instances, sep="\n")
    indices_diff = set(indices_equal)
    return failed_instances, y_pred_labels, indices_equal

# 1197
def print_errors_and_predictions(model_type, false_instances, predictions, indices):
    for i,j in zip(false_instances, indices):
        print("{}: {} {}".format(model_type, i, predictions[j]))


if __name__ == "__main__":
    platform = "local"
    paths = {"local": "../data/verb_veridicality_evaluation.tsv", "cl": "/home/students/meier/MA/verb_veridicality/verb_veridicality_evaluation.tsv" }
    f = paths[platform]
    #results = sys.argv[1]
    #key_pos_or_neg = sys.argv[2]
    #outputfile = sys.argv[3]

    content = []
    with open(f, "r", encoding="utf-8") as f:
        for line in f:
            content.append(line.split("\t"))

    plus_plus = return_indices(content, 212, signature="+/+\n")
    plus_minus = return_indices(content, 100, signature="+/-\n")
    minus_plus = return_indices(content, 25, signature="-/+\n")
    neutral_plus = return_indices(content, 63, signature="o/+\n")
    neutral_minus = return_indices(content, 28, signature="o/-\n")
    minus_neutral = return_indices(content, 55, signature="-/o\n")
    plus_neutral = return_indices(content, 80, signature="+/o\n")
    neutral_neutral = return_indices(content, 935, signature="o/o\n") #935

    indices_dict = {"plus_plus": plus_plus, "plus_minus": plus_minus, "minus_plus": minus_plus, "neutral_plus": neutral_plus,
                    "neutral_minus": neutral_minus, "minus_neutral": minus_neutral, "plus_neutral": plus_neutral,
                    "neutral_neutral": neutral_neutral}

    indices_key = "plus_plus"
    positive = "../preprocess/verb_verid_nor.csv"
    negative = "../preprocess/verb_verid_neg.csv"
    key_pos_or_neg = "pos"
    pos_or_neg = {"pos": positive, "neg": negative}
    file = pos_or_neg[key_pos_or_neg]

    gold = pd.read_csv(file)

    # load the results file

    # Positive results

    # Positive results

    # BART
    #"""
    bart_42 = pd.read_csv("../results/veridical/predictions/pos/text/Bart_veridicality_nor_results_15175.csv") #AMRBART_veridicality_pos_text_3036.csv")#"BART_17_verid_neg_3036.csv")
    bart_17 = pd.read_csv("../results/veridical/predictions/pos/text/BART_17_verid_pos_3036.csv") #AMRBART_17_veridicality_pos_text_2277.csv")#"BART_17_verid_neg_3036.csv")
    bart_67 = pd.read_csv("../results/veridical/predictions/pos/text/BART_67_verid_pos_4554.csv")

    # AMRBART Text
    results_42 = pd.read_csv("../results/veridical/predictions/pos/text/amrbart_text_42_tokenizer_pos_3036.csv") #AMRBART_veridicality_pos_text_3036.csv")#"BART_17_verid_neg_3036.csv")
    results_17 = pd.read_csv("../results/veridical/predictions/pos/text/amrbart_text_17_tokenizer_pos_2277.csv") #AMRBART_17_veridicality_pos_text_2277.csv")#"BART_17_verid_neg_3036.csv")
    results_67 = pd.read_csv("../results/veridical/predictions/pos/text/amrbart_text_67_tokenizer_pos_5313.csv") #AMRBART_67_veridicality_pos_text_5313.csv")#"BART_67_verid_neg_4554.csv")#"Bart_veridicality_neg_results_15175.csv")

    # Graph
    results_42_graph_only = pd.read_csv("../results/veridical/predictions/pos/graph/amrbart_graph_42_graph_tokenizer_pos_2277.csv")
    results_17_graph_only = pd.read_csv("../results/veridical/predictions/pos/graph/amrbart_graph_17_graph_tokenizer_pos_3036.csv") #AMRBART_17_veridicality_pos_graph_only_3036.csv")
    results_67_graph_only = pd.read_csv("../results/veridical/predictions/pos/graph/amrbart_graph_67_graph_tokenizer_pos_3795.csv") #AMRBART_67_veridicality_pos_graph_only_3795.csv")

    # Joint
    results_42_joint = pd.read_csv("../results/veridical/predictions/pos/joint/amrbart_joint_42_tokenizer_pos_6072.csv") #AMRBART_verid_joint_pos_7590.csv")
    results_17_joint = pd.read_csv("../results/veridical/predictions/pos/joint/amrbart_joint_17_tokenizer_pos_6072.csv") #AMRBART_17_verid_joint_pos_5313.csv")
    results_67_joint = pd.read_csv("../results/veridical/predictions/pos/joint/amrbart_joint_67_tokenizer_pos_7590.csv") #AMRBART_67_verid_joint_pos_5313.csv")
    #"""
    # Text # neg

    """
    # BART
    bart_42 = pd.read_csv("../results/veridical/predictions/neg/text/Bart_veridicality_neg_results_15175.csv") #AMRBART_veridicality_neg_text_3036.csv")#"BART_17_verid_neg_3036.csv")
    bart_17 = pd.read_csv("../results/veridical/predictions/neg/text/BART_17_verid_neg_3036.csv")#"BART_17_verid_neg_3036.csv")
    bart_67 = pd.read_csv("../results/veridical/predictions/neg/text/BART_67_verid_neg_4554.csv")

    # AMRBART Text
    results_42 = pd.read_csv("../results/veridical/predictions/neg/text/amrbart_text_42_tokenizer_neg_3036.csv") #AMRBART_veridicality_neg_text_3036.csv")#"BART_17_verid_neg_3036.csv")
    results_17 = pd.read_csv("../results/veridical/predictions/neg/text/amrbart_text_17_tokenizer_neg_2277.csv")#"BART_17_verid_neg_3036.csv")
    results_67 = pd.read_csv("../results/veridical/predictions/neg/text/amrbart_text_67_tokenizer_neg_5313.csv")#"BART_67_verid_neg_4554.csv")#"Bart_veridicality_neg_results_15175.csv")

    # Graph
    results_42_graph_only = pd.read_csv("../results/veridical/predictions/neg/graph/amrbart_graph_42_graph_tokenizer_neg_2277.csv")
    results_17_graph_only = pd.read_csv("../results/veridical/predictions/neg/graph/amrbart_graph_17_graph_tokenizer_neg_3036.csv")
    results_67_graph_only = pd.read_csv("../results/veridical/predictions/neg/graph/amrbart_graph_67_graph_tokenizer_neg_3795.csv")

    # Joint
    results_42_joint = pd.read_csv("../results/veridical/predictions/neg/joint/amrbart_joint_42_tokenizer_neg_6072.csv") # AMRBART_verid_joint_neg_7590.csv")
    results_17_joint = pd.read_csv("../results/veridical/predictions/neg/joint/amrbart_joint_17_tokenizer_neg_6072.csv") # AMRBART_17_verid_joint_neg_5313.csv")
    results_67_joint = pd.read_csv("../results/veridical/predictions/neg/joint/amrbart_joint_67_tokenizer_neg_7590.csv") # AMRBART_67_verid_joint_neg_5313.csv")
    """
    #print(results.head())
    #df.rename(index={0:"Index", 1:"label"})

    # BART
    bart_res42_failed, bart_predictions_42, bart_indices_42 = indepth_results_failed(gold, bart_42, indices_dict[indices_key])
    bart_res17_failed, bart_predictions_17, bart_indices_17 = indepth_results_failed(gold, bart_17, indices_dict[indices_key])
    bart_res67_failed, bart_predictions_67, bart_indices_67 = indepth_results_failed(gold, bart_67, indices_dict[indices_key])


    res42_failed, predictions_42, indices_42 = indepth_results_failed(gold, results_42, indices_dict[indices_key])
    res17_failed, predictions_17, indices_17 = indepth_results_failed(gold, results_17, indices_dict[indices_key])
    res67_failed, predictions_67, indices_67 = indepth_results_failed(gold, results_67, indices_dict[indices_key])
    # graph
    res42_failed_graph_only, predictions_42_graph, indices_42_graph = indepth_results_failed(gold, results_42_graph_only, indices_dict[indices_key])
    res17_failed_graph_only, predictions_17_graph, indices_17_graph = indepth_results_failed(gold, results_17_graph_only, indices_dict[indices_key])
    res67_failed_graph_only, predictions_67_graph, indices_67_graph = indepth_results_failed(gold, results_67_graph_only, indices_dict[indices_key])
    # joint
    results_42_joint_failed, predictions_42_joint, indices_42_joint = indepth_results_failed(gold, results_42_joint, indices_dict[indices_key])
    results_17_joint_failed, predictions_17_joint, indices_17_joint = indepth_results_failed(gold, results_17_joint, indices_dict[indices_key])
    results_67_joint_failed, predictions_67_joint, indices_67_joint = indepth_results_failed(gold, results_67_joint, indices_dict[indices_key])

    # successfull instances
    # BART
    bart_res42_succed, bart_predictions_42, bart_indices_42_succed = indepth_results_succed(gold, bart_42, indices_dict[indices_key])
    bart_res17_succed, bart_predictions_17, bart_indices_17_succed = indepth_results_succed(gold, bart_17, indices_dict[indices_key])
    bart_res67_succed, bart_predictions_67, bart_indices_67_succed = indepth_results_succed(gold, bart_67, indices_dict[indices_key])

    res42_succed, predictions_42, indices_42_succed = indepth_results_failed(gold, results_42, indices_dict[indices_key])
    res17_succed, predictions_17, indices_17_succed = indepth_results_failed(gold, results_17, indices_dict[indices_key])
    res67_succed, predictions_67, indices_67_succed = indepth_results_failed(gold, results_67, indices_dict[indices_key])
    # graph
    res42_succed_graph_only, predictions_42_graph, indices_42_graph_succed = indepth_results_failed(gold, results_42_graph_only, indices_dict[indices_key])
    res17_succed_graph_only, predictions_17_graph, indices_17_graph_succed = indepth_results_failed(gold, results_17_graph_only, indices_dict[indices_key])
    res67_succed_graph_only, predictions_67_graph, indices_67_graph_succed = indepth_results_failed(gold, results_67_graph_only, indices_dict[indices_key])
    # joint
    results_42_joint_succed, predictions_42_joint, indices_42_joint_succed = indepth_results_failed(gold, results_42_joint, indices_dict[indices_key])
    results_17_joint_succed, predictions_17_joint, indices_17_joint_succed = indepth_results_failed(gold, results_17_joint, indices_dict[indices_key])
    results_67_joint_succed, predictions_67_joint, indices_67_joint_succed = indepth_results_failed(gold, results_67_joint, indices_dict[indices_key])

    bart_success = bart_res42_succed + bart_res17_succed + bart_res67_succed
    text_success = res42_succed + res17_succed + res67_succed
    graph_sucess = res42_succed_graph_only + res17_succed_graph_only + res67_succed_graph_only
    joint_success = results_42_joint_succed + results_17_joint_succed + results_67_joint_succed

    bart_success_length_prem = sum([len(i[1]) for i in bart_success]) / len(bart_success)
    text_success_length_prem = sum([len(i[1]) for i in text_success]) / len(text_success)
    graph_success_length_prem = sum([len(i[1]) for i in graph_sucess]) / len(graph_sucess)
    joint_success_length_prem = sum([len(i[1]) for i in joint_success]) / len(joint_success)

    bart_success_length_hypo = sum([len(i[2]) for i in bart_success]) / len(bart_success)
    text_success_length_hypo = sum([len(i[2]) for i in text_success]) / len(text_success)
    graph_success_length_hypo = sum([len(i[2]) for i in graph_sucess]) / len(graph_sucess)
    joint_success_length_hypo = sum([len(i[2]) for i in joint_success]) / len(joint_success)

    print(" ---- SUCCESS ----")
    print("Bart successfull premise length: ", round(bart_success_length_prem, 2))
    print("AMRBART Text successfull premise length: ", round(text_success_length_prem, 2))
    print("AMRBART Graph successfull premise length: ", round(graph_success_length_prem, 2))
    print("AMRBART Joint successfull premise length: ",round(joint_success_length_prem, 2))
    print("\n")
    print("BART successfull hypo length :", round(bart_success_length_hypo, 2))
    print("AMRBART Text successfull hypo length: ", round(text_success_length_hypo, 2))
    print("AMRBART Graph successfull hypo length: ",round(graph_success_length_hypo,2))
    print("AMRBART Joint successfull hypo length: ",round(joint_success_length_hypo, 2))


    bart_failed = bart_res42_failed + bart_res17_failed + bart_res67_failed
    text_failed = res42_failed + res17_failed + res67_failed
    graph_failed = res42_failed_graph_only + res17_failed_graph_only + res67_failed_graph_only
    joint_failed = results_42_joint_failed + results_17_joint_failed + results_67_joint_failed

    bart_failed_length_prem = sum([len(i[1]) for i in bart_failed]) / len(bart_failed)
    text_failed_length_prem = sum([len(i[1]) for i in text_failed]) / len(text_failed)
    graph_failed_length_prem = sum([len(i[1]) for i in graph_failed]) / len(graph_failed)
    joint_failed_length_prem = sum([len(i[1]) for i in joint_failed]) / len(joint_failed)

    bart_failed_length_hypo = sum([len(i[2]) for i in bart_failed]) / len(bart_failed)
    text_failed_length_hypo = sum([len(i[2]) for i in text_failed]) / len(text_failed)
    graph_failed_length_hypo = sum([len(i[2]) for i in graph_failed]) / len(graph_failed)
    joint_failed_length_hypo = sum([len(i[2]) for i in joint_failed]) / len(joint_failed)

    print("\n ---- Failed ----")
    print("Bart failed premise length: ", round(bart_failed_length_prem, 2))
    print("AMRBART Text failed premise length: ", round(text_failed_length_prem, 2))
    print("AMRBART Graph failed premise length: ",round(graph_failed_length_prem, 2))
    print("AMRBART Joint failed premise length: ",round(joint_failed_length_prem, 2))
    print("\n")
    print("BART failed hypo length :", round(bart_failed_length_hypo, 2))
    print("AMRBART Text failed hypo length: ", round(text_failed_length_hypo, 2))
    print("AMRBART Graph failed hypo length: ",round(graph_failed_length_hypo,2))
    print("AMRBART Joint failed hypo length: ", round(joint_failed_length_hypo, 2))



    # BART
    bart_res42_failed_tuples = [tuple(lst) for lst in bart_res42_failed]
    bart_res17_failed_tuples = [tuple(lst) for lst in bart_res17_failed]
    bart_res67_failed_tuples = [tuple(lst) for lst in bart_res67_failed]

    res42_failed_tuples = [tuple(lst) for lst in res42_failed]
    res17_failed_tuples = [tuple(lst) for lst in res17_failed]
    res67_failed_tuples = [tuple(lst) for lst in res67_failed]
    # graph
    res42_failed_tuples_graph_only = [tuple(lst) for lst in res42_failed_graph_only]
    res17_failed_tuples_graph_only = [tuple(lst) for lst in res17_failed_graph_only]
    res67_failed_tuples_graph_only = [tuple(lst) for lst in res67_failed_graph_only]
    # joint
    res42_failed_tuples_joint = [tuple(lst) for lst in results_42_joint_failed]
    res17_failed_tuples_joint = [tuple(lst) for lst in results_17_joint_failed]
    res67_failed_tuples_joint = [tuple(lst) for lst in results_67_joint_failed]

    # success
    # BART
    bart_res42_success_tuples = [tuple(lst) for lst in bart_res42_succed]
    bart_res17_success_tuples = [tuple(lst) for lst in bart_res17_succed]
    bart_res67_success_tuples = [tuple(lst) for lst in bart_res67_succed]

    res42_success_tuples = [tuple(lst) for lst in res42_succed]
    res17_success_tuples = [tuple(lst) for lst in res17_succed]
    res67_success_tuples = [tuple(lst) for lst in res67_succed]
    # graph
    res42_success_tuples_graph_only = [tuple(lst) for lst in res42_succed_graph_only]
    res17_success_tuples_graph_only = [tuple(lst) for lst in res17_succed_graph_only]
    res67_success_tuples_graph_only = [tuple(lst) for lst in res67_succed_graph_only]
    # joint
    res42_success_tuples_joint = [tuple(lst) for lst in results_42_joint_succed]
    res17_success_tuples_joint = [tuple(lst) for lst in results_17_joint_succed]
    res67_success_tuples_joint = [tuple(lst) for lst in results_67_joint_succed]


    # BART
    bart_res42_failed_set = set(bart_res42_failed_tuples)
    bart_res17_failed_set = set(bart_res17_failed_tuples)
    bart_res67_failed_set = set(bart_res67_failed_tuples)

    res42_failed_set = set(res42_failed_tuples)
    res17_failed_set = set(res17_failed_tuples)
    res67_failed_set = set(res67_failed_tuples)
    # graph
    res42_failed_set_graph_only = set(res42_failed_tuples_graph_only)
    res17_failed_set_graph_only = set(res17_failed_tuples_graph_only)
    res67_failed_set_graph_only = set(res67_failed_tuples_graph_only)
    # joint
    res42_failed_set_joint = set(res42_failed_tuples_joint)
    res17_failed_set_joint = set(res17_failed_tuples_joint)
    res67_failed_set_joint = set(res67_failed_tuples_joint)

    # succeess
    # BART
    res42_correct_set = set(bart_res42_success_tuples)
    res17_correct_set = set(bart_res17_success_tuples)
    res67_correct_set = set(bart_res67_success_tuples)
    # amrbart text
    res42_correct_set = set(res42_success_tuples)
    res17_correct_set = set(res17_success_tuples)
    res67_correct_set = set(res67_success_tuples)
    # graph
    res42_correct_set_graph_only = set(res42_success_tuples_graph_only)
    res17_correct_set_graph_only = set(res17_success_tuples_graph_only)
    res67_correct_set_graph_only = set(res67_success_tuples_graph_only)
    # joint
    res42_correct_set_joint = set(res42_success_tuples_joint)
    res17_correct_set_joint = set(res17_success_tuples_joint)
    res67_correct_set_joint = set(res67_success_tuples_joint)

    indices_bart = bart_indices_42.intersection(bart_indices_17, bart_indices_67)
    indices_text = indices_42.intersection(indices_17, indices_67)
    indices_graph = indices_42_graph.intersection(indices_17_graph, indices_67_graph)
    indices_joint = indices_42_joint.intersection(indices_17_joint, indices_67_joint)

    unified_predictions_bart = [[i,j,k] for i,j,k in zip(bart_predictions_42, bart_predictions_17, bart_predictions_67)]
    print("unified preds: ", unified_predictions_bart)


    print("\n")

    print("Length of errors sets (Bart): {}, {}, {} Sum: {}".format(len(bart_res42_failed_set), len(bart_res17_failed_set),
                                                                       len(bart_res67_failed_set), len(bart_res42_failed_set) +len(bart_res17_failed_set)+ len(bart_res67_failed_set)))


    print("\n Length of errors sets (text): {}, {}, {} Sum: {}".format(len(res42_failed_set), len(res17_failed_set),
                                                                       len(res67_failed_set), len(res42_failed_set) +len(res17_failed_set)+ len(res67_failed_set)))

    print("\n Length of errors sets (graph): {}, {}, {} Sum: {}".format(len(res42_failed_set_graph_only),
                                                        len(res17_failed_set_graph_only),
                                                        len(res67_failed_set_graph_only), len(res42_failed_set_graph_only)+len(res17_failed_set_graph_only)+len(res67_failed_set_graph_only)))

    print("\n Length of errors sets (joint): {}, {}, {} Sum: {}".format(len(res42_failed_set_joint),
                                                        len(res17_failed_set_joint),
                                                        len(res67_failed_set_joint), len(res42_failed_set_joint)+len(res17_failed_set_joint)+len(res67_failed_set_joint)))

    print("\n")
    print_errors_and_predictions("Common Errors Bart", bart_res42_failed_set.intersection(bart_res17_failed_set, bart_res67_failed_set), unified_predictions_bart, indices_bart)
    print("\n")
    print_errors_and_predictions("Common Errors AMRBART Text",
                                 res42_failed_set.intersection(res17_failed_set, res67_failed_set),
                                 unified_predictions_bart, indices_bart)
    print("\n")
    print_errors_and_predictions("Common Errors AMRBART Graph",
                                 res42_failed_set_graph_only.intersection(res17_failed_set_graph_only, res67_failed_set_graph_only),
                                 unified_predictions_bart, indices_bart)
    print("\n")
    print_errors_and_predictions("Common Errors AMRBART Joint",
                                 res42_correct_set_joint.intersection(res17_correct_set_joint, res67_correct_set_joint),
                                 unified_predictions_bart, indices_bart)
    #"""
    print("\n Common Errors BART: \n", *bart_res42_failed_set.intersection(bart_res17_failed_set, bart_res67_failed_set), sep="\n")
    print("\n Common Errors Text: \n", *res42_failed_set.intersection(res17_failed_set, res67_failed_set), sep="\n")
    print("\n Common Errors Graph: \n", *res42_failed_set_graph_only.intersection(res17_failed_set_graph_only, res67_failed_set_graph_only), sep="\n")

    print("\n Correct Instances Joint: \n",
          *res42_correct_set_joint.intersection(res17_correct_set_joint, res67_correct_set_joint), sep="\n")

    #print("\n Common Errors Joint: \n", *res42_failed_set_joint.intersection(res17_failed_set_joint, res67_failed_set_joint), sep="\n")

    """

    model_set_bart = bart_res42_failed_set.intersection(bart_res17_failed_set, bart_res67_failed_set)
    model_set_1 = res42_failed_set.intersection(res17_failed_set, res67_failed_set)
    model_set_2 = res42_failed_set_graph_only.intersection(res17_failed_set_graph_only, res67_failed_set_graph_only)
    model_set_3 = res42_failed_set_joint.intersection(res17_failed_set_joint, res67_failed_set_joint)

    intersection_between_models = model_set_bart & model_set_1 & model_set_2 & model_set_3 # model_set_1.intersection(model_set_2)
    intersection_between_indices = indices_bart & indices_text & indices_graph & indices_joint

    print(len(intersection_between_models), len(intersection_between_indices))
    #print("\n Intersection of common errors: \n", *intersection_between_models, sep="\n")
    for instance, idx in zip(intersection_between_models, intersection_between_indices):
        print(instance, idx)
        print("Bart 42: ", bart_predictions_42[idx])
        print("Bart 17: ", bart_predictions_17[idx])
        print("Bart 67: ", bart_predictions_67[idx])

        print("Text 42: ", predictions_42[idx])
        print("Text 17: ", predictions_17[idx])
        print("Text 67: ", predictions_67[idx])

        print("Graph 42: ", predictions_42_graph[idx])
        print("Graph 17: ", predictions_17_graph[idx])
        print("Graph 67: ", predictions_67_graph[idx])

        print("Joint 42: ", predictions_42_joint[idx])
        print("Joint 17: ", predictions_17_joint[idx])
        print("Joint 67: ", predictions_67_joint[idx])


    print("Text 42: ", predictions_42)
    print("Text 17: ", predictions_17)
    print("Text 67: ", predictions_67)


    print("Graph 42 ", predictions_42_graph)
    print("Graph 17 ",predictions_17_graph)
    print("Graph 67 ",predictions_67_graph)

    print("Joint 42 ",predictions_42_joint)
    print("Joint 17 ",predictions_17_joint)
    print("Joint 67 ",predictions_67_joint)
    #"""







