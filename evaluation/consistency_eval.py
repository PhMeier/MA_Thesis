
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
    :param signature_indices:
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

def count_verbs(data):
    dictionary = {}
    for line in data:
        key = line[2] + " " + line[1]
        if key in dictionary:
            dictionary[key] += 1
        else:
            dictionary[key] = 1
    return dictionary


def get_results_for_specific_verb(verb, to_or_that, data):
    """
    Returns verb specific instances.
    :param verb: Verb which is queried
    :param to_or_that: Either "to" or "that" which is queried additionally
    :param data: Failed or correct instances
    :return:
    """
    verb_specific_instances = []
    for instance in data:
        #print(instance)
        if instance[1] == to_or_that and instance[2] == verb:
            verb_specific_instances.append(instance)
    return verb_specific_instances


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

    indices_key = "neutral_neutral"
    positive = "../utils/veridicality_pos.csv"
    negative = "../utils/veridicality_neg.csv"
    key_pos_or_neg = "pos"
    pos_or_neg = {"pos": positive, "neg": negative}
    file = pos_or_neg[key_pos_or_neg]

    gold = pd.read_csv(file)

    # load the results file

    # Positive results
    #"""
    results_42 = pd.read_csv("../results/veridical/predictions/pos/text/amrbart_text_42_corrected_pos_3036.csv") #AMRBART_veridicality_pos_text_3036.csv")#"BART_17_verid_neg_3036.csv")
    results_17 = pd.read_csv("../results/veridical/predictions/pos/text/amrbart_text_17_corrected_pos_2277.csv") #AMRBART_17_veridicality_pos_text_2277.csv")#"BART_17_verid_neg_3036.csv")
    results_67 = pd.read_csv("../results/veridical/predictions/pos/text/amrbart_text_67_corrected_pos_5313.csv") #AMRBART_67_veridicality_pos_text_5313.csv")#"BART_67_verid_neg_4554.csv")#"Bart_veridicality_neg_results_15175.csv")

    # Graph
    results_42_graph_only = pd.read_csv("../results/veridical/predictions/pos/graph/AMRBART_veridicality_pos_graph_only_2277.csv")
    results_17_graph_only = pd.read_csv("../results/veridical/predictions/pos/graph/amrbart_17_graph_veridical_pos_3036.csv") #AMRBART_17_veridicality_pos_graph_only_3036.csv")
    results_67_graph_only = pd.read_csv("../results/veridical/predictions/pos/graph/amrbart_67_graph_veridical_pos_3795.csv") #AMRBART_67_veridicality_pos_graph_only_3795.csv")

    # Joint
    results_42_joint = pd.read_csv("../results/veridical/predictions/pos/joint/amrbart_42_joint_ft_pos_6072.csv") #AMRBART_verid_joint_pos_7590.csv")
    results_17_joint = pd.read_csv("../results/veridical/predictions/pos/joint/amrbart_67_joint_ft_pos_7590.csv") #AMRBART_17_verid_joint_pos_5313.csv")
    results_67_joint = pd.read_csv("../results/veridical/predictions/pos/joint/amrbart_17_joint_ft_pos_6072.csv") #AMRBART_67_verid_joint_pos_5313.csv")
    #"""

    # Text # neg
    """
    results_42 = pd.read_csv("../results/veridical/predictions/neg/text/amrbart_text_42_corrected_neg_3036.csv") #AMRBART_veridicality_neg_text_3036.csv")#"BART_17_verid_neg_3036.csv")
    results_17 = pd.read_csv("../results/veridical/predictions/neg/text/amrbart_text_17_corrected_neg_2277.csv")#"BART_17_verid_neg_3036.csv")
    results_67 = pd.read_csv("../results/veridical/predictions/neg/text/amrbart_text_67_corrected_neg_5313.csv")#"BART_67_verid_neg_4554.csv")#"Bart_veridicality_neg_results_15175.csv")

    # Graph
    results_42_graph_only = pd.read_csv("../results/veridical/predictions/neg/graph/AMRBART_veridicality_neg_graph_only_2277.csv")
    results_17_graph_only = pd.read_csv("../results/veridical/predictions/neg/graph/AMRBART_17_veridicality_neg_graph_only_3036.csv")
    results_67_graph_only = pd.read_csv("../results/veridical/predictions/neg/graph/AMRBART_67_veridicality_neg_graph_only_3795.csv")

    # Joint
    results_42_joint = pd.read_csv("../results/veridical/predictions/neg/joint/amrbart_42_joint_ft_neg_6072.csv") # AMRBART_verid_joint_neg_7590.csv")
    results_17_joint = pd.read_csv("../results/veridical/predictions/neg/joint/amrbart_67_joint_ft_neg_7590.csv") # AMRBART_17_verid_joint_neg_5313.csv")
    results_67_joint = pd.read_csv("../results/veridical/predictions/neg/joint/amrbart_17_joint_ft_neg_6072.csv") # AMRBART_67_verid_joint_neg_5313.csv")
    """
    #print(results.head())
    #df.rename(index={0:"Index", 1:"label"})

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
    res42_succed, predictions_42, indices_42_succed = indepth_results_succed(gold, results_42, indices_dict[indices_key])
    res17_succed, predictions_17, indices_17_succed = indepth_results_succed(gold, results_17, indices_dict[indices_key])
    res67_succed, predictions_67, indices_67_succed = indepth_results_succed(gold, results_67, indices_dict[indices_key])
    # graph
    res42_succed_graph_only, predictions_42_graph, indices_42_graph_succed = indepth_results_succed(gold, results_42_graph_only, indices_dict[indices_key])
    res17_succed_graph_only, predictions_17_graph, indices_17_graph_succed = indepth_results_succed(gold, results_17_graph_only, indices_dict[indices_key])
    res67_succed_graph_only, predictions_67_graph, indices_67_graph_succed = indepth_results_succed(gold, results_67_graph_only, indices_dict[indices_key])
    # joint
    results_42_joint_succed, predictions_42_joint, indices_42_joint_succed = indepth_results_succed(gold, results_42_joint, indices_dict[indices_key])
    results_17_joint_succed, predictions_17_joint, indices_17_joint_succed = indepth_results_succed(gold, results_17_joint, indices_dict[indices_key])
    results_67_joint_succed, predictions_67_joint, indices_67_joint_succed = indepth_results_succed(gold, results_67_joint, indices_dict[indices_key])


    text_failed = res42_failed + res17_failed + res67_failed
    graph_failed = res42_failed_graph_only + res17_failed_graph_only + res67_failed_graph_only
    joint_failed = results_42_joint_failed + results_17_joint_failed + results_67_joint_failed

    c_text = count_verbs(text_failed)
    c_graph = count_verbs(graph_failed)
    c_joint = count_verbs(joint_failed)
    # for google sheets
    print("Text")
    print("\n".join("{}\t{}".format(k, v) for k, v in sorted(c_text.items(), key=lambda x:x[1], reverse=True)))
    print("\nGraph")
    print("\n".join("{}\t{}".format(k, v) for k, v in sorted(c_graph.items(), key=lambda x:x[1], reverse=True)))
    print("\nJoint")
    print("\n".join("{}\t{}".format(k, v) for k, v in sorted(c_joint.items(), key=lambda x:x[1], reverse=True)))



    """
    query_verb = "mean"
    query_aux = "to"
    x = get_results_for_specific_verb(query_verb, query_aux, results_42_joint_failed)
    print("\n{} {} {} Length: {}".format(query_verb, query_aux, "42", len(x)), *x, sep="\n")

    x = get_results_for_specific_verb(query_verb, query_aux, results_17_joint_failed)
    print("\n{} {} {} Length: {}".format(query_verb, query_aux, "17", len(x)), *x, sep="\n")

    x = get_results_for_specific_verb(query_verb, query_aux, results_67_joint_failed)
    print("\n{} {} {} Length: {}".format(query_verb, query_aux, "67", len(x)), *x, sep="\n")
    """


    # convert list to tuples, otherwise no set operation is possible
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














