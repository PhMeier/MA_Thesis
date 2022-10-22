
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
    #print(len(index))
    assert len(index) == count
    return index


def indepth_results(y_true, y_pred, signature_indices):
    """
    Compare the predictions to gold labels. Find the instances which are false (or match).
    :param y_true:
    :param y_pred:
    :param signature_indices:
    :return:
    """
    #print(signature_indices)
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

# 1197




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
    positive = "../preprocess/verb_verid_nor.csv"
    negative = "../preprocess/verb_verid_neg.csv"
    key_pos_or_neg = "neg"
    pos_or_neg = {"pos": positive, "neg": negative}
    file = pos_or_neg[key_pos_or_neg]

    gold = pd.read_csv(file)

    # load the results file

    # Positive results
    """
    results_42 = pd.read_csv("../results/veridical/AMRBART_veridicality_pos_text_3036.csv")#"BART_17_verid_neg_3036.csv")
    results_17 = pd.read_csv("../results/veridical/AMRBART_17_veridicality_pos_text_2277.csv")#"BART_17_verid_neg_3036.csv")
    results_67 = pd.read_csv("../results/veridical/AMRBART_67_veridicality_pos_text_5313.csv")#"BART_67_verid_neg_4554.csv")#"Bart_veridicality_neg_results_15175.csv")

    # Graph
    results_42_graph_only = pd.read_csv("../results/veridical/AMRBART_veridicality_pos_graph_only_2277.csv")
    results_17_graph_only = pd.read_csv("../results/veridical/AMRBART_17_veridicality_pos_graph_only_3036.csv")
    results_67_graph_only = pd.read_csv("../results/veridical/AMRBART_67_veridicality_pos_graph_only_3795.csv")

    # Joint
    results_42_joint = pd.read_csv("../results/veridical/amrbart_42_joint_ft_pos_6072.csv") #AMRBART_verid_joint_pos_7590.csv")
    results_17_joint = pd.read_csv("../results/veridical/amrbart_17_joint_ft_pos_6072.csv") #AMRBART_17_verid_joint_pos_5313.csv")
    results_67_joint = pd.read_csv("../results/veridical/amrbart_67_joint_ft_pos_7590.csv") #AMRBART_67_verid_joint_pos_5313.csv")
    """


    # Text # neg

    results_42 = pd.read_csv("../results/veridical/AMRBART_veridicality_neg_text_3036.csv")#"BART_17_verid_neg_3036.csv")
    results_17 = pd.read_csv("../results/veridical/AMRBART_17_veridicality_neg_text_2277.csv")#"BART_17_verid_neg_3036.csv")
    results_67 = pd.read_csv("../results/veridical/AMRBART_67_veridicality_neg_text_5313.csv")#"BART_67_verid_neg_4554.csv")#"Bart_veridicality_neg_results_15175.csv")

    # Graph
    results_42_graph_only = pd.read_csv("../results/veridical/AMRBART_veridicality_neg_graph_only_2277.csv")
    results_17_graph_only = pd.read_csv("../results/veridical/AMRBART_17_veridicality_neg_graph_only_3036.csv")
    results_67_graph_only = pd.read_csv("../results/veridical/AMRBART_67_veridicality_neg_graph_only_3795.csv")

    # Joint
    results_42_joint = pd.read_csv("../results/veridical/amrbart_42_joint_ft_neg_6072.csv") # AMRBART_verid_joint_neg_7590.csv")
    results_17_joint = pd.read_csv("../results/veridical/amrbart_17_joint_ft_neg_6072.csv") # AMRBART_17_verid_joint_neg_5313.csv") ("../results/veridical/amrbart_67_joint_ft_neg_7590.csv") 
    results_67_joint = pd.read_csv("../results/veridical/amrbart_67_joint_ft_neg_7590.csv") # AMRBART_67_verid_joint_neg_5313.csv")

    #print(results.head())
    #df.rename(index={0:"Index", 1:"label"})

    res42_failed, predictions_42, indices_42 = indepth_results(gold, results_42, indices_dict[indices_key])
    res17_failed, predictions_17, indices_17 = indepth_results(gold, results_17, indices_dict[indices_key])
    res67_failed, predictions_67, indices_67 = indepth_results(gold, results_67, indices_dict[indices_key])
    # graph
    res42_failed_graph_only, predictions_42_graph, indices_42_graph = indepth_results(gold, results_42_graph_only, indices_dict[indices_key])
    res17_failed_graph_only, predictions_17_graph, indices_17_graph = indepth_results(gold, results_17_graph_only, indices_dict[indices_key])
    res67_failed_graph_only, predictions_67_graph, indices_67_graph = indepth_results(gold, results_67_graph_only, indices_dict[indices_key])
    # joint
    results_42_joint_failed, predictions_42_joint, indices_42_joint = indepth_results(gold, results_42_joint, indices_dict[indices_key])
    results_17_joint_failed, predictions_17_joint, indices_17_joint = indepth_results(gold, results_17_joint, indices_dict[indices_key])
    results_67_joint_failed, predictions_67_joint, indices_67_joint = indepth_results(gold, results_67_joint, indices_dict[indices_key])


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

    print("\n Length of errors sets (text): {}, {}, {} Sum: {}".format(len(res42_failed_set), len(res17_failed_set),
                                                                       len(res67_failed_set), len(res42_failed_set) +len(res17_failed_set)+ len(res67_failed_set)))

    print("\n Length of errors sets (graph): {}, {}, {} Sum: {}".format(len(res42_failed_set_graph_only),
                                                        len(res17_failed_set_graph_only),
                                                        len(res67_failed_set_graph_only), len(res42_failed_set_graph_only)+len(res17_failed_set_graph_only)+len(res67_failed_set_graph_only)))

    print("\n Length of errors sets (joint): {}, {}, {} Sum: {}".format(len(res42_failed_set_joint),
                                                        len(res17_failed_set_joint),
                                                        len(res67_failed_set_joint), len(res42_failed_set_joint)+len(res17_failed_set_joint)+len(res67_failed_set_joint)))

    print("\n Common Errors: \n", *res42_failed_set.intersection(res17_failed_set, res67_failed_set), sep="\n")
    print("\n Common Errors Graph: \n", *res42_failed_set_graph_only.intersection(res17_failed_set_graph_only, res67_failed_set_graph_only), sep="\n")

    print("\n Common Errors Joint: \n", *res42_failed_set_joint.intersection(res17_failed_set_joint, res67_failed_set_joint), sep="\n")


    indices_text = indices_42.intersection(indices_17, indices_67)
    indices_graph = indices_42_graph.intersection(indices_17_graph, indices_67_graph)
    indices_joint = indices_42_joint.intersection(indices_17_joint, indices_67_joint)

    model_set_1 = res42_failed_set.intersection(res17_failed_set, res67_failed_set)
    model_set_2 = res42_failed_set_graph_only.intersection(res17_failed_set_graph_only, res67_failed_set_graph_only)
    model_set_3 = res42_failed_set_joint.intersection(res17_failed_set_joint, res67_failed_set_joint)

    intersection_between_models = model_set_1 & model_set_2 & model_set_3 # model_set_1.intersection(model_set_2)
    intersection_between_indices = indices_text & indices_graph & indices_joint

    print(len(intersection_between_models), len(intersection_between_indices))
    #print("\n Intersection of common errors: \n", *intersection_between_models, sep="\n")
    for instance, idx in zip(intersection_between_models, intersection_between_indices):
        print(instance, idx)
        print("Text 42: ", predictions_42[idx])
        print("Text 17: ", predictions_17[idx])
        print("Text 67: ", predictions_67[idx])

        print("Graph 42: ", predictions_42_graph[idx])
        print("Graph 17: ", predictions_17_graph[idx])
        print("Graph 67: ", predictions_67_graph[idx])

        print("Joint 42: ", predictions_42_joint[idx])
        print("Joint 17: ", predictions_17_joint[idx])
        print("Joint 67: ", predictions_67_joint[idx])

    #"""
    print(predictions_42)
    print(predictions_17)
    print(predictions_67)


    print(predictions_42_graph)
    print(predictions_17_graph)
    print(predictions_67_graph)

    print(predictions_42_joint)
    print(predictions_17_joint)
    print(predictions_67_joint)
    #"""







