from itertools import product

import numpy as np
from deepsig import aso, bootstrap_power_analysis

seed = 17

def statistical_significance_mnli_test_set():
    amr_text = [88.903, 89.230, 88.546]
    amr_graph = [83.411, 83.973, 83.513]
    amr_joint = [89.250, 89.189, 88.331]
    baseline_scores = [88.536, 89.332, 89.271]

    min_eps_text = aso(amr_text, baseline_scores, seed=seed)
    min_eps_graph = aso(amr_graph, baseline_scores, seed=seed)
    min_eps_joint = aso(amr_joint, baseline_scores, seed=seed)
    print("Epsilon Text: ", min_eps_text)
    print("Epsilon Graph: ", min_eps_graph)
    print("Epsilon Joint: ", min_eps_joint)
    power = bootstrap_power_analysis(amr_text, show_progress=False)
    print(power)
    power = bootstrap_power_analysis(amr_graph, show_progress=False)
    print(power)
    power = bootstrap_power_analysis(amr_joint, show_progress=False)
    print(power)
    power = bootstrap_power_analysis(baseline_scores, show_progress=False)
    print(power)

def statistical_significance_mnli_val_matched_set():
    amr_text = [89.24, 89.45, 89.1]
    amr_graph = [82.89, 83.16, 82.63]
    amr_joint = [87.96, 87.96, 87.79]
    baseline_scores = [88.71, 89.34, 89.72]

    min_eps_text = aso(amr_text, baseline_scores, seed=seed)
    min_eps_graph = aso(amr_graph, baseline_scores, seed=seed)
    min_eps_joint = aso(amr_joint, baseline_scores, seed=seed)
    print("Epsilon Text: ", min_eps_text)
    print("Epsilon Graph: ", min_eps_graph)
    print("Epsilon Joint: ", min_eps_joint)


def veridicality_pos():

    #amr_text = [98.27, 98.00, 96.00, 15.34, 5.95, 93.94, 92.92, 15.04]
    #amr_graph = [96.07, 97.33, 68.00, 5.82, 14.29, 84.85, 91.67, 6.24]
    #amr_joint = [93.55, 98.00, 93.33, 31.74, 17.86, 93.34, 90.00, 29.23]
    #baseline_scores = [97.32, 98.33, 93.33, 16.59, 13.10, 92.73, 91.67, 20.85]
    amr_text_acc = [35.18, 40.25, 46.4]
    amr_graph_acc = [31.51, 35.11, 34.31]
    amr_joint_acc = [50.6, 51.8, 46.06]
    baseline_scores_acc = [42.72, 43.46, 47.46]


    amr_text = [37.6, 46.93, 51.26]
    amr_graph = [37.68, 41.55, 39.85]
    amr_joint = [56.34, 53.14,52.25]
    baseline_scores = [46.69,47.59,49.98]

    min_eps_text = aso(amr_text, baseline_scores, seed=seed)
    min_eps_graph = aso(amr_graph, baseline_scores, seed=seed)
    min_eps_joint = aso(amr_joint, baseline_scores, seed=seed)
    print("Epsilon Text: ", min_eps_text)
    print("Epsilon Graph: ", min_eps_graph)
    print("Epsilon Joint: ", min_eps_joint)

    min_eps_text = aso(amr_text_acc, baseline_scores_acc, seed=seed)
    min_eps_graph = aso(amr_graph_acc, baseline_scores_acc, seed=seed)
    min_eps_joint = aso(amr_joint_acc, baseline_scores_acc, seed=seed)
    print("Epsilon Text Acc: ", min_eps_text)
    print("Epsilon Graph Acc: ", min_eps_graph)
    print("Epsilon Joint Acc: ", min_eps_joint)


def veridicality_neg():

    amr_text_acc = [32.98, 39.05, 39.99]
    amr_graph_acc = [25.03, 25.43, 29.11]
    amr_joint_acc = [49.0, 37.58, 38.52]
    baseline_scores_acc = [40.45, 29.17, 39.79]

    amr_text = [34.79, 40.26, 41.26]
    amr_graph = [24.99, 25.66, 28.95]
    amr_joint = [47.31, 39.13, 39.23]
    baseline_scores = [40.96,31.99,40.86]

    min_eps_text = aso(amr_text, baseline_scores, seed=seed)
    min_eps_graph = aso(amr_graph, baseline_scores, seed=seed)
    min_eps_joint = aso(amr_joint, baseline_scores, seed=seed)
    print("Epsilon Text: ", min_eps_text)
    print("Epsilon Graph: ", min_eps_graph)
    print("Epsilon Joint: ", min_eps_joint)


    min_eps_text = aso(amr_text_acc, baseline_scores_acc, seed=seed)
    min_eps_graph = aso(amr_graph_acc, baseline_scores_acc, seed=seed)
    min_eps_joint = aso(amr_joint_acc, baseline_scores_acc, seed=seed)
    print("Epsilon Text Acc: ", min_eps_text)
    print("Epsilon Graph Acc: ", min_eps_graph)
    print("Epsilon Joint Acc: ", min_eps_joint)


def veridicality_pos_fine():
    M = 3  # Number of datasets
    N = 3 # number of seeds
    baseline_scores = [[98.11, 98.00, 92.00, 23.81, 14.29, 92.73, 93.75, 17.75], [98.11, 99.00, 96.00, 22.22, 3.57, 92.73, 93.75, 19.14], \
                  [95.75, 98.00, 92.00, 3.75, 21.43, 92.73, 87.5, 25.67]]
    amr_text = [[99.53, 98.00, 96.00, 4.76, 0.0, 98.18, 95.00, 6.52], [99.06, 98.00, 96.00, 11.11, 7.14, 92.73, 92.5, 14.65],
                [96.23, 98.00, 96.00, 30.16, 10.71, 90.91, 91.25, 23.96]]
    amr_graph = [[95.75, 97.00, 52.00, 1.59, 7.14, 87.27, 92.50, 3.63], [96.22, 97.00, 88.00, 6.35, 21.43, 83.64, 91.25, 7.91],
                [96.23, 98.00, 64.00, 9.52, 14.29, 83.64, 91.25, 7.17]]
    amr_joint = [[97.17, 99.00, 92.00, 26.98, 32.14, 92.73, 92.50, 29.84], [88.68, 98.00, 92.00, 39.68, 0.0, 94.55, 85.00, 34.43],
                 [94.81, 97.00, 96.00, 28.57, 21.43, 92.73, 92.50, 23.42]]


    min_eps_text = [aso(a, b, confidence_level=0.95, num_comparisons=M, seed=seed) for a, b in zip(amr_text, baseline_scores)]
    min_eps_graph = [aso(a, b, confidence_level=0.95, num_comparisons=M, seed=seed) for a, b in
                    zip(amr_graph, baseline_scores)]
    min_eps_joint = [aso(a, b, confidence_level=0.95, num_comparisons=M, seed=seed) for a, b in
                    zip(amr_joint, baseline_scores)]
    #min_eps_graph = aso(amr_graph, baseline_scores, seed=seed)
    #min_eps_joint = aso(amr_joint, baseline_scores, seed=seed)
    print("Epsilon Text: ", min_eps_text)
    print("Epsilon Graph: ", min_eps_graph)
    print("Epsilon Joint: ", min_eps_joint)
    power = bootstrap_power_analysis(amr_text[0], show_progress=False)
    print(power)
    power = bootstrap_power_analysis(amr_graph[0], show_progress=False)
    print(power)
    power = bootstrap_power_analysis(amr_joint[0], show_progress=False)
    print(power)
    power = bootstrap_power_analysis(baseline_scores[0], show_progress=False)
    print(power)


def veridicality_neg_fine():
    M = 3  # Number of datasets
    N = 3 # number of seeds
    baseline_scores = [[81.60, 86.00, 96.00, 50.79, 92.86, 0.00, 36.25, 25.24], [83.02, 90.00, 96.00, 53.97, 92.86, 0.0, 10.0, 8.45], \
                  [78.3, 91.00, 96.00, 46.03, 96.43, 0.0, 35.0, 24.71]]
    amr_text = [[99.53, 98.00, 96.00, 4.76, 0.0, 98.18, 95.00, 6.52], [99.06, 98.00, 96.00, 11.11, 7.14, 92.73, 92.5, 14.65],
                [96.23, 98.00, 96.00, 30.16, 10.71, 90.91, 91.25, 23.96]]
    amr_graph = [[95.75, 97.00, 52.00, 1.59, 7.14, 87.27, 92.50, 3.63], [96.22, 97.00, 88.00, 6.35, 21.43, 83.64, 91.25, 7.91],
                [96.23, 98.00, 64.00, 9.52, 14.29, 83.64, 91.25, 7.17]]
    amr_joint = [[97.17, 99.00, 92.00, 26.98, 32.14, 92.73, 92.50, 29.84], [88.68, 98.00, 92.00, 39.68, 0.0, 94.55, 85.00, 34.43],
                 [94.81, 97.00, 96.00, 28.57, 21.43, 92.73, 92.50, 23.42]]


    min_eps_text = [aso(a, b, confidence_level=0.95, num_comparisons=M, seed=seed) for a, b in zip(amr_text, baseline_scores)]
    min_eps_graph = [aso(a, b, confidence_level=0.95, num_comparisons=M, seed=seed) for a, b in
                    zip(amr_graph, baseline_scores)]
    min_eps_joint = [aso(a, b, confidence_level=0.95, num_comparisons=M, seed=seed) for a, b in
                    zip(amr_joint, baseline_scores)]
    #min_eps_graph = aso(amr_graph, baseline_scores, seed=seed)
    #min_eps_joint = aso(amr_joint, baseline_scores, seed=seed)
    print("Epsilon Text: ", min_eps_text)
    print("Epsilon Graph: ", min_eps_graph)
    print("Epsilon Joint: ", min_eps_joint)



if __name__ == "__main__":

    #statistical_significance_mnli_test_set()
    #statistical_significance_mnli_val_matched_set()
    #veridicality_pos()
    #veridicality_neg_fine()
    veridicality_neg()


    #my_model_scored_samples_per_run = [np.random.normal(loc=0.3, scale=0.8, size=M) for _ in range(N)]
    #baseline_scored_samples_per_run = [np.random.normal(loc=0, scale=1, size=M) for _ in range(N)]
    """
    my_model_scored_samples_per_run = [[83.411], [83.973], [ 3.513]]
    baseline_scored_samples_per_run = [[88.536], [89.332], [89.271]]
    print(baseline_scored_samples_per_run)
    pairs = list(product(my_model_scored_samples_per_run, baseline_scored_samples_per_run))
    print(pairs)
    # epsilon_min values with Bonferroni correction
    eps_min = [aso(a, b, confidence_level=0.95, num_comparisons=len(pairs), seed=seed) for a, b in pairs]
    #print("{}".format({a:b for a,b in zip(pairs, eps_min)}))
    print(eps_min)
    """
