import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import product
from experiment import grid_search, analyze_trust_history
from generate_data import generate_adj_mtx, generate_beta_true
import random

def get_rel_norm_value(friendly_ids, iteration, rel_norm_history_exp):
    friendly_sum = 0
    for friendly_id in friendly_ids:
        friendly_sum += rel_norm_history_exp[friendly_id][iteration]
    return friendly_sum / len(friendly_ids)

p = 100
sparsity = 0.08
n_iter = 100

beta_support = random.sample(range(p), round(p * sparsity))
beta_true = np.zeros(p)
beta_true[beta_support[0:3], ] = 1
beta_true[beta_support[4:7], ] = 0.35

df_out = pd.DataFrame(columns = ["scheme", "thickness", "rel_norm", "F1", "norm", "sparsity", "adversary_type"])

for scheme, local_or_global, metric in product(["G", "A", "(AC)"], ["local", "global"], ["rel_norm", "Y"]): 
    K = 20
    shape = "band"
    if shape == "band":
        topology_params = {"adjacency_matrix": generate_adj_mtx(K, shape, thickness = 6)}
        adversary_params = {"which_adversaries": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19], "corrupt_fraction": {1:0.5, 3:1, 5:0.5, 7:1, 9:0.5, 11:1, 13:0.5, 15:1, 17:0.5, 19:1}, "adversary_type": "X"}
        all_friendly_ids = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        adversary2_ids = [3, 7, 11, 15, 19]
        adversary_ids = [1, 5, 9, 13, 17]
    
    data_params = {"n": 50, "p": p, "SNR": 1, "beta_true": beta_true}
    algorithm_params = {"scheme": scheme, "max_step_size": 1, "n_iter": n_iter}
    grid_params = {"n_rep": 3, "thresholds": 10 ** np.arange(-1, 1.1, 0.5), "local_or_global": local_or_global, "metric": metric, "seed": 1234}
    start_params = {"start": "identical", "beta0": None}
    trust_params = {"info": "Both", "accelerate": True, "include": 0}
    
    algorithm_params["thresholds"] = grid_search(topology_params, data_params, algorithm_params, grid_params, start_params, trust_params, adversary_params)
    experiment_params = {"n_rep": 10, "seed": 12345}
    
    rel_norm_exp, F1_exp, gradient_history_exp, model_history_exp, F1_history_exp, rel_norm_history_exp, beta_history_exp, beta_true_exp = analyze_trust_history(topology_params, data_params, algorithm_params, experiment_params, start_params, trust_params, adversary_params)
    
    all_client_ids = list(range(0, K))
    
    time_points = range(100)
    rows = []
    adjacency_matrix = topology_params["adjacency_matrix"]
    for i in all_friendly_ids:
        # friendly and adversary2
        for j in adversary2_ids:
            if adjacency_matrix[i, j] == 1:
                for k in time_points:
                    a = sum(x for x in gradient_history_exp[i][j][0:(k + 1)] if x > 0)
                    b = -1 * sum(x for x in gradient_history_exp[i][j][0:(k + 1)] if x < 0)
                    rows.append([k, "grad_fa2", a / (a + b)])
                    a = sum(x for x in model_history_exp[i][j][0:(k + 1)] if x > 0)
                    b = -1 * sum(x for x in model_history_exp[i][j][0:(k + 1)] if x < 0)
                    rows.append([k, "model_fa2", a / (a + b)])
        # friendly and adversary
        for j in adversary_ids:
            if adjacency_matrix[i, j] == 1:
                for k in time_points:
                    a = sum(x for x in gradient_history_exp[i][j][0:(k + 1)] if x > 0)
                    b = -1 * sum(x for x in gradient_history_exp[i][j][0:(k + 1)] if x < 0)
                    rows.append([k, "grad_fa", a / (a + b)])
                    a = sum(x for x in model_history_exp[i][j][0:(k + 1)] if x > 0)
                    b = -1 * sum(x for x in model_history_exp[i][j][0:(k + 1)] if x < 0)
                    rows.append([k, "model_fa", a / (a + b)])
        # friendly and friendly
        for j in list(set([0, 2, 4, 6, 8, 10, 12, 14, 16, 18]) - set([i])):
            if adjacency_matrix[i, j] == 1:
                for k in time_points:
                    a = sum(x for x in gradient_history_exp[i][j][0:(k + 1)] if x > 0)
                    b = -1 * sum(x for x in gradient_history_exp[i][j][0:(k + 1)] if x < 0)
                    rows.append([k, "grad_ff", a / (a + b)])
                    a = sum(x for x in model_history_exp[i][j][0:(k + 1)] if x > 0)
                    b = -1 * sum(x for x in model_history_exp[i][j][0:(k + 1)] if x < 0)
                    rows.append([k, "model_ff", a / (a + b)])
                
    df = pd.DataFrame(rows, columns = ["iteration", "group", "trust"])
    
    rel_norm_values = [get_rel_norm_value(all_friendly_ids, i, rel_norm_history_exp) for i in [1, 2, 4, 6, 16, 32, 64]]
    
    # plt.figure()
    # grad_plot = sns.violinplot(x = "iteration", y = "trust", hue = "group", hue_order = ["grad_fa2", "grad_fa", "grad_ff"], data = df.query("(group in ['grad_fa2', 'grad_fa', 'grad_ff']) and (iteration in [1, 2, 4, 8, 16, 32, 64])"), cut = 0, inner = "point", scale = "width")
    # grad_plot.set_title(f"Scheme = {scheme}, thickness = {thickness}, rel_norm = {np.mean(rel_norm_exp[all_friendly_ids]):.3f}, F1 = {np.mean(F1_exp[all_friendly_ids]):.3f}")
    # plt.plot([0, 1, 2, 3, 4, 5, 6], rel_norm_values, marker="o", color="red", linestyle="-")
    # grad_plot.legend(
    #     bbox_to_anchor=(1.05, 1),   # x=1.05, y=1 relative to axes
    #     loc='upper left',           # anchor point of the legend box
    #     borderaxespad=0
    # )
    
    # plt.figure()
    # model_plot = sns.violinplot(x = "iteration", y = "trust", hue = "group", hue_order = ["model_fa2", "model_fa", "model_ff"], data = df.query("(group in ['model_fa2', 'model_fa', 'model_ff']) and (iteration in [1, 2, 4, 8, 16, 32, 64])"), cut = 0, inner = "point", scale = "width")
    # model_plot.set_title(f"Scheme = {scheme}, thickness = {thickness}, rel_norm = {np.mean(rel_norm_exp[all_friendly_ids]):.3f}, F1 = {np.mean(F1_exp[all_friendly_ids]):.3f}")
    # plt.plot([0, 1, 2, 3, 4, 5, 6], rel_norm_values, marker="o", color="red", linestyle="-")
    # model_plot.legend(
    #     bbox_to_anchor=(1.05, 1),   # x=1.05, y=1 relative to axes
    #     loc='upper left',           # anchor point of the legend box
    #     borderaxespad=0
    # )
    
    # compute L2 norm and sparsity
    norm_out = 0
    sparsity_out = 0
    for i in all_friendly_ids:
        norm_out += LA.norm(beta_history_exp[i][n_iter - 1])
        sparsity_out += np.mean(beta_history_exp[i][n_iter - 1] != 0)
    norm_out /= len(all_friendly_ids)
    sparsity_out /= len(all_friendly_ids)
    
    # add to data frame "method", "thickness", "rel_norm", "F1", "norm", "sparsity"
    new_row = pd.DataFrame({"scheme": [scheme], "local_or_global": [local_or_global], "metric": [metric], "rel_norm": [np.mean(rel_norm_exp[all_friendly_ids])], "F1": [np.mean(F1_exp[all_friendly_ids])], "norm": [norm_out], "sparsity": [sparsity_out]})
    df_out = pd.concat([df_out, new_row], ignore_index = True)
    
    print(scheme)
    
print("\a")