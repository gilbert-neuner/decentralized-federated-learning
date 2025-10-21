import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import product
from experiment import grid_search, analyze_trust_history
from generate_data import generate_adj_mtx

def get_rel_norm_value(friendly_ids, iteration, rel_norm_history_exp):
    friendly_sum = 0
    for friendly_id in friendly_ids:
        friendly_sum += rel_norm_history_exp[friendly_id][iteration]
    return friendly_sum / len(friendly_ids)

for scheme, thickness in product(["G", "A", "(AC)"], [2, 6, 10]): # 
    K = 20
    shape = "band"
    if shape == "band":
        topology_params = {"adjacency_matrix": generate_adj_mtx(K, shape, thickness)}
        adversary_params = {"which_adversaries": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19], "corrupt_fraction": {1:0.5, 3:1, 5:0.5, 7:1, 9:0.5, 11:1, 13:0.5, 15:1, 17:0.5, 19:1}}
        all_friendly_ids = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        adversary2_ids = [3, 7, 11, 15, 19]
        adversary_ids = [1, 5, 9, 13, 17]
    elif shape == "bridged ring":
        adjacency_matrix = generate_adj_mtx(K, "ring")
        adjacency_matrix[4, 6] = 1
        adjacency_matrix[6, 4] = 1
        adjacency_matrix[9, 11] = 1
        adjacency_matrix[11, 9] = 1
        adjacency_matrix[14, 16] = 1
        adjacency_matrix[16, 14] = 1
        topology_params = {"adjacency_matrix": adjacency_matrix}
        adversary_params = {"which_adversaries": [0, 5, 10, 15], "corrupt_fraction": {0:0.5, 5:1, 10:0.5, 15:1}}
        all_friendly_ids = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]
        adversary2_ids = [5, 15]
        adversary_ids = [0, 10]
    
    data_params = {"n": 50, "p": 100, "SNR": 1, "sparsity": 0.05} # {"n": 1, "p": 2, "SNR": 1, "sparsity": 0.5}
    algorithm_params = {"scheme": scheme, "max_step_size": 1, "n_iter": 100}
    grid_params = {"n_rep": 3, "thresholds": 10 ** np.arange(-1, 1.1, 0.5), "metric": "rel_norm", "seed": 1234}
    start_params = {"start": "identical", "beta0": None}
    trust_params = {"info": "Both", "accelerate": True, "include": 0}
    
    algorithm_params["threshold"] = grid_search(topology_params, data_params, algorithm_params, grid_params, start_params, trust_params, adversary_params)
    experiment_params = {"n_rep": 10, "seed": 12345}
    
    rel_norm_exp, F1_exp, gradient_history_exp, model_history_exp, F1_history_exp, rel_norm_history_exp, beta_history_exp = analyze_trust_history(topology_params, data_params, algorithm_params, experiment_params, start_params, trust_params, adversary_params)
    
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
    
    plt.figure()
    grad_plot = sns.violinplot(x = "iteration", y = "trust", hue = "group", hue_order = ["grad_fa2", "grad_fa", "grad_ff"], data = df.query("(group in ['grad_fa2', 'grad_fa', 'grad_ff']) and (iteration in [1, 2, 4, 8, 16, 32, 64])"), cut = 0, inner = "point", scale = "width")
    grad_plot.set_title(f"Scheme = {scheme}, thickness = {thickness}, rel_norm = {np.mean(rel_norm_exp[all_friendly_ids]):.3f}, F1 = {np.mean(F1_exp[all_friendly_ids]):.3f}")
    plt.plot([0, 1, 2, 3, 4, 5, 6], rel_norm_values, marker="o", color="red", linestyle="-")
    grad_plot.legend(
        bbox_to_anchor=(1.05, 1),   # x=1.05, y=1 relative to axes
        loc='upper left',           # anchor point of the legend box
        borderaxespad=0
    )
    
    plt.figure()
    model_plot = sns.violinplot(x = "iteration", y = "trust", hue = "group", hue_order = ["model_fa2", "model_fa", "model_ff"], data = df.query("(group in ['model_fa2', 'model_fa', 'model_ff']) and (iteration in [1, 2, 4, 8, 16, 32, 64])"), cut = 0, inner = "point", scale = "width")
    model_plot.set_title(f"Scheme = {scheme}, thickness = {thickness}, rel_norm = {np.mean(rel_norm_exp[all_friendly_ids]):.3f}, F1 = {np.mean(F1_exp[all_friendly_ids]):.3f}")
    plt.plot([0, 1, 2, 3, 4, 5, 6], rel_norm_values, marker="o", color="red", linestyle="-")
    model_plot.legend(
        bbox_to_anchor=(1.05, 1),   # x=1.05, y=1 relative to axes
        loc='upper left',           # anchor point of the legend box
        borderaxespad=0
    )
    
    print(scheme)
    
print("\a")

# orange_cmap = plt.get_cmap('Oranges')
# blue_cmap   = plt.get_cmap('Blues')
# green_cmap  = plt.get_cmap('Greens')

# plt.figure()
# for seq in range(0, 20):
#     coords = np.stack(beta_history_exp[seq])
#     x, y = coords[:, 0], coords[:, 1]
#     if seq in all_friendly_ids:
#         col = green_cmap((seq - min(all_friendly_ids)) / (len(all_friendly_ids) - 1))
#     elif seq in adversary_ids:
#         col = orange_cmap((seq - min(adversary_ids)) / (len(adversary_ids) - 1))
#     elif seq in adversary2_ids:
#         col = blue_cmap((seq - min(adversary2_ids)) / (len(adversary2_ids) - 1))
#     plt.plot(x, y, alpha = 0.5, color = col)
# plt.show()
    
# TODO: why are rel_norm_means different for G and (GC)