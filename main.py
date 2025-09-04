from experiment import *
from generate_data import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import product

def get_rel_norm_value(friendly_ids, iteration, rel_norm_history_exp):
    friendly_sum = 0
    for friendly_id in friendly_ids:
        friendly_sum += rel_norm_history_exp[friendly_id][iteration]
    return friendly_sum / len(friendly_ids)

for scheme, which_adversaries, include in product(["G", "A", "(AC)", "CG", "GC", "(GC)"], [[0, 1, 2, 3], [0, 1, 2, 3, 4, 5]], [-0.6, -0.4, 0, 0.4, 0.6]):
    K = 10
    shape = "mesh"
    topology_params = {"adjacency_matrix": generate_adj_mtx(K, shape)}
    
    # adjacency_matrix = np.ones([K, K], dtype = int)
    # adjacency_matrix[5, 6] = 0
    # adjacency_matrix[6, 5] = 0
    # adjacency_matrix[5, 7] = 0
    # adjacency_matrix[7, 5] = 0
    # adjacency_matrix[0, 7] = 0
    # adjacency_matrix[7, 0] = 0
    # adjacency_matrix[1, 9] = 0
    # adjacency_matrix[9, 1] = 0
    
    data_params = {"n": 50, "p": 100, "SNR": 1, "sparsity": 0.05}
    algorithm_params = {"scheme": scheme, "max_step_size": 1, "n_iter": 100}
    grid_params = {"n_rep": 3, "thresholds": 10 ** np.arange(-1, 1.1, 0.5), "metric": "rel_norm", "seed": 1234}
    start_params = {"start": "identical", "beta0": None}
    trust_params = {"info": "Both", "accelerate": True, "include": include}
    adversary_params = {"which_adversaries": which_adversaries, "corrupt_fraction": 1}
    
    algorithm_params["threshold"] = grid_search(topology_params, data_params, algorithm_params, grid_params, start_params, trust_params, adversary_params)
    experiment_params = {"n_rep": 10, "seed": 12345}
    
    rel_norm_exp, F1_exp, gradient_history_exp, model_history_exp, F1_history_exp, rel_norm_history_exp = analyze_trust_history(topology_params, data_params, algorithm_params, experiment_params, start_params, trust_params, adversary_params)
    
    all_client_ids = list(range(0, K))
    all_friendly_ids = [client_id for client_id in all_client_ids if client_id not in which_adversaries]
    
    time_points = range(100)
    rows = []
    for i in all_friendly_ids:
        for j in which_adversaries:
            for k in time_points:
                a = sum(x for x in gradient_history_exp[i][j][0:(k + 1)] if x > 0)
                b = -1 * sum(x for x in gradient_history_exp[i][j][0:(k + 1)] if x < 0)
                rows.append([k, "grad_fa", a / (a + b)])
                a = sum(x for x in model_history_exp[i][j][0:(k + 1)] if x > 0)
                b = -1 * sum(x for x in model_history_exp[i][j][0:(k + 1)] if x < 0)
                rows.append([k, "model_fa", a / (a + b)])
        for j in list(set([5, 6, 7, 8, 9]) - set([i])):
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
    grad_plot = sns.violinplot(x = "iteration", y = "trust", hue = "group", data = df.query("(group in ['grad_fa', 'grad_ff']) and (iteration in [1, 2, 4, 8, 16, 32, 64])"), cut = 0, inner = "point", scale = "width")
    grad_plot.set_title(f"Scheme = {scheme}, adversaries = {len(which_adversaries)}, include = {include}, rel_norm = {np.mean(rel_norm_exp[all_friendly_ids]):.3f}, F1 = {np.mean(F1_exp[all_friendly_ids]):.3f}")
    plt.plot([0, 1, 2, 3, 4, 5, 6], rel_norm_values, marker="o", color="red", linestyle="-")
    grad_plot.legend(
        bbox_to_anchor=(1.05, 1),   # x=1.05, y=1 relative to axes
        loc='upper left',           # anchor point of the legend box
        borderaxespad=0
    )
    
    plt.figure()
    model_plot = sns.violinplot(x = "iteration", y = "trust", hue = "group", data = df.query("(group in ['model_fa', 'model_ff']) and (iteration in [1, 2, 4, 8, 16, 32, 64])"), cut = 0, inner = "point", scale = "width")
    model_plot.set_title(f"Scheme = {scheme}, adversaries = {len(which_adversaries)}, include = {include}, rel_norm = {np.mean(rel_norm_exp[all_friendly_ids]):.3f}, F1 = {np.mean(F1_exp[all_friendly_ids]):.3f}")
    plt.plot([0, 1, 2, 3, 4, 5, 6], rel_norm_values, marker="o", color="red", linestyle="-")
    model_plot.legend(
        bbox_to_anchor=(1.05, 1),   # x=1.05, y=1 relative to axes
        loc='upper left',           # anchor point of the legend box
        borderaxespad=0
    )
    
    print(scheme, which_adversaries, include)
    
    # TODO: why are rel_norm_means different for G and (GC)