# import numpy as np
# from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import product
from experiment import grid_search, replicate_algorithm
# from generate_data import generate_adj_mtx, generate_beta_true
# import random

df_out = pd.DataFrame(columns = ["scheme", "clip", "max_step_size", "n_iter", "threshold", "random_displace", "winsorize", "info", "accelerate", "include", "n", "p", "SNR", "sparsity", "adversary_type", "corrupt_fraction", "jitter", "seed", "client_id", "F1", "beta_rel_norm", "Y_rel_norm", "metric", "num_adversaries"])

# data_params: n, p, SNR, sparsity
# topology_params: adjacency_matrix
# algorithm_params: scheme, clip, max_step_size, n_iter, threshold, beta0, random_displace, winsorize
# trust_params: info, accelerate, cosine_recompute, include
# adversary_params: which_adversaries, corrupt_fraction, jitter, adversary_type
# rep_params: replicate, seed
# grid_params: n_rep, seed, grid, shared_threshold, metric
for (scheme, num_adversaries, metric) in product(["D-SGD", "D-PSGD", "G", "ClippedGossip", "AG", "A,G", "G,A", "S", "AS", "global"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ["F1", "beta_rel_norm"]):
    if scheme == "D-SGD" or scheme == "D-PSGD":
        clip = False
    else:
        clip = True
    if scheme == "D-SGD" or scheme == "D-PSGD" or scheme == "G" or scheme == "ClippedGossip" or scheme == "global":
        info = "None"
    else:
        info = "Both"
    if scheme == "G" or scheme == "global":
        which_adversaries = range(0)
    else:
        which_adversaries = range(num_adversaries)
    threshold = grid_search(algorithm_params = {"scheme": scheme, "clip": clip}, trust_params = {"info": info}, adversary_params = {"which_adversaries": which_adversaries}, grid_params = {"metric": metric})   
    for rep in range(10):
        df_replicate, _, _ = replicate_algorithm(algorithm_params = {"scheme": scheme, "clip": clip, "threshold": threshold}, trust_params = {"info": info}, adversary_params = {"which_adversaries": which_adversaries}, rep_params = {"replicate": rep})
        df_replicate["metric"] = metric
        df_replicate["num_adversaries"] = num_adversaries
        df_out = pd.concat([df_out, df_replicate], ignore_index = True)

print("\a")

F1_out = (
    df_out
        .query("adversary_type == 'friendly' and metric == 'F1'")
        .groupby(["scheme", "num_adversaries"], as_index = False)[["F1"]]
        .mean()
)

rel_norm_out = (
    df_out
        .query("adversary_type == 'friendly' and metric == 'beta_rel_norm'")
        .groupby(["scheme", "num_adversaries"], as_index = False)[["beta_rel_norm"]]
        .mean()
)

color_map = {
    "D-SGD": "#e6194B",
    "D-PSGD": "#800000",
    "G": "#f58231",
    "ClippedGossip": "#3cb44b",
    "G,A": "#4363d8",
    "AG": "#000075",
    "A,G": "#42d4f4",
    "AS": "#911eb4",
    "S": "#f032e6",
    "global": "gold"
}

fig, axes = plt.subplots(1, 2)

sns.lineplot(
    data = F1_out,
    x = "num_adversaries",
    y = "F1",
    hue = "scheme",
    palette = color_map,
    ax = axes[0]
)

sns.lineplot(
    data = rel_norm_out,
    x = "num_adversaries",
    y = "beta_rel_norm",
    hue = "scheme",
    palette = color_map,
    ax = axes[1]
)

for ax in axes:
    ax.legend_.remove()
    
handles, labels = axes[0].get_legend_handles_labels()
    
new_labels = ["A, G", "AG", "AS", "ClippedGossip", "D-PSGD", "D-SGD", "G (local)", "G, A", "S", "G (global)"]
    
fig.legend(handles, new_labels, loc = "center left", bbox_to_anchor = (1, 0.5))

axes[0].set_xlabel("Number of Byzantine Clients")
axes[1].set_xlabel("Number of Byzantine Clients")
axes[1].set_ylabel("Relative Norm")
fig.suptitle("Experiment 1: Effect of Number of Byzantine Clients")

fig.subplots_adjust(wspace=0.3)

plt.tight_layout()