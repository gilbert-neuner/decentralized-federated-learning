# import numpy as np
# from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import product
from experiment import grid_search, replicate_algorithm
# from generate_data import generate_adj_mtx, generate_beta_true
# import random

df_out = pd.DataFrame(columns = ["scheme", "clip", "max_step_size", "n_iter", "threshold", "random_displace", "winsorize", "info", "accelerate", "include", "n", "p", "SNR", "sparsity", "adversary_type", "corrupt_fraction", "jitter", "seed", "client_id", "F1", "beta_rel_norm", "Y_rel_norm", "metric", "thickness"])

# data_params: n, p, SNR, sparsity
# topology_params: adjacency_matrix, K, shape, thickness
# algorithm_params: scheme, clip, max_step_size, n_iter, threshold, beta0, random_displace, winsorize
# trust_params: info, accelerate, cosine_recompute, include
# adversary_params: which_adversaries, corrupt_fraction, jitter, adversary_type
# rep_params: replicate, seed
# grid_params: n_rep, seed, grid, shared_threshold, metric
for (scheme, thickness, metric) in product(["D-SGD", "D-PSGD", "G", "ClippedGossip", "AG", "A,G", "G,A", "S", "AS", "global"], [2, 4, 6, 8, 10], ["F1", "beta_rel_norm"]):
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
        which_adversaries = range(0, 20, 2)
    threshold = grid_search(topology_params = {"K": 20, "thickness": thickness}, algorithm_params = {"scheme": scheme, "clip": clip}, trust_params = {"info": info}, adversary_params = {"which_adversaries": which_adversaries}, grid_params = {"metric": metric})   
    for rep in range(10):
        df_replicate, _, _ = replicate_algorithm(topology_params = {"K": 20, "thickness": thickness}, algorithm_params = {"scheme": scheme, "clip": clip, "threshold": threshold}, trust_params = {"info": info}, adversary_params = {"which_adversaries": which_adversaries}, rep_params = {"replicate": rep})
        df_replicate["metric"] = metric
        df_replicate["thickness"] = thickness
        df_out = pd.concat([df_out, df_replicate], ignore_index = True)

print("\a")

# F1_out = (
#     df_out
#         .query("adversary_type == 'friendly' and metric == 'F1'")[["scheme", "thickness", "F1"]]
# )

# rel_norm_out = (
#     df_out
#         .query("adversary_type == 'friendly' and metric == 'beta_rel_norm'")[["scheme", "thickness", "beta_rel_norm"]]
# )


# precision_out = (
#     df_out
#         .query("adversary_type == 'friendly' and metric == 'F1'")[["scheme", "thickness", "precision"]]
# )


# recall_out = (
#     df_out
#         .query("adversary_type == 'friendly' and metric == 'F1'")[["scheme", "thickness", "recall"]]
# )

F1_out = (
    df_out
        .query("adversary_type == 'friendly' and metric == 'F1' and scheme in ['D-SGD', 'G', 'ClippedGossip', 'S', 'AS', 'global']")[["scheme", "thickness", "F1"]]
)

rel_norm_out = (
    df_out
        .query("adversary_type == 'friendly' and metric == 'beta_rel_norm' and scheme in ['D-SGD', 'G', 'ClippedGossip', 'S', 'AS', 'global']")[["scheme", "thickness", "beta_rel_norm"]]
)


precision_out = (
    df_out
        .query("adversary_type == 'friendly' and metric == 'F1' and scheme in ['D-SGD', 'G', 'ClippedGossip', 'S', 'AS', 'global']")[["scheme", "thickness", "precision"]]
)


recall_out = (
    df_out
        .query("adversary_type == 'friendly' and metric == 'F1' and scheme in ['D-SGD', 'G', 'ClippedGossip', 'S', 'AS', 'global']")[["scheme", "thickness", "recall"]]
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

fig, axes = plt.subplots(2, 2, sharex = True, figsize = (12, 8), constrained_layout = True)

sns.lineplot(
    data = F1_out,
    x = "thickness",
    y = "F1",
    hue = "scheme",
    palette = color_map,
    errorbar = "se",
    ax = axes[0, 0]
)

sns.lineplot(
    data = rel_norm_out,
    x = "thickness",
    y = "beta_rel_norm",
    hue = "scheme",
    palette = color_map,
    errorbar = "se",
    ax = axes[0, 1]
)

sns.lineplot(
    data = precision_out,
    x = "thickness",
    y = "precision",
    hue = "scheme",
    palette = color_map,
    errorbar = "se",
    ax = axes[1, 0]
)

sns.lineplot(
    data = recall_out,
    x = "thickness",
    y = "recall",
    hue = "scheme",
    palette = color_map,
    errorbar = "se",
    ax = axes[1, 1]
)

for i in range(2):
    for j in range(2):
        axes[i, j].legend_.remove()
    
handles, labels = axes[0, 0].get_legend_handles_labels()
    
# labels = ["D-SGD", "D-PSGD", "G (local)", "ClippedGossip", "AG", "A,G", "G,A", "S", "AS", "G (global)"]
labels = ["D-SGD", "G (local)", "ClippedGossip", "S", "AS", "G (global)"]
    
# fig.legend(handles, labels, loc = "lower center", ncol = 5, bbox_to_anchor = (0.5, -0.05))
fig.legend(handles, labels, loc = "lower center", ncol = 6, bbox_to_anchor = (0.5, -0.025))

axes[0, 0].set_xlabel(r"$\tau$")
axes[0, 1].set_xlabel(r"$\tau$")
axes[1, 0].set_xlabel(r"$\tau$")
axes[1, 1].set_xlabel(r"$\tau$")
axes[0, 1].set_ylabel("relative norm")
fig.suptitle("Experiment 3: Effect of Connectivity")

fig.subplots_adjust(bottom = 0.12)

plt.tight_layout()