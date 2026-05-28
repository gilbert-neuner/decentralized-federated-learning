# import numpy as np
# from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import product
from experiment import grid_search, replicate_algorithm
# from generate_data import generate_adj_mtx, generate_beta_true
# import random

df_out = pd.DataFrame(columns = ["scheme", "clip", "max_step_size", "n_iter", "threshold", "random_displace", "winsorize", "info", "accelerate", "include", "n", "p", "SNR", "sparsity", "adversary_type", "corrupt_fraction", "jitter", "seed", "client_id", "F1", "beta_rel_norm", "Y_rel_norm", "metric", "corrupt_fraction_exp", "adversary_type_exp"])

# data_params: n, p, SNR, sparsity
# topology_params: adjacency_matrix
# algorithm_params: scheme, clip, max_step_size, n_iter, threshold, beta0, random_displace, winsorize
# trust_params: info, accelerate, cosine_recompute, include
# adversary_params: which_adversaries, corrupt_fraction, jitter, adversary_type
# rep_params: replicate, seed
# grid_params: n_rep, seed, grid, shared_threshold, metric
for (scheme, corrupt_fraction_exp, adversary_type_exp) in product(["S", "AS"], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], ["g", "Y", "X", "X_cols", "X_rows"]):
    threshold = grid_search(algorithm_params = {"scheme": scheme}, adversary_params = {"corrupt_fraction": corrupt_fraction_exp, "adversary_type": adversary_type_exp})   
    for rep in range(10):
        df_replicate, _, _ = replicate_algorithm(algorithm_params = {"scheme": scheme, "threshold": threshold}, adversary_params = {"corrupt_fraction": corrupt_fraction_exp, "adversary_type": adversary_type_exp}, rep_params = {"replicate": rep})
        df_replicate["corrupt_fraction_exp"] = corrupt_fraction_exp
        df_replicate["adversary_type_exp"] = adversary_type_exp
        df_out = pd.concat([df_out, df_replicate], ignore_index = True)

print("\a")

S_out = (
    df_out
        .query("adversary_type == 'friendly' and scheme == 'S' and (adversary_type_exp == 'g' or adversary_type_exp == 'X' or adversary_type_exp == 'Y')")[["corrupt_fraction_exp", "beta_rel_norm", "adversary_type_exp"]]
)

AS_out = (
    df_out
        .query("adversary_type == 'friendly' and scheme == 'AS' and (adversary_type_exp == 'g' or adversary_type_exp == 'X' or adversary_type_exp == 'Y')")[["corrupt_fraction_exp", "beta_rel_norm", "adversary_type_exp"]]
)

fig, axes = plt.subplots(1, 2, sharey = True)

sns.lineplot(
    data = AS_out,
    x = "corrupt_fraction_exp",
    y = "beta_rel_norm",
    hue = "adversary_type_exp",
    errorbar = "se",
    ax = axes[0]
)

sns.lineplot(
    data = S_out,
    x = "corrupt_fraction_exp",
    y = "beta_rel_norm",
    hue = "adversary_type_exp",
    errorbar = "se",
    ax = axes[1]
)

for ax in axes:
    ax.legend_.remove()
    
handles, labels = axes[0].get_legend_handles_labels()
    
fig.legend(handles, labels, loc = "center left", bbox_to_anchor = (1, 0.5))

axes[0].set_xlabel(r"$p_\text{flip}$")
axes[1].set_xlabel(r"$p_\text{flip}$")
axes[0].set_ylabel("Relative Norm")
axes[1].set_ylabel("Relative Norm")
axes[0].set_title("AS")
axes[1].set_title("S")
fig.suptitle("Experiment 4: Effect of Byzantine Failure Type")

fig.subplots_adjust(wspace=0.3)

plt.tight_layout()