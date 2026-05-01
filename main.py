# import numpy as np
# from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import product
from experiment import grid_search, replicate_algorithm
from generate_data import generate_adj_mtx #, generate_beta_true
# import random

df_out = pd.DataFrame(columns = ["scheme", "clip", "max_step_size", "n_iter", "threshold", "random_displace", "winsorize", "info", "accelerate", "include", "n", "p", "SNR", "sparsity", "adversary_type", "corrupt_fraction", "jitter", "seed", "client_id", "F1", "beta_rel_norm", "Y_rel_norm"])

# data_params: n, p, SNR, sparsity
# topology_params: adjacency_matrix
# algorithm_params: scheme, clip, max_step_size, n_iter, threshold, beta0, random_displace, winsorize
# trust_params: info, accelerate, cosine_recompute, include
# adversary_params: which_adversaries, corrupt_fraction, jitter, adversary_type
# rep_params: replicate, seed
for (scheme, n) in product(["G", "global"], [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]):
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
        pass # which_adversaries = range(num_adversaries)
    threshold = 0 # grid_search(data_params = {"n": n}, topology_params = {"adjacency_matrix": generate_adj_mtx(1)}, algorithm_params = {"scheme": scheme, "clip": clip}, trust_params = {"info": info}, adversary_params = {"which_adversaries": which_adversaries})   
    for rep in range(1):
        df_replicate, _, _ = replicate_algorithm(data_params = {"n": n}, topology_params = {"adjacency_matrix": generate_adj_mtx(1)}, algorithm_params = {"scheme": scheme, "clip": clip, "threshold": threshold}, trust_params = {"info": info}, adversary_params = {"which_adversaries": which_adversaries}, rep_params = {"replicate": rep})
        # df_replicate["num_adversaries"] = num_adversaries
        df_out = pd.concat([df_out, df_replicate], ignore_index = True)

print("\a")

df = (
    df_out
        .query("adversary_type == 'friendly'")
        .groupby(["scheme", "n"], as_index = False)[["F1", "beta_rel_norm"]]
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

sns.lineplot(
    data = df,
    x = "n",
    y = "F1",
    hue = "scheme",
    palette = color_map    
)

plt.show()
plt.close()

sns.lineplot(
    data = df,
    x = "n",
    y = "beta_rel_norm",
    hue = "scheme",
    palette = color_map    
)

plt.show()
plt.close()