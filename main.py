# import numpy as np
# from numpy import linalg as LA
# import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
from itertools import product
from experiment import grid_search, replicate_algorithm
# from generate_data import generate_adj_mtx, generate_beta_true
# import random

df_out = pd.DataFrame(columns = ["scheme", "max_step_size", "n_iter", "threshold", "random_displace", "winsorize", "info", "accelerate", "include", "n", "p", "SNR", "sparsity", "adversary_type", "corrupt_fraction", "jitter", "seed", "client_id", "F1", "beta_rel_norm", "Y_rel_norm"])

for (scheme, ) in product(["G", "AG", "A,G", "G,A", "S", "AS"]): 
    threshold = grid_search()
    df_replicate, trust_replicate, beta_true = replicate_algorithm(algorithm_params = {"scheme": scheme, "threshold": threshold})
    df_out = pd.concat([df_out, df_replicate], ignore_index = True)
    
print("\a")

df = (
    df_out
        .query("adversary_type == 'friendly'")
        .groupby(["scheme"], as_index = False)[["F1", "beta_rel_norm", "Y_rel_norm"]]
        .mean()
)