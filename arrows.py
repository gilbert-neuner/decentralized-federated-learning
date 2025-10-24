import numpy as np
import matplotlib.pyplot as plt
from experiment import analyze_trust_history
from generate_data import generate_adj_mtx

K = 10
topology_params = {"adjacency_matrix": generate_adj_mtx(K, shape = "mesh")}
adversary_params = {"which_adversaries": [1, 3, 5, 7, 9], "corrupt_fraction": {1:1, 3:1, 5:1, 7:1, 9:1}, "adversary_type": "local_model", "corrupt_coefficient": 1.01}
all_friendly_ids = [0, 2, 4, 6, 8]
adversary2_ids = []
adversary_ids = [1, 3, 5, 7, 9]
data_params = {"n": 1, "p": 2, "SNR": 1, "sparsity": 0.5}
algorithm_params = {"scheme": "(AC)", "max_step_size": 0.1, "n_iter": 100, "threshold": 1}
start_params = {"start": "random", "beta0": None}
trust_params = {"info": "Both", "accelerate": True, "include": 0}
experiment_params = {"n_rep": 10, "seed": 12345}

rel_norm_exp, F1_exp, gradient_history_exp, model_history_exp, F1_history_exp, rel_norm_history_exp, beta_history_exp, beta_true_exp = analyze_trust_history(topology_params, data_params, algorithm_params, experiment_params, start_params, trust_params, adversary_params)

all_client_ids = list(range(0, K))

orange_cmap = plt.get_cmap('Oranges')
blue_cmap   = plt.get_cmap('Blues')
green_cmap  = plt.get_cmap('Greens')

plt.figure()
for seq in all_client_ids:
    coords = np.stack(beta_history_exp[seq])
    x, y = coords[:, 0], coords[:, 1]
    dx = np.diff(x)
    dy = np.diff(y)
    if seq in all_friendly_ids:
        col = "green"
    elif seq in adversary_ids:
        col = "orange"
    elif seq in adversary2_ids:
        col = "orange"
    plt.quiver(x[:-1], y[:-1], dx, dy, color = col, angles='xy', scale_units='xy', scale = 1)
plt.plot(beta_true_exp[0], beta_true_exp[1], 'ro') 
plt.show()

print("\a")
    
# TODO: why are rel_norm_means different for G and (GC)