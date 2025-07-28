from experiment import *
from generate_data import *
from itertools import product

# initialize network topology
K = 10
adjacency_matrix = np.ones([K, K], dtype = int)
adjacency_matrix[5, 6] = 0
adjacency_matrix[6, 5] = 0
adjacency_matrix[5, 7] = 0
adjacency_matrix[7, 5] = 0
adjacency_matrix[0, 7] = 0
adjacency_matrix[7, 0] = 0
adjacency_matrix[1, 9] = 0
adjacency_matrix[9, 1] = 0

# fixed parameters
fixed_params = {
    "adjacency_matrix": adjacency_matrix,
    "n": 50,
    "p": 100,
    "max_step_size": 1,
    "n_iter": 100,
    "SNR": 2,
    "sparsity": 0.05,
    "start": "identical",
    "beta0": None,
    "corrupt_fraction": 1
}

# varying parameters
varying_params = {
    "scheme": ["G", "A", "(AC)", "CG", "GC", "(GC)"],
    "trust": ["None", "Gradient", "Model", "Both"],
    "which_adversaries": [[], [0, 1, 2, 3, 4]]
}

varying_params_keys = varying_params.keys()
varying_params_values = varying_params.values()

# parameters unique to grid search
n_rep_grid_search = 3
metric = "rel_norm"
thresholds = 10 ** np.arange(-1, 1.1, 0.5)
seed_grid_search = 1234

# parameters unique to experiment
n_rep_experiment = 10
seed_experiment = 12345

# main loop
results = {}

for instantiated_values in product(*varying_params_values):
    instantiated_params = dict(zip(varying_params_keys, instantiated_values))
    full_params = {**fixed_params, **instantiated_params}
    best_lambda = grid_search(**full_params, n_rep = n_rep_grid_search, metric = metric, thresholds = thresholds, seed = seed_grid_search)
    rel_norm_mean, rel_norm_se, F1_mean, F1_se = run_experiment(**full_params, n_rep = n_rep_experiment, threshold = best_lambda, seed = seed_experiment)
    results[tuple(sorted(instantiated_params.items()))] = rel_norm_mean

print("\a")

# TODO: why are rel_norm_means different for G and (GC)