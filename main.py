from experiment import *
from generate_data import *

K = 10
# shape = "ring"

adjacency_matrix = np.ones([K, K], dtype = int)
adjacency_matrix[5, 6] = 0
adjacency_matrix[6, 5] = 0
adjacency_matrix[5, 7] = 0
adjacency_matrix[7, 5] = 0
adjacency_matrix[0, 7] = 0
adjacency_matrix[7, 0] = 0
adjacency_matrix[1, 9] = 0
adjacency_matrix[9, 1] = 0

metric = "rel_norm"

scheme = "(GC)"

trust = "Both"

which_adversaries = [0, 1, 2, 3, 4]

corrupt_fraction = 1

best_lambda = grid_search(adjacency_matrix, metric = metric, scheme = scheme, trust = trust, which_adversaries = which_adversaries, corrupt_fraction = corrupt_fraction)

rel_norm_mean, rel_norm_se, F1_mean, F1_se = run_experiment(adjacency_matrix, n_rep = 10, scheme = scheme, trust = trust, threshold = best_lambda, which_adversaries = which_adversaries, corrupt_fraction = corrupt_fraction)

print("\a")

print(rel_norm_mean)

# TODO: why are rel_norm_means different for G and (GC)