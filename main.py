from experiment import *
from generate_data import *

K = 10
shape = "mesh"

adjacency_matrix = generate_adj_mtx(K = K, shape = shape)

scheme = "(GC)"

which_adversaries = [0, 4]

corrupt_fraction = 1

best_lambda = grid_search(adjacency_matrix, metric = "F1", scheme = scheme, which_adversaries = which_adversaries, corrupt_fraction = corrupt_fraction)

rel_norm_mean, rel_norm_se, F1_mean, F1_se = run_experiment(adjacency_matrix, n_rep = 10, scheme = scheme, threshold = best_lambda, which_adversaries = which_adversaries, corrupt_fraction = corrupt_fraction)

print("\a")

print(F1_mean)
