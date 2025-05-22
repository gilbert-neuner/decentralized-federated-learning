from experiment import *
from generate_data import *

K = 10
shape = "ring"

adjacency_matrix = generate_adj_mtx(K = K, shape = shape)

scheme = "(GC)"

start = "identical"

best_lambda = grid_search(adjacency_matrix, scheme = scheme, start = start)

rel_norm_mean, rel_norm_se, F1_mean, F1_se = run_experiment(adjacency_matrix, n_rep = 10, scheme = scheme, start = start, threshold = best_lambda)

print("\a")