import numpy as np
from generate_data import *
from communication_network import Communication_Network
from diagnostic import *

K = 10
shape = "star"
adjacency_matrix = generate_adj_mtx(K, shape)

random.seed(1)
np.random.seed(1)

p = 100
beta_true = generate_beta_true(p)
beta0 = np.zeros_like(beta_true)

n = 50
X = []
Y = []
SNR = 1 # 1
for k in range(K):
    x, y = generate_X_Y(n, p, beta_true, SNR)
    X.append(x)
    Y.append(y)
    
a = {k : 0.5 for k in range(K)}
b = a

threshold = 10**-0.5 # 10**-0.5

comm_graph = Communication_Network(adjacency_matrix, beta0, X, Y, a, b, threshold)

max_step_size = 1 # 1

comm_graph.run_experiment(n_iter = 100, scheme = "gradients_models_threshold", max_step_size = max_step_size)

comm_graph.comm_graph[0].beta_curr

def rule(x):
    return x > 0
    
confusion_matrix(rule, beta_true, comm_graph.comm_graph[1].beta_curr)

F1(rule, beta_true, comm_graph.comm_graph[1].beta_curr)

rel_norm(beta_true, comm_graph.comm_graph[1].beta_curr)
