import numpy as np
from generate_data import *
from communication_network import Communication_Network
from diagnostic import *

K = 10
adjacency_matrix = np.ones([K, K], dtype = int)

random.seed(1)

p = 100
beta_true = generate_beta_true(p)
beta0 = np.zeros_like(beta_true)

n = 50
X = []
Y = []
for k in range(K):
    x, y = generate_X_Y(n, p, beta_true)
    X.append(x)
    Y.append(y)
    
a = {k : 0.5 for k in range(K)}
b = a

threshold = 10**0.5 # 10**-0.5

max_step_size = 1 # 1
step_size_tol = 10**-2 # 10**-5

comm_graph = Communication_Network(adjacency_matrix, beta0, X, Y, a, b, threshold, max_step_size, step_size_tol)

comm_graph.run_experiment(100, "threshold")

comm_graph.comm_graph[0].beta_curr

def rule(x):
    return x > 0
    
confusion_matrix(rule, beta_true, comm_graph.comm_graph[0].beta_curr)

F1(rule, beta_true, comm_graph.comm_graph[0].beta_curr)
