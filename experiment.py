from client import Client
from generate_data import *

random.seed(1)

K = 10
n = 50
p = 100
beta_true = generate_beta_true(p)
beta_0 = np.zeros_like(beta_true)
betas = {i : beta_0 for i in range(K)}
gradients = {i : beta_0 for i in range(K)}
a = {i : 1 for i in range(K)}
b = {i : 0 for i in range(K)}

# initialize communication graph
comm_graph = []
for k in range(K):
    X, Y = generate_X_Y(n, p, beta_true)
    comm_graph.append(Client(k, 0, X, Y, betas, gradients, a, b, 10**-0.5, {10**1, 10**0.5, 10**0, 10**-0.5, 10**-1}))

# main
for iteration in range(100):
    for i in range(K):
        for j in range(K):
            comm_graph[j].betas[i] = comm_graph[i].betas[i]
    for i in range(K):
        for j in range(K):
            comm_graph[i].gradients[j] = comm_graph[j].compute_gradient(i)
    for i in range(K):
        comm_graph[i].betas[i] = comm_graph[i].select_step_size()

np.nonzero(comm_graph[0].betas[0])
np.nonzero(beta_true)
