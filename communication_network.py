import numpy as np
from client import Client

class Communication_Network:
    # REWRITE THIS IN THE FUTURE SO BETAS, A, B, THRESHOLD, STEP_SIZES DON'T HAVE TO BE SHARED
    def __init__(self, adjacency_matrix, beta0, X, Y, a, b, threshold = 10**-0.5, max_step_size = 1, step_size_tol = 10**-5):
        self.comm_graph = []
        self.K = np.shape(adjacency_matrix)[0]
        for k in range(self.K):
            neighbors = set(np.where(adjacency_matrix[k, :] == 1)[0])
            self.comm_graph.append(Client(client_id = k, neighbors = neighbors, X = X[k], Y = Y[k], beta_curr = beta0, a = a, b = b, threshold = threshold, max_step_size = max_step_size, step_size_tol = step_size_tol))
    
    # i sends to its neighbors
    def broadcast_models(self):
        for i in range(self.K):
            for j in self.comm_graph[i].neighbors:
                self.comm_graph[j].betas_temp[i] = np.copy(self.comm_graph[i].betas_temp[i])
    
    # j sends to its neighbors
    def compute_and_broadcast_gradients(self):
        for j in range(self.K):
            for i in self.comm_graph[j].neighbors:
                self.comm_graph[i].gradients[j] = self.comm_graph[j].compute_gradient(i)
    
    def run_experiment(self, n_iter, scheme):
        if scheme == "threshold":
            for iteration in range(n_iter):
                for k in range(self.K):
                    self.comm_graph[k].select_step_size(scheme = scheme)
        elif scheme == "gradients_threshold":
            for iteration in range(n_iter):
                self.broadcast_models()
                self.compute_and_broadcast_gradients()
                for k in range(self.K):
                    self.comm_graph[k].select_step_size(scheme = scheme)
            
            