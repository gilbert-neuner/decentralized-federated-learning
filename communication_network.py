import numpy as np
from client import Client

class Communication_Network:
    # REWRITE THIS IN THE FUTURE SO BETAS, A, B, THRESHOLD, STEP_SIZES DON'T HAVE TO BE SHARED
    def __init__(self, adjacency_matrix, beta0, X, Y, threshold):
        self.comm_graph = []
        self.K = np.shape(adjacency_matrix)[0]
        for k in range(self.K):
            neighbors = list(np.where(adjacency_matrix[k, :] == 1)[0])
            self.comm_graph.append(Client(client_id = k, neighbors = neighbors, X = X[k], Y = Y[k], beta_curr = beta0, threshold = threshold))
            
    def BROADCAST(self):
        for i in range(self.K):
            self.comm_graph[i].compute_gradient()
            for j in self.comm_graph[i].neighbors:
                self.comm_graph[j].betas_temp[i] = np.copy(self.comm_graph[i].betas_temp[i]) # models
                self.comm_graph[j].gradients[i] = np.copy(self.comm_graph[i].gradients[i]) # gradients
    
    def run_experiment(self, n_iter, scheme, threshold = 10**-0.5, max_step_size = 1):
        for iteration in range(n_iter):
            self.BROADCAST()
            for i in range(self.K):
                self.comm_graph[i].select_step_size(scheme, iteration, max_step_size)