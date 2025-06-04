import numpy as np
from numpy import linalg as LA
from client import Client, Adversary
from generate_data import *

class Communication_Network:
    def __init__(self, adjacency_matrix, X, Y, which_adversaries, corrupt_fraction):
        self.comm_graph = []
        self.K = np.shape(adjacency_matrix)[0]
        self.p = np.shape(X[0])[1]
        for k in range(self.K):
            neighbors = list(np.where(adjacency_matrix[k, :] == 1)[0])
            if(k in which_adversaries):                
                self.comm_graph.append(Adversary(client_id = k, neighbors = neighbors, X = X[k], Y = Y[k], corrupt_fraction = corrupt_fraction))
            else:
                self.comm_graph.append(Client(client_id = k, neighbors = neighbors, X = X[k], Y = Y[k]))
            
    def BROADCAST(self):
        for i in range(self.K):
            self.comm_graph[i].compute_gradient()
            for j in self.comm_graph[i].neighbors:
                self.comm_graph[j].betas_temp[i] = np.copy(self.comm_graph[i].betas_temp[i]) # models
                self.comm_graph[j].gradients[i] = np.copy(self.comm_graph[i].gradients[i]) # gradients
                
    def initialize_start(self, start = "identical", beta0 = None):
        if(beta0 is None):
            beta0 = np.zeros(self.p)
        if(start == "identical"):
            for k in range(self.K):
                self.comm_graph[k].beta_curr = beta0
                self.comm_graph[k].betas_temp[k] = beta0
        elif(start == "random"):
            for k in range(self.K):
                displacement = np.random.uniform(-1, 1, 100)
                displacement *= 5 / LA.norm(displacement)
                self.comm_graph[k].beta_curr = beta0 + displacement
                self.comm_graph[k].betas_temp[k] = beta0 + displacement
    
    def run_algorithm(self, beta0 = None, max_step_size = 1, n_iter = 100, scheme = "G", start = "identical", threshold = 10**-0.5):
        self.initialize_start(start, beta0)
        for iteration in range(n_iter):
            self.BROADCAST()
            for i in range(self.K):
                self.comm_graph[i].select_step_size(scheme, iteration, max_step_size, threshold)