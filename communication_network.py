import numpy as np
from numpy import linalg as LA
from client import Client, Adversary
from generate_data import *
from diagnostic import *

class Communication_Network:
    # topology_params: adjacency_matrix
    # data_params: X, Y
    # adversary_params: which_adversaries, corrupt_fraction
    def __init__(self, topology_params, data_params, adversary_params):
        adjacency_matrix = topology_params["adjacency_matrix"]
        X = data_params["X"]
        Y = data_params["Y"]
        self.comm_graph = []
        self.K = np.shape(adjacency_matrix)[0]
        self.p = np.shape(X[0])[1]
        which_adversaries = adversary_params["which_adversaries"]
        for k in range(self.K):
            neighbors = list(np.where(adjacency_matrix[k, :] == 1)[0])
            topology_params = {"client_id": k, "neighbors": neighbors}
            data_params = {"X": X[k], "Y": Y[k]}
            if(k in which_adversaries): 
                self.comm_graph.append(Adversary(topology_params = topology_params, data_params = data_params, adversary_params = adversary_params))
            else:
                self.comm_graph.append(Client(topology_params = topology_params, data_params = data_params))
            
    def BROADCAST(self):
        for i in range(self.K):
            self.comm_graph[i].compute_gradient()
            for j in self.comm_graph[i].neighbors:
                self.comm_graph[j].betas_temp[i] = np.copy(self.comm_graph[i].betas_temp[i]) # models
                self.comm_graph[j].gradients[i] = np.copy(self.comm_graph[i].gradients[i]) # gradients
                
    # start_params: start, beta0
    def initialize_start(self, start_params):
        start = start_params["start"]
        beta0 = start_params["beta0"]
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
    
    # algorithm_params: scheme, max_step_size, n_iter, threshold
    # start_params: start, beta0
    # trust_params: trust
    # diagnostic_params: beta_true
    def run_algorithm(self, algorithm_params, start_params, trust_params, diagnostic_params):
        scheme = algorithm_params["scheme"]
        max_step_size = algorithm_params["max_step_size"]
        n_iter = algorithm_params["n_iter"]
        threshold = algorithm_params["threshold"]
        start = start_params["start"]
        beta0 = start_params["beta0"]
        trust = trust_params["trust"]
        beta_true = diagnostic_params["beta_true"]
        
        self.initialize_start(start_params)
        if(trust == "None"):
            for iteration in range(n_iter):
                self.BROADCAST()
                for i in range(self.K):
                    self.comm_graph[i].select_step_size(scheme, iteration, max_step_size, threshold)
        else:
            F1_history = [[[]] for _ in range(self.K)]
            rel_norm_history = [[] for _ in range(self.K)]
            for iteration in range(n_iter):
                self.BROADCAST()
                for i in range(self.K):
                    self.comm_graph[i].update_trust(trust)
                    self.comm_graph[i].select_step_size(scheme, iteration, max_step_size, threshold)
                    self.comm_graph[i].update_betas_old()
                    
                    if(beta_true is not None):
                        F1_history[i].append(F1(confusion_matrix(beta_true, self.comm_graph[i].beta_curr)))
                        rel_norm_history[i].append(rel_norm(beta_true, self.comm_graph[i].beta_curr))
                    
        return F1_history, rel_norm_history