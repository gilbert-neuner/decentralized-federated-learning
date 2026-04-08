import numpy as np
from client import Client, Adversary
from generate_data import generate_adj_mtx

def unpack_params(param_dict, k = 0):
    out = {}
    for key, val in param_dict.items():
        try:
            out[key] = val[k]
            if isinstance(out[key], str):
                out[key] = val
        except Exception:
            out[key] = val
    return out
        
class Communication_Network:
    # data_params: n, p, SNR, sparsity, beta_true
    # topology_params: adjacency_matrix
    # algorithm_params: scheme, max_step_size, n_iter, threshold, beta0, random_displace, winsorize
    # trust_params: info, accelerate, cosine_recompute, include
    # adversary_params: which_adversaries, corrupt_fraction, jitter, adversary_type
    def __init__(self, data_params, topology_params, algorithm_params = {}, trust_params = {}, adversary_params = {}):
        # topology_params
        self.adjacency_matrix = topology_params.get("adjacency_matrix", generate_adj_mtx())
        self.K = np.shape(self.adjacency_matrix)[0]
        self.comm_graph = []
        # algorithm_params
        self.algorithm_params = algorithm_params
        # adversary_params
        self.which_adversaries = adversary_params.get("which_adversaries", range(0, self.K, 2))
        self.which_friendly = [i for i in range(self.K) if i not in self.which_adversaries]
        # initialize each client
        for k in range(self.K):
            # client data_params
            data_params_k = unpack_params(data_params, k)
            # client topology_params
            neighbors = list(np.where(self.adjacency_matrix[k, :] == 1)[0])
            topology_params_k = {"client_id": k, "neighbors": neighbors}
            # client algorithm_params
            algorithm_params_k = unpack_params(algorithm_params, k)
            # client trust_params
            trust_params_k = unpack_params(trust_params, k)
            # client adversary_params
            adversary_params_k = unpack_params(adversary_params, k)
            if(k in self.which_adversaries):
                self.comm_graph.append(Adversary(data_params_k, topology_params_k, algorithm_params_k, trust_params_k, adversary_params_k))
            else:
                self.comm_graph.append(Client(data_params_k, topology_params_k, algorithm_params_k, trust_params_k))
                
    def BROADCAST(self):
        for i in range(self.K):
            self.comm_graph[i].compute_gradient()
            for j in self.comm_graph[i].neighbors:
                self.comm_graph[j].betas_temp[i] = np.copy(self.comm_graph[i].betas_temp[i]) # models
                self.comm_graph[j].gradients[i] = np.copy(self.comm_graph[i].gradients[i]) # gradients
                
    def run_algorithm(self):
        for iteration in range(unpack_params(self.algorithm_params).get("n_iter", 100)):
            self.BROADCAST()
            for k in range(self.K):
                self.comm_graph[k].update_trust()
                if self.comm_graph[k].winsorize > 0:
                    self.comm_graph[k].WINSORIZE()
                self.comm_graph[k].select_step_size()
                self.comm_graph[k].update_betas_old()