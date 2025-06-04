from communication_network import Communication_Network
from diagnostic import *
from generate_data import *

def grid_search(adjacency_matrix, n = 50, p = 100, n_rep = 3, max_step_size = 1, metric = "rel_norm", n_iter = 100, scheme = "G", seed = 1234, SNR = 2, sparsity = 0.05, start = "identical", beta0 = None, thresholds = 10 ** np.arange(-1, 1.1, 0.5), which_adversaries = [], corrupt_fraction = 1):
    K = np.shape(adjacency_matrix)[0]
    REL_NORMS = np.zeros(len(thresholds))
    F1S = np.zeros(len(thresholds))
    
    for replicate in range(n_rep):
        random.seed(replicate + seed)
        np.random.seed(replicate + seed)
        
        beta_true = generate_beta_true(p, sparsity)
        X = []
        Y = []
        for k in range(K):
            x, y = generate_X_Y(n, beta_true, SNR)
            X.append(x)
            Y.append(y)
            
        comm_graph = Communication_Network(adjacency_matrix, X, Y, which_adversaries, corrupt_fraction)
        
        for lamb in range(len(thresholds)):
            comm_graph.run_algorithm(beta0, max_step_size, n_iter, scheme, start, threshold = thresholds[lamb])
                
            rel_norm_out = 0
            confusion_matrix_out = np.zeros([2, 2])
            which_benign = [i for i in range(K) if i not in which_adversaries]
            for k in which_benign:
                rel_norm_out += rel_norm(beta_true, comm_graph.comm_graph[k].beta_curr)
                confusion_matrix_out += confusion_matrix(beta_true, comm_graph.comm_graph[k].beta_curr)
            REL_NORMS[lamb] += rel_norm_out / len(which_benign)
            F1S[lamb] += F1(confusion_matrix_out)
            
        print(round(100 * (replicate + 1) / n_rep), "%")
            
    if metric == "rel_norm":
        candidates = np.where(REL_NORMS == REL_NORMS.min())[0]
        best_index = candidates[np.argmax(F1S[candidates])]
        return thresholds[best_index]
    elif metric == "F1":
        candidates = np.where(F1S == F1S.max())[0]
        best_index = candidates[np.argmin(REL_NORMS[candidates])]
        return thresholds[best_index]

def run_experiment(adjacency_matrix, n = 50, p = 100, n_rep = 10, max_step_size = 1, n_iter = 100, scheme = "G", seed = 12345, SNR = 2, sparsity = 0.05, start = "identical", beta0 = None, threshold = 10**-0.5, which_adversaries = [], corrupt_fraction = 1):
    K = np.shape(adjacency_matrix)[0]
    REL_NORMS = np.zeros((K, n_rep))
    F1S = np.zeros((K, n_rep))
    for replicate in range(n_rep):
        random.seed(replicate + seed)
        np.random.seed(replicate + seed)
        
        beta_true = generate_beta_true(p, sparsity)
        X = []
        Y = []
        for k in range(K):
            x, y = generate_X_Y(n, beta_true, SNR)
            X.append(x)
            Y.append(y)
            
        comm_graph = Communication_Network(adjacency_matrix, X, Y, which_adversaries, corrupt_fraction)
        
        comm_graph.run_algorithm(beta0, max_step_size, n_iter, scheme, start, threshold)
            
        rel_norm_out = 0
        confusion_matrix_out = np.zeros([2, 2])
        for k in range(K):
            REL_NORMS[k, replicate] = rel_norm(beta_true, comm_graph.comm_graph[k].beta_curr)
            F1S[k, replicate] = F1(confusion_matrix(beta_true, comm_graph.comm_graph[k].beta_curr))
        
        print(round(100 * (replicate + 1) / n_rep), "%")
    
    return np.mean(REL_NORMS, axis = 1), np.std(REL_NORMS, axis = 1) / n_rep ** 0.5, np.mean(F1S, axis = 1), np.std(F1S, axis = 1) / n_rep ** 0.5