from communication_network import Communication_Network
from diagnostic import *
from generate_data import *

def grid_search(adjacency_matrix, n = 50, p = 100, n_rep = 3, max_step_size = 1, metric = "rel_norm", n_iter = 100, scheme = "G", seed = 1234, SNR = 2, sparsity = 0.05, start = "identical", beta0 = None, thresholds = 10 ** np.arange(-1, 1.1, 0.5)):
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
            
        comm_graph = Communication_Network(adjacency_matrix, X, Y)
        
        for lamb in range(len(thresholds)):
            comm_graph.run_algorithm(beta0, max_step_size, n_iter, scheme, start, threshold = thresholds[lamb])
                
            rel_norm_out = 0
            confusion_matrix_out = np.zeros([2, 2])
            for k in range(K):
                rel_norm_out += rel_norm(beta_true, comm_graph.comm_graph[k].beta_curr)
                confusion_matrix_out += confusion_matrix(beta_true, comm_graph.comm_graph[k].beta_curr)
            REL_NORMS[lamb] += rel_norm_out / K
            F1S[lamb] += F1(confusion_matrix_out)
            
        print(round(100 * (replicate + 1) / n_rep), "%")
            
    if metric == "rel_norm":
        best_norm = np.where(REL_NORMS == REL_NORMS.min())
        return thresholds[np.argmax(F1S)]
    elif metric == "F1":
        best_norm = np.where(REL_NORMS == F1S.min())
        return thresholds[np.argmax(REL_NORMS)]

def run_experiment(adjacency_matrix, n = 50, p = 100, n_rep = 10, max_step_size = 1, n_iter = 100, scheme = "G", seed = 12345, SNR = 2, sparsity = 0.05, start = "identical", beta0 = None, threshold = 10**-0.5):
    K = np.shape(adjacency_matrix)[0]
    REL_NORMS = []
    F1S = []
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
            
        comm_graph = Communication_Network(adjacency_matrix, X, Y)
        
        comm_graph.run_algorithm(beta0, max_step_size, n_iter, scheme, start, threshold)
            
        rel_norm_out = 0
        confusion_matrix_out = np.zeros([2, 2])
        for k in range(K):
            rel_norm_out += rel_norm(beta_true, comm_graph.comm_graph[k].beta_curr)
            confusion_matrix_out += confusion_matrix(beta_true, comm_graph.comm_graph[k].beta_curr)
        REL_NORMS.append(rel_norm_out / K)
        F1S.append(F1(confusion_matrix_out))
        
        print(round(100 * (replicate + 1) / n_rep), "%")
    
    return np.mean(REL_NORMS), np.std(REL_NORMS) / n_rep ** 0.5, np.mean(F1S), np.std(F1S) / n_rep ** 0.5