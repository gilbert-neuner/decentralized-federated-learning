from communication_network import Communication_Network
from diagnostic import *
from generate_data import *

# topology_params: adjacency_matrix
# data_params: n, p, SNR, sparsity
# algorithm_params: scheme, max_step_size, n_iter
# grid_params: n_rep, thresholds, metric, seed
# start_params: start, beta0
# trust_params: trust
# adversary_params: which_adversaries, corrupt_fraction
def grid_search(topology_params, data_params, algorithm_params, grid_params, start_params, trust_params, adversary_params):
    adjacency_matrix = topology_params["adjacency_matrix"]
    n = data_params["n"]
    p = data_params["p"]
    SNR = data_params["SNR"]
    sparsity = data_params["sparsity"]
    scheme = algorithm_params["scheme"]
    max_step_size = algorithm_params["max_step_size"]
    n_iter = algorithm_params["n_iter"]
    n_rep = grid_params["n_rep"]
    thresholds = grid_params["thresholds"]
    metric = grid_params["metric"]
    seed = grid_params["seed"]
    start = start_params["start"]
    beta0 = start_params["beta0"]
    trust = trust_params["trust"]
    which_adversaries = adversary_params["which_adversaries"]
    corrupt_fraction = adversary_params["corrupt_fraction"]
    
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
            
        data_params_XY = {"X": X, "Y": Y}
        comm_graph = Communication_Network(topology_params, data_params_XY, adversary_params)
        
        for lamb in range(len(thresholds)):
            algorithm_params["threshold"] = thresholds[lamb]
            diagnostic_params = {"beta_true": beta_true}
            comm_graph.run_algorithm(algorithm_params, start_params, trust_params, diagnostic_params)
                
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

# topology_params: adjacency_matrix
# data_params: n, p, SNR, sparsity
# algorithm_params: scheme, max_step_size, n_iter, threshold
# experiment_params: n_rep, seed
# start_params: start, beta0
# trust_params: trust
# adversary_params: which_adversaries, corrupt_fraction
def run_experiment(topology_params, data_params, algorithm_params, experiment_params, start_params, trust_params, adversary_params):
    adjacency_matrix = topology_params["adjacency_matrix"]
    n = data_params["n"]
    p = data_params["p"]
    SNR = data_params["SNR"]
    sparsity = data_params["sparsity"]
    scheme = algorithm_params["scheme"]
    max_step_size = algorithm_params["max_step_size"]
    n_iter = algorithm_params["n_iter"]
    threshold = algorithm_params["threshold"]
    n_rep = experiment_params["n_rep"]
    seed = experiment_params["seed"]
    start = start_params["start"]
    beta0 = start_params["beta0"]
    trust = trust_params["trust"]
    which_adversaries = adversary_params["which_adversaries"]
    corrupt_fraction = adversary_params["corrupt_fraction"]
    
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
        
        data_params_XY = {"X": X, "Y": Y}
        comm_graph = Communication_Network(topology_params, data_params_XY, adversary_params)
        
        diagnostic_params = {"beta_true": beta_true}
        comm_graph.run_algorithm(algorithm_params, start_params, trust_params, diagnostic_params)
        
        for k in range(K):
            REL_NORMS[k, replicate] = rel_norm(beta_true, comm_graph.comm_graph[k].beta_curr)
            F1S[k, replicate] = F1(confusion_matrix(beta_true, comm_graph.comm_graph[k].beta_curr))
        
        print(round(100 * (replicate + 1) / n_rep), "%")
    
    return np.mean(REL_NORMS, axis = 1), np.std(REL_NORMS, axis = 1) / n_rep ** 0.5, np.mean(F1S, axis = 1), np.std(F1S, axis = 1) / n_rep ** 0.5

def analyze_trust_history(topology_params, data_params, algorithm_params, experiment_params, start_params, trust_params, adversary_params):
    adjacency_matrix = topology_params["adjacency_matrix"]
    n = data_params["n"]
    p = data_params["p"]
    SNR = data_params["SNR"]
    sparsity = data_params["sparsity"]
    scheme = algorithm_params["scheme"]
    max_step_size = algorithm_params["max_step_size"]
    n_iter = algorithm_params["n_iter"]
    threshold = algorithm_params["threshold"]
    n_rep = experiment_params["n_rep"]
    seed = experiment_params["seed"]
    start = start_params["start"]
    beta0 = start_params["beta0"]
    trust = trust_params["trust"]
    which_adversaries = adversary_params["which_adversaries"]
    corrupt_fraction = adversary_params["corrupt_fraction"]
    
    K = np.shape(adjacency_matrix)[0]
    
    rel_norms = np.zeros(K)
    F1s = np.zeros(K)
    gradient_history = [0 for _ in range(K)]
    model_history = [0 for _ in range(K)]
    
    random.seed(seed)
    np.random.seed(seed)
    
    beta_true = generate_beta_true(p, sparsity)
    X = []
    Y = []
    for k in range(K):
        x, y = generate_X_Y(n, beta_true, SNR)
        X.append(x)
        Y.append(y)
    
    data_params_XY = {"X": X, "Y": Y}
    comm_graph = Communication_Network(topology_params, data_params_XY, adversary_params)
    
    diagnostic_params = {"beta_true": beta_true}
    F1_history, rel_norm_history = comm_graph.run_algorithm(algorithm_params, start_params, trust_params, diagnostic_params)
    
    for k in range(K):
        rel_norms[k] = rel_norm(beta_true, comm_graph.comm_graph[k].beta_curr)
        F1s[k] = F1(confusion_matrix(beta_true, comm_graph.comm_graph[k].beta_curr))
        gradient_history[k] = comm_graph.comm_graph[k].gradient_history
        model_history[k] = comm_graph.comm_graph[k].model_history
        
    return rel_norms, F1s, gradient_history, model_history, F1_history, rel_norm_history