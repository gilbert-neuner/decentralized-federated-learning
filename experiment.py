import numpy as np
import random
from communication_network import Communication_Network
from generate_data import generate_beta_true, generate_X_Y, generate_adj_mtx
from diagnostic import confusion_matrix, F1, rel_norm
import pandas as pd
            
# data_params: n, p, SNR, sparsity
# topology_params: adjacency_matrix
# algorithm_params: scheme, max_step_size, n_iter, threshold, beta0, random_displace, winsorize
# trust_params: info, accelerate, cosine_recompute, include
# adversary_params: which_adversaries, corrupt_fraction, adversary_type
# rep_params: replicate, seed
def replicate_algorithm(data_params = {}, topology_params = {}, algorithm_params = {}, trust_params = {}, adversary_params = {}, rep_params = {}):
    df_out = pd.DataFrame(columns = ["scheme", "max_step_size", "n_iter", "threshold", "random_displace", "info", "accelerate", "include", "n", "p", "SNR", "sparsity", "adversary_type", "corrupt_fraction", "seed", "client_id", "F1", "beta_rel_norm", "Y_rel_norm"])
    # data_params
    n = data_params.get("n", 50)
    p = data_params.get("p", 100)
    SNR = data_params.get("SNR", 1)
    sparsity = data_params.get("sparsity", 0.05)
    # topology_params
    adjacency_matrix = topology_params.get("adjacency_matrix", generate_adj_mtx())
    K = np.shape(adjacency_matrix)[0]
    trust_out = np.zeros([K, K])
    # rep_params
    replicate = rep_params.get("replicate", 0)
    seed = rep_params.get("seed", 12345)

    # SET SEED
    random.seed(replicate + seed)
    np.random.seed(replicate + seed)
    # data_params
    beta_true = generate_beta_true(p, sparsity)
    X, Y = generate_X_Y(K, n, beta_true, SNR)
    data_params_curr = {"X": X, "Y": Y}
    comm_graph = Communication_Network(data_params_curr, topology_params, algorithm_params, trust_params, adversary_params)
    comm_graph.run_algorithm()
    
    for k in range(K):
        new_row = {}
        new_row["scheme"] = [comm_graph.comm_graph[k].scheme]
        new_row["max_step_size"] = [comm_graph.comm_graph[k].max_step_size]
        new_row["n_iter"] = [comm_graph.comm_graph[k].n_iter]
        new_row["threshold"] = [comm_graph.comm_graph[k].threshold]
        new_row["random_displace"] = [comm_graph.comm_graph[k].random_displace]
        new_row["winsorize"] = [comm_graph.comm_graph[k].winsorize]
        new_row["info"] = [comm_graph.comm_graph[k].info]
        new_row["accelerate"] = [comm_graph.comm_graph[k].accelerate]
        new_row["include"] = [comm_graph.comm_graph[k].include]
        new_row["n"] = [comm_graph.comm_graph[k].n]
        new_row["p"] = [comm_graph.comm_graph[k].p]
        new_row["SNR"] = [SNR]
        new_row["sparsity"] = [sparsity]
        new_row["adversary_type"] = [comm_graph.comm_graph[k].adversary_type]
        new_row["corrupt_fraction"] = [comm_graph.comm_graph[k].corrupt_fraction]
        new_row["seed"] = [replicate + seed]
        new_row["client_id"] = [comm_graph.comm_graph[k].client_id]
        new_row["F1"] = [F1(confusion_matrix(beta_true, comm_graph.comm_graph[k].beta_curr))]
        new_row["beta_rel_norm"] = [rel_norm(beta_true, comm_graph.comm_graph[k].beta_curr)]
        new_row["Y_rel_norm"] = [rel_norm(comm_graph.comm_graph[k].Y, comm_graph.comm_graph[k].X @ comm_graph.comm_graph[k].beta_curr)]
        new_row = pd.DataFrame(new_row)
        df_out = pd.concat([df_out, new_row], ignore_index = True)
        
        for k_neighbor in comm_graph.comm_graph[k].neighbors:
            trust_out[k, k_neighbor] = comm_graph.comm_graph[k].trust[k_neighbor]

    return df_out, trust_out, beta_true

# data_params: n, p, SNR, sparsity
# topology_params: adjacency_matrix
# algorithm_params: scheme, max_step_size, n_iter, beta0, random_displace
# trust_params: info, accelerate, cosine_recompute, include
# adversary_params: which_adversaries, corrupt_fraction, adversary_type
# grid_params: n_rep, seed, grid, shared_threshold, metric

def grid_search(data_params = {}, topology_params = {}, algorithm_params = {}, trust_params = {}, adversary_params = {}, grid_params = {}):
    # topology_params
    adjacency_matrix = topology_params.get("adjacency_matrix", generate_adj_mtx())
    K = np.shape(adjacency_matrix)[0]
    # grid_params
    n_rep = grid_params.get("n_rep", 1)
    seed = grid_params.get("seed", 1234)
    grid = grid_params.get("grid", 10 ** np.arange(-1, 1.1, 0.5))
    MEASUREMENTS = np.zeros([len(grid), K])
    shared_threshold = grid_params.get("shared_threshold", False)
    metric = grid_params.get("metric", "Y")
    
    for replicate in range(n_rep):
        for threshold in grid:
            algorithm_params["threshold"] = threshold
            rep_params = {"replicate": replicate, "seed": seed}
            df_out, _, _ = replicate_algorithm(data_params, topology_params, algorithm_params, trust_params, adversary_params, rep_params)
            # adversary_params
            which_friendly = df_out.index[df_out["adversary_type"] == "friendly"]
            for k in which_friendly:
                if metric == "F1":
                    MEASUREMENTS[threshold, k] += df_out.loc[k, "F1"]
                elif metric == "beta_rel_norm":
                    MEASUREMENTS[threshold, k] += df_out.loc[k, "beta_rel_norm"]
                elif metric == "Y_rel_norm":
                    MEASUREMENTS[threshold, k] += df_out.loc[k, "Y_rel_norm"]
            
        print(round(100 * (replicate + 1) / n_rep), "%")
        
        if shared_threshold:
            if metric == "F1":
                return grid[np.argmax(np.sum(MEASUREMENTS, axis = 1), axis = 0)]
            else:
                return grid[np.argmin(np.sum(MEASUREMENTS, axis = 1), axis = 0)]
        else:
            if metric == "F1":
                global_threshold = grid[np.argmax(np.sum(MEASUREMENTS, axis = 1))]
                return [(grid[np.argmax(MEASUREMENTS, axis = 0)[i]] if i in which_friendly else global_threshold) for i in range(K)]
            else:
                global_threshold = grid[np.argmin(np.sum(MEASUREMENTS, axis = 1))]
                return [(grid[np.argmin(MEASUREMENTS, axis = 0)[i]] if i in which_friendly else global_threshold) for i in range(K)]
