import numpy as np
import random

def generate_beta_true(p, sparsity = 0.05):
    beta_support = random.sample(range(p), round(p * sparsity))
    beta_true = np.zeros(p)
    for i in beta_support:
        beta_true[beta_support, ] = 1
    return beta_true

def generate_X_Y(n, beta_true, SNR = 1):
    p = np.shape(beta_true)[0]
    sigma = np.dot(beta_true, beta_true / SNR) ** 0.5
    X = np.random.multivariate_normal(np.zeros(p), np.identity(p), n)
    epsilon = np.random.normal(0, sigma, n)
    Y = X @ beta_true + epsilon
    return X, Y

def generate_adj_mtx(K = 10, shape = "mesh"):
    if shape == "mesh":
        adjacency_matrix = np.ones([K, K], dtype = int)
    elif shape == "line":
        adjacency_matrix = np.zeros([K, K], dtype = int)
        for i in range(K):
            for j in range(K):
                if abs(i - j) <= 1:
                    adjacency_matrix[i, j] = 1
    elif shape == "ring":
        adjacency_matrix = generate_adj_mtx(K, shape = "line")
        adjacency_matrix[0, K - 1] = 1
        adjacency_matrix[K - 1, 0] = 1
    elif shape == "star":
        adjacency_matrix = np.zeros([K, K], dtype = int)
        for k in range(K):
            adjacency_matrix[0, k] = 1
            adjacency_matrix[k, 0] = 1
            adjacency_matrix[k, k] = 1
    return adjacency_matrix
        