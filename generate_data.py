import numpy as np
import random

def generate_beta_true(p, sparsity = 0.05):
    beta_support = random.sample(range(p), round(p * sparsity))
    beta_true = np.zeros(p)
    for i in beta_support:
        beta_true[beta_support, ] = 1
    return beta_true

def generate_X_Y(n, p, beta_true, SNR = 1):
    sigma = np.dot(beta_true, beta_true / SNR) ** 0.5
    X = np.random.multivariate_normal(np.zeros(p), np.identity(p), n)
    epsilon = np.random.normal(0, sigma, n)
    Y = X @ beta_true + epsilon
    return X, Y