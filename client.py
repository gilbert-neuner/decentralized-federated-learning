import numpy as np
from numpy import linalg as LA

class Client:
    """Represent a client."""
    
    def __init__(self, client_id, neighbors, X, Y, betas, gradients, a, b, threshold, step_sizes):
        """
        Initialize a client.

        Parameters
        ----------
        client_id : int
            Id of client.
        neighbors : set
            Set of neighbor id's.
        X : np.ndarray
            Matrix of shape (n, p).
        Y : np.ndarray
            Vector of shape (1, ).
        betas : dict
            Dict, keys are id's, and values are vectors of shape (p, ).
        gradients : dict
            Dict, keys are id's, and values are vectors of shape (p, ).
        a : dict
            Dict, keys are id's, and values are floats.
        b : dict
            Dict, keys are id's, and values are floats.
        threshold : float
            Soft-thresholding parameter.
        step_sizes : set
            Set of step sizes to attempt at each iteration.

        Returns
        -------
        None.

        """
        
        self.client_id = client_id
        self.neighbors = neighbors
        self.X = X
        self.Y = Y
        self.betas = betas
        self.gradients = gradients
        self.a = a
        self.b = b
        self.threshold = threshold
        self.step_sizes = step_sizes
    
    def compute_gradient(self, i):
        """
        Compute gradient of local objective function evaluated at i-th current local model.

        Parameters
        ----------
        i : int
            Id of client's model at which gradient is evaluated.

        Returns
        -------
        np.ndarray
            Vector of shape (p, ).

        """
        
        return np.transpose(self.X) @ (self.X @ self.betas[i] - self.Y) / np.shape(self.X)[0]
    
    def update_a_b(self):
        pass
    
    def compute_trust(self, j):
        return self.a[j] / (self.a[j] + self.b[j])
    
    def compute_similarity(self, j):
        return 0.5 * (np.dot(self.gradients[self.client_id], self.gradients[j]) + 1)
    
    def compute_importance(self, j):
        trust = self.compute_trust(j)
        similarity = self.compute_similarity(j)
        return similarity ** (-1 + 1 / trust)
    
    def aggregate_gradient(self, beta, step_size):
        aggregated_gradient = np.zeros_like(self.gradients[self.client_id])
        normalizing_constant = 0
        for j in self.gradients:
            importance = self.compute_importance(j)
            aggregated_gradient += importance * self.gradients[j] * np.minimum(1, LA.norm(self.gradients[self.client_id]) / LA.norm(self.gradients[j]))
            normalizing_constant += importance
        return beta - step_size * aggregated_gradient / normalizing_constant
        
    def update_model(self, beta, step_size):
        return np.sign(beta) * np.maximum(np.abs(beta) - step_size * self.threshold, 0.0)
    
    def objective_function(self, beta):
        return 1 / (2 * np.shape(self.X)[0]) * LA.norm(self.Y - self.X @ beta) ** 2 + self.threshold * LA.norm(beta, ord = 1)
    
    def select_step_size(self):
        beta_old = self.betas[self.client_id]
        beta_incumbent = beta_old
        f_incumbent = self.objective_function(beta_incumbent)
        for alpha in self.step_sizes:
            beta_challenger = self.aggregate_gradient(beta_old, alpha)
            beta_challenger = self.update_model(beta_challenger, alpha)
            f_challenger = self.objective_function(beta_challenger)
            if(f_challenger < f_incumbent):
                beta_incumbent = beta_challenger
                f_incumbent = f_challenger
        return beta_incumbent