import numpy as np
from numpy import linalg as LA

class Client:
    """Represent a client."""
    
    def __init__(self, client_id, neighbors, X, Y, beta_curr, a, b, threshold = 10**-0.5):
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
        beta_curr : dict
            Vectors of shape (p, ).
        a : dict
            Dict, keys are id's, and values are floats.
        b : dict
            Dict, keys are id's, and values are floats.
        threshold : float
            Soft-thresholding parameter.

        Returns
        -------
        None.

        """
        
        self.client_id = client_id
        self.neighbors = neighbors
        self.X = X
        self.Y = Y
        self.beta_curr = np.copy(beta_curr)
        self.betas_temp = {neighbor : np.copy(beta_curr) for neighbor in neighbors}
        self.gradients = {neighbor : np.zeros_like(beta_curr) for neighbor in neighbors}
        self.a = a
        self.a_temp = a
        self.b = b
        self.b_temp = b
        self.threshold = threshold
    
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
        return np.transpose(self.X) @ (self.X @ self.betas_temp[i] - self.Y) / np.shape(self.X)[0]
    
    def update_a_b(self):
        pass
    
    def compute_trust(self, j):
        if j == self.client_id:
            return 1
        return self.a[j] / (self.a[j] + self.b[j])
    
    # later have to change this to be model based
    def compute_similarity(self, j):
        if j == self.client_id:
            return 1
        if LA.norm(self.gradients[self.client_id]) * LA.norm(self.gradients[j]) == 0:
            return 0
        return 0.5 * (np.dot(self.gradients[self.client_id], self.gradients[j]) / (LA.norm(self.gradients[self.client_id]) * LA.norm(self.gradients[j])) + 1)
    
    def compute_importance(self, j):
        trust = self.compute_trust(j)
        similarity = self.compute_similarity(j)
        return similarity ** (-1 + 1 / trust)
    
    def normalize_magnitude(self, u, v):
        """
        

        Parameters
        ----------
        u : np.ndarray
            A vector.
        v : np.array
            A vector with the same shape as u.

        Returns
        -------
        np.ndarray
            A vector in the direction of u, with magnitude no larger than that of v.

        """
        if LA.norm(u) == 0:
            return u
        return u * np.minimum(1, LA.norm(v) / LA.norm(u))
    
    def aggregate_gradient(self):
        aggregated_gradient = np.zeros_like(self.gradients[self.client_id])
        normalizing_constant = 0
        for j in self.gradients:
            importance = self.compute_importance(j)
            aggregated_gradient += importance * self.normalize_magnitude(self.gradients[j], self.gradients[self.client_id])
            normalizing_constant += importance
        return aggregated_gradient / normalizing_constant
        
    def apply_gradient(self, gradient, step_size):
        self.betas_temp[self.client_id] -= step_size * gradient
        
    def soft_threshold(self, step_size):
        self.betas_temp[self.client_id] = np.sign(self.betas_temp[self.client_id]) * np.maximum(np.abs(self.betas_temp[self.client_id]) - step_size * self.threshold, 0.0)
    
    def aggregate_model(self, step_size):
        aggregated_diff = np.zeros_like(self.betas_temp[self.client_id])
        normalizing_constant = 0
        for j in self.betas_temp:
            importance = self.compute_importance(j)
            aggregated_diff += importance * self.normalize_magnitude(self.betas_temp[j] - self.betas_temp[self.client_id], self.gradients[self.client_id])
            normalizing_constant += importance
        self.betas_temp[self.client_id] += step_size * aggregated_diff / normalizing_constant
        
    def objective_function(self):
        return 1 / (2 * np.shape(self.X)[0]) * LA.norm(self.Y - self.X @ self.betas_temp[self.client_id]) ** 2 + self.threshold * LA.norm(self.betas_temp[self.client_id], ord = 1)
            
    def reset_beta_temp(self):
        self.betas_temp[self.client_id] = np.copy(self.beta_curr)
        
    def update_beta_curr(self):
        self.beta_curr = np.copy(self.betas_temp[self.client_id])
        
    def select_step_size(self, scheme, curr_iter, max_step_size = 1):
        invphi = (5 ** 0.5 - 1) / 2
        a = 0
        b = max_step_size
        
        if scheme == "threshold":
            while b - a > 1 / (curr_iter + 1):
                c = b - (b - a) * invphi
                self.apply_gradient(self.compute_gradient(self.client_id), c)
                self.soft_threshold(c)
                fc = self.objective_function()
                self.reset_beta_temp()
                d = a + (b - a) * invphi
                self.apply_gradient(self.compute_gradient(self.client_id), d)
                self.soft_threshold(d)
                fd = self.objective_function()
                self.reset_beta_temp()
                if fc < fd:
                    b = d
                else:
                    a = c        
            self.apply_gradient(self.compute_gradient(self.client_id), (a + b) / 2)
            self.soft_threshold((a + b) / 2)
            self.update_beta_curr()
            self.reset_beta_temp()
        elif scheme == "gradients_threshold":
            aggregated_gradient = self.aggregate_gradient()
            while b - a > 1 / (curr_iter + 1):
                c = b - (b - a) * invphi
                self.apply_gradient(aggregated_gradient, c)
                self.soft_threshold(c)
                fc = self.objective_function()
                self.reset_beta_temp()
                d = a + (b - a) * invphi
                self.apply_gradient(aggregated_gradient, d)
                self.soft_threshold(d)
                fd = self.objective_function()
                self.reset_beta_temp()
                if fc < fd:
                    b = d
                else:
                    a = c        
            self.apply_gradient(aggregated_gradient, (a + b) / 2)
            self.soft_threshold((a + b) / 2)
            self.update_beta_curr()
            self.reset_beta_temp()
        elif scheme == "models_threshold":
            while b - a > 1 / (curr_iter + 1):
                c = b - (b - a) * invphi
                self.aggregate_model(c)
                self.apply_gradient(self.compute_gradient(self.client_id), c)
                self.soft_threshold(c)
                fc = self.objective_function()
                self.reset_beta_temp()
                d = a + (b - a) * invphi
                self.aggregate_model(d)
                self.apply_gradient(self.compute_gradient(self.client_id), d)
                self.soft_threshold(d)
                fd = self.objective_function()
                self.reset_beta_temp()
                if fc < fd:
                    b = d
                else:
                    a = c       
            self.aggregate_model((a + b) / 2)
            self.apply_gradient(self.compute_gradient(self.client_id), (a + b) / 2)
            self.soft_threshold((a + b) / 2)
            self.update_beta_curr()
            self.reset_beta_temp()
        elif scheme == "models_gradients_threshold": # why this might not work: gradient is not computed at curr location
            aggregated_gradient = self.aggregate_gradient()
            while b - a > 1 / (curr_iter + 1):
                c = b - (b - a) * invphi
                self.aggregate_model(c)
                self.apply_gradient(aggregated_gradient, c)
                self.soft_threshold(c)
                fc = self.objective_function()
                self.reset_beta_temp()
                d = a + (b - a) * invphi
                self.aggregate_model(d)
                self.apply_gradient(aggregated_gradient, d)
                self.soft_threshold(d)
                fd = self.objective_function()
                self.reset_beta_temp()
                if fc < fd:
                    b = d
                else:
                    a = c
            self.aggregate_model((a + b) / 2)
            self.apply_gradient(aggregated_gradient, (a + b) / 2)
            self.soft_threshold((a + b) / 2)
            self.update_beta_curr()
            self.reset_beta_temp()
        elif scheme == "gradients_models_threshold": # why this might not work: there is an intermediate models step between gradients and thresholding
            aggregated_gradient = self.aggregate_gradient()
            while b - a > 1 / (curr_iter + 1):
                c = b - (b - a) * invphi
                self.apply_gradient(aggregated_gradient, c)
                self.aggregate_model(c)
                self.soft_threshold(c)
                fc = self.objective_function()
                self.reset_beta_temp()
                d = a + (b - a) * invphi
                self.apply_gradient(aggregated_gradient, d)
                self.aggregate_model(d)
                self.soft_threshold(d)
                fd = self.objective_function()
                self.reset_beta_temp()
                if fc < fd:
                    b = d
                else:
                    a = c
            self.apply_gradient(aggregated_gradient, (a + b) / 2)
            self.aggregate_model((a + b) / 2)
            self.soft_threshold((a + b) / 2)
            self.update_beta_curr()
            self.reset_beta_temp()
        elif scheme == "gm_threshold":
            pass