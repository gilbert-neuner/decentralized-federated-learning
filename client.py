import numpy as np
from numpy import linalg as LA
import random

class Client:
    """Represent a client."""
    
    def __init__(self, client_id, neighbors, X, Y):
        """
        Initialize a client.

        Parameters
        ----------
        client_id : int
            Id of client.
        neighbors : set
            List of neighbor id's.
        X : np.ndarray
            Matrix of shape (n, p).
        Y : np.ndarray
            Vector of shape (n, ).

        Returns
        -------
        None.

        """
        
        self.client_id = client_id # identifies this client
        self.neighbors = neighbors # client_id of neighbors
        self.X = X # training data
        self.Y = Y # training data
        self.n = np.shape(X)[0]
        self.p = np.shape(X)[1]
        self.beta_curr = np.zeros(self.p) # current beta
        self.betas_temp = {neighbor : np.copy(self.beta_curr) for neighbor in neighbors} # workspace
        self.gradients = {neighbor : np.zeros_like(self.beta_curr) for neighbor in neighbors} # workspace
        
    # HELPER FUNCTIONS
    
    def compute_gradient(self):
        self.gradients[self.client_id] = np.transpose(self.X) @ (self.X @ self.betas_temp[self.client_id] - self.Y) / self.n
    
    def normalize_magnitude(self, u, v):
        if LA.norm(u) == 0:
            return u
        return u * np.minimum(1, LA.norm(v) / LA.norm(u))
    
    # BASIC ACTIONS
    
    def AGGREGATE(self, step_size):
        aggregated_gradient = np.zeros_like(self.gradients[self.client_id])
        for j in self.neighbors:
            aggregated_gradient += self.normalize_magnitude(self.gradients[j], self.gradients[self.client_id])
        self.betas_temp[self.client_id] -= step_size * aggregated_gradient / len(self.neighbors)
        
    def CONSENSUS(self, step_size):
        aggregated_diff = np.zeros_like(self.betas_temp[self.client_id])
        for j in self.neighbors:
            aggregated_diff += self.normalize_magnitude(self.betas_temp[j] - self.betas_temp[self.client_id], self.gradients[self.client_id])
        self.betas_temp[self.client_id] += step_size * aggregated_diff / len(self.neighbors)
        
    def AGGREGATE_CONSENSUS(self, step_size):
        aggregated_suggestion = np.zeros_like(self.gradients[self.client_id])
        for j in self.neighbors:
            aggregated_suggestion += -1 * self.normalize_magnitude(self.gradients[j], self.gradients[self.client_id])
            aggregated_suggestion += self.normalize_magnitude(self.betas_temp[j] - self.betas_temp[self.client_id], self.gradients[self.client_id])
        self.betas_temp[self.client_id] += step_size * aggregated_suggestion / (2 * len(self.neighbors))
        
    def GRADIENT(self, step_size, everybody = False):
        if(everybody):
            for j in self.neighbors:
                self.betas_temp[j] -= step_size * self.gradients[j]
        else:
            self.betas_temp[self.client_id] -= step_size * self.gradients[self.client_id]
        
    def GRADIENT_CONSENSUS(self, step_size):
        aggregated_suggestion = np.zeros_like(self.betas_temp[self.client_id])
        for j in self.neighbors:
            aggregated_suggestion += self.normalize_magnitude(self.betas_temp[j] - self.betas_temp[self.client_id], self.gradients[self.client_id])
        aggregated_suggestion -= len(self.neighbors) * self.gradients[self.client_id]
        self.betas_temp[self.client_id] += aggregated_suggestion / (2 * len(self.neighbors))
        
    def THRESHOLD(self, step_size, threshold):
        self.betas_temp[self.client_id] = np.sign(self.betas_temp[self.client_id]) * np.maximum(np.abs(self.betas_temp[self.client_id]) - step_size * threshold, 0.0)
        
    # OPTIMIZATION
    
    def objective_function(self, threshold):
        return 1 / (2 * np.shape(self.X)[0]) * LA.norm(self.Y - self.X @ self.betas_temp[self.client_id]) ** 2 + threshold * LA.norm(self.betas_temp[self.client_id], ord = 1)
            
    def reset_beta_temp(self):
        self.betas_temp[self.client_id] = np.copy(self.beta_curr)
        
    def update_beta_curr(self):
        self.beta_curr = np.copy(self.betas_temp[self.client_id])
        
    def select_step_size(self, scheme, curr_iter, max_step_size, threshold):
        invphi = (5 ** 0.5 - 1) / 2
        a = 0
        b = max_step_size
        
        if scheme == "G":
            while b - a > 1 / (curr_iter + 1):
                c = b - (b - a) * invphi
                self.GRADIENT(c)
                self.THRESHOLD(c, threshold)
                fc = self.objective_function(threshold)
                self.reset_beta_temp()
                
                d = a + (b - a) * invphi
                self.GRADIENT(d)
                self.THRESHOLD(d, threshold)
                fd = self.objective_function(threshold)
                self.reset_beta_temp()
                
                if fc < fd:
                    b = d
                else:
                    a = c        
            self.GRADIENT((a + b) / 2)
            self.THRESHOLD((a + b) / 2, threshold)
            self.update_beta_curr()
            self.reset_beta_temp()
        elif scheme == "A":
            while b - a > 1 / (curr_iter + 1):
                c = b - (b - a) * invphi
                self.AGGREGATE(c)
                self.THRESHOLD(c, threshold)
                fc = self.objective_function(threshold)
                self.reset_beta_temp()
                
                d = a + (b - a) * invphi
                self.AGGREGATE(d)
                self.THRESHOLD(d, threshold)
                fd = self.objective_function(threshold)
                self.reset_beta_temp()
                
                if fc < fd:
                    b = d
                else:
                    a = c        
            self.AGGREGATE((a + b) / 2)
            self.THRESHOLD((a + b) / 2, threshold)
            self.update_beta_curr()
            self.reset_beta_temp()
        elif scheme == "(AC)":
            while b - a > 1 / (curr_iter + 1):
                c = b - (b - a) * invphi
                self.AGGREGATE_CONSENSUS(c)
                self.THRESHOLD(c, threshold)
                fc = self.objective_function(threshold)
                self.reset_beta_temp()
                
                d = a + (b - a) * invphi
                self.AGGREGATE_CONSENSUS(d)
                self.THRESHOLD(d, threshold)
                fd = self.objective_function(threshold)
                self.reset_beta_temp()
                
                if fc < fd:
                    b = d
                else:
                    a = c        
            self.AGGREGATE_CONSENSUS((a + b) / 2)
            self.THRESHOLD((a + b) / 2, threshold)
            self.update_beta_curr()
            self.reset_beta_temp()
        elif scheme == "CG":
            while b - a > 1 / (curr_iter + 1):
                c = b - (b - a) * invphi
                self.CONSENSUS(c)
                self.compute_gradient()
                self.GRADIENT(c)
                self.THRESHOLD(c, threshold)
                fc = self.objective_function(threshold)
                self.reset_beta_temp()
                
                d = a + (b - a) * invphi
                self.CONSENSUS(d)
                self.compute_gradient()
                self.GRADIENT(d)
                self.THRESHOLD(d, threshold)
                fd = self.objective_function(threshold)
                self.reset_beta_temp()
                
                if fc < fd:
                    b = d
                else:
                    a = c        
            self.CONSENSUS((a + b) / 2)
            self.compute_gradient()
            self.GRADIENT((a + b) / 2)
            self.THRESHOLD((a + b) / 2, threshold)
            self.update_beta_curr()
            self.reset_beta_temp()
        elif scheme == "GC":
            while b - a > 1 / (curr_iter + 1):
                c = b - (b - a) * invphi
                self.GRADIENT(c, True)
                self.CONSENSUS(c)
                self.THRESHOLD(c, threshold)
                fc = self.objective_function(threshold)
                self.reset_beta_temp()
                
                d = a + (b - a) * invphi
                self.GRADIENT(d, True)
                self.CONSENSUS(d)
                self.THRESHOLD(d, threshold)
                fd = self.objective_function(threshold)
                self.reset_beta_temp()
                
                if fc < fd:
                    b = d
                else:
                    a = c        
            self.GRADIENT((a + b) / 2, True)
            self.CONSENSUS((a + b) / 2)
            self.THRESHOLD((a + b) / 2, threshold)
            self.update_beta_curr()
            self.reset_beta_temp()
        elif scheme == "(GC)":
            while b - a > 1 / (curr_iter + 1):
                c = b - (b - a) * invphi
                self.GRADIENT_CONSENSUS(c)
                self.THRESHOLD(c, threshold)
                fc = self.objective_function(threshold)
                self.reset_beta_temp()
                
                d = a + (b - a) * invphi
                self.GRADIENT_CONSENSUS(d)
                self.THRESHOLD(d, threshold)
                fd = self.objective_function(threshold)
                self.reset_beta_temp()
                
                if fc < fd:
                    b = d
                else:
                    a = c        
            self.GRADIENT_CONSENSUS((a + b) / 2)
            self.THRESHOLD((a + b) / 2, threshold)
            self.update_beta_curr()
            self.reset_beta_temp()
            
class Adversary(Client):
    def __init__(self, client_id, neighbors, X, Y, corrupt_fraction):
        n = np.shape(X)[0]
        Y[random.sample(range(n), round(n * corrupt_fraction))] *= -1
        super().__init__(client_id, neighbors, X, Y)
        
    def select_step_size(self, scheme, curr_iter, max_step_size, threshold):
        invphi = (5 ** 0.5 - 1) / 2
        a = 0
        b = max_step_size

        while b - a > 1 / (curr_iter + 1):
            c = b - (b - a) * invphi
            self.GRADIENT(c)
            self.THRESHOLD(c, threshold)
            fc = self.objective_function(threshold)
            self.reset_beta_temp()
            
            d = a + (b - a) * invphi
            self.GRADIENT(d)
            self.THRESHOLD(d, threshold)
            fd = self.objective_function(threshold)
            self.reset_beta_temp()
            
            if fc < fd:
                b = d
            else:
                a = c        
        self.GRADIENT((a + b) / 2)
        self.THRESHOLD((a + b) / 2, threshold)
        self.update_beta_curr()
        self.reset_beta_temp()