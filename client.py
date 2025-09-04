import numpy as np
from numpy import linalg as LA
import random

class Client:
    # topology_params: client_id, neighbors
    # data_params: X, Y
    def __init__(self, topology_params, data_params):
        # parameters
        self.client_id = topology_params["client_id"] # identifies this client
        self.neighbors = topology_params["neighbors"] # client_id of neighbors
        self.X = data_params["X"] # training data
        self.Y = data_params["Y"] # training data
        self.n = np.shape(self.X)[0]
        self.p = np.shape(self.X)[1]
        # optimization
        self.beta_curr = np.zeros(self.p) # current beta
        self.betas_temp = {j: np.copy(self.beta_curr) for j in self.neighbors} # workspace
        self.gradients = {j: np.zeros_like(self.beta_curr) for j in self.neighbors} # workspace
        # trust
        self.a_trust = {j: 1 for j in self.neighbors}
        self.b_trust = {j: 1 for j in self.neighbors}
        self.trust = {j: 1 for j in self.neighbors}
        self.betas_old = {j: np.copy(self.beta_curr) for j in self.neighbors}
        # history
        self.gradient_history = {j: [] for j in self.neighbors}
        self.model_history = {j: [] for j in self.neighbors}
        
    # HELPER FUNCTIONS
    
    def compute_gradient(self):
        self.gradients[self.client_id] = np.transpose(self.X) @ (self.X @ self.betas_temp[self.client_id] - self.Y) / self.n
    
    def normalize_magnitude(self, u, v):
        if LA.norm(u) == 0:
            return u
        return u * np.minimum(1, LA.norm(v) / LA.norm(u))
    
    def cosine_similarity(self, j):
        gradient_ij = np.transpose(self.X) @ (self.X @ self.betas_temp[j] - self.Y) / self.n
        return 0.5 * (np.dot(gradient_ij, self.gradients[j]) / (LA.norm(gradient_ij) * LA.norm(self.gradients[j])) + 1)
    
    def model_similarity(self, j):
        return LA.norm(self.betas_temp[self.client_id] - self.betas_temp[j]) / LA.norm(self.betas_old[self.client_id] - self.betas_old[j])
    
    def rank_dict(self, d, desc):
        # Sort the items by value
        sorted_items = sorted(d.items(), key=lambda x: x[1], reverse = desc)
        # Assign ranks
        ranks = {k: rank / (len(d) - 1) for rank, (k, v) in enumerate(sorted_items)}
        return ranks
    
    def check_gradient(self, include):
        cosine_similarity_measures = {j: self.cosine_similarity(j) for j in list(set(self.neighbors) - set([self.client_id]))}
        ranked_cosines = self.rank_dict(cosine_similarity_measures, desc = True)
        if include < 0:
            return {j: cosine_similarity_measures[j] > 0.5 and ranked_cosines[j] < -1 * include for j in list(set(self.neighbors) - set([self.client_id]))}
        else:
            return {j: cosine_similarity_measures[j] > 0.5 or ranked_cosines[j] < include for j in list(set(self.neighbors) - set([self.client_id]))}
    
    def check_model(self, include):
        model_similarity_measures = {j: self.model_similarity(j) for j in list(set(self.neighbors) - set([self.client_id]))}
        ranked_models = self.rank_dict(model_similarity_measures, desc = False)
        if include < 0:
            return {j: model_similarity_measures[j] < 1 and ranked_models[j] < -1 * include for j in list(set(self.neighbors) - set([self.client_id]))}
        else:
            return {j: model_similarity_measures[j] < 1 or ranked_models[j] < include for j in list(set(self.neighbors) - set([self.client_id]))}
    
    def update_gradient_based_trust(self, accelerate, include):
        gradient_suspicion = self.check_gradient(include)
        if not accelerate:
            for j in list(set(self.neighbors) - set([self.client_id])):
                if gradient_suspicion[j]:
                    self.a_trust[j] += 1
                    self.gradient_history[j].append(1)
                else:
                    self.b_trust[j] += 1
                    self.gradient_history[j].append(-1)
        elif accelerate:
            for j in list(set(self.neighbors) - set([self.client_id])):
                if len(self.gradient_history[j]) == 0:
                    if gradient_suspicion[j]:
                        self.a_trust[j] += 1
                        self.gradient_history[j].append(1)
                    else:
                        self.b_trust[j] += 1
                        self.gradient_history[j].append(-1)
                else:
                    if gradient_suspicion[j] and self.gradient_history[j][-1] > 0:
                        self.a_trust[j] += self.gradient_history[j][-1] + 1
                        self.gradient_history[j].append(self.gradient_history[j][-1] + 1)
                    elif gradient_suspicion[j] and self.gradient_history[j][-1] <= 0:
                        self.a_trust[j] += 1
                        self.gradient_history[j].append(1)
                    elif not gradient_suspicion[j] and self.gradient_history[j][-1] > 0:
                        self.b_trust[j] += 1
                        self.gradient_history[j].append(-1)
                    elif not gradient_suspicion[j] and self.gradient_history[j][-1] <= 0:
                        self.b_trust[j] += -1 * self.gradient_history[j][-1] + 1
                        self.gradient_history[j].append(self.gradient_history[j][-1] - 1)
    
    def update_model_based_trust(self, accelerate, include):
        model_suspicion = self.check_model(include)
        if not accelerate:
            for j in list(set(self.neighbors) - set([self.client_id])):
                if model_suspicion[j]:
                    self.b_trust[j] += 1
                    self.model_history[j].append(-1)
                else:
                    self.a_trust[j] += 1
                    self.model_history[j].append(1)
        elif accelerate:
            for j in list(set(self.neighbors) - set([self.client_id])):
                if len(self.model_history[j]) == 0:
                    if model_suspicion[j]:
                        self.b_trust[j] += 1
                        self.model_history[j].append(-1)
                    else:
                        self.a_trust[j] += 1
                        self.model_history[j].append(1)
                else:
                    if not model_suspicion[j] and self.model_history[j][-1] > 0:
                        self.b_trust[j] += 1
                        self.model_history[j].append(-1)
                    elif not model_suspicion[j] and self.model_history[j][-1] <= 0:
                        self.b_trust[j] += -1 * self.model_history[j][-1] + 1
                        self.model_history[j].append(self.model_history[j][-1] - 1)
                    elif model_suspicion[j] and self.model_history[j][-1] > 0:
                        self.a_trust[j] += self.model_history[j][-1] + 1
                        self.model_history[j].append(self.model_history[j][-1] + 1)
                    elif model_suspicion[j] and self.model_history[j][-1] <= 0:
                        self.a_trust[j] += 1
                        self.model_history[j].append(1)
            
    
    def update_trust(self, trust_params):
        info = trust_params["info"]
        accelerate = trust_params["accelerate"]
        include = trust_params["include"]
        
        if info == "Gradient":
            self.update_gradient_based_trust(accelerate, include)
        elif info == "Model":
            self.update_model_based_trust(accelerate, include)
        elif info == "Both":
            self.update_gradient_based_trust(accelerate, include)
            self.update_model_based_trust(accelerate, include)
        for j in list(set(self.neighbors) - set([self.client_id])):
            self.trust[j] = self.a_trust[j] / (self.a_trust[j] + self.b_trust[j])
    
    # BASIC ACTIONS
    
    def AGGREGATE(self, step_size):
        aggregated_gradient = np.zeros_like(self.gradients[self.client_id])
        for j in self.neighbors:
            aggregated_gradient += self.trust[j] * self.normalize_magnitude(self.gradients[j], self.gradients[self.client_id])
        self.betas_temp[self.client_id] -= step_size * aggregated_gradient / len(self.neighbors)
        
    def CONSENSUS(self, step_size):
        aggregated_diff = np.zeros_like(self.betas_temp[self.client_id])
        for j in self.neighbors:
            aggregated_diff += self.trust[j] * self.normalize_magnitude(self.betas_temp[j] - self.betas_temp[self.client_id], self.gradients[self.client_id])
        self.betas_temp[self.client_id] += step_size * aggregated_diff / len(self.neighbors)
        
    def AGGREGATE_CONSENSUS(self, step_size):
        aggregated_suggestion = np.zeros_like(self.gradients[self.client_id])
        for j in self.neighbors:
            aggregated_suggestion -= self.trust[j] * self.normalize_magnitude(self.gradients[j], self.gradients[self.client_id])
            aggregated_suggestion += self.trust[j] * self.normalize_magnitude(self.betas_temp[j] - self.betas_temp[self.client_id], self.gradients[self.client_id])
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
            aggregated_suggestion += self.trust[j] * self.normalize_magnitude(self.betas_temp[j] - self.betas_temp[self.client_id], self.gradients[self.client_id])
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
        
    def update_betas_old(self):
        self.betas_old = {j : np.copy(self.betas_temp[j]) for j in self.neighbors}
        
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
    # topology_params: client_id, neighbors
    # data_params: X, Y
    # adversary_params: corrupt_fraction
    def __init__(self, topology_params, data_params, adversary_params):
        n = np.shape(data_params["X"])[0]
        corrupt_fraction = adversary_params["corrupt_fraction"]
        data_params["Y"][random.sample(range(n), round(n * corrupt_fraction))] *= -1
        super().__init__(topology_params, data_params)
        
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