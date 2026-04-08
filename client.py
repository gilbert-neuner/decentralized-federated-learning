import numpy as np
from numpy import linalg as LA
import random

class Client:
    # INITIALIZER
    
    # data_params: X, Y
    # topology_params: client_id, neighbors
    # algorithm_params: scheme, max_step_size, n_iter, threshold, beta0, random_displace, winsorize
    # trust_params: info, accelerate, cosine_recompute, include
    # adversary_params: adversary_type, corrupt_fraction, jitter
    def __init__(self, data_params, topology_params, algorithm_params = {}, trust_params = {}):
        # data_params
        self.X = data_params["X"]
        self.Y = data_params["Y"]
        self.n = np.shape(self.X)[0]
        self.p = np.shape(self.X)[1]
        # topology_params
        self.client_id = topology_params["client_id"]
        self.neighbors = topology_params["neighbors"]
        # algorithm_params
        self.scheme = algorithm_params.get("scheme", "G")
        self.max_step_size = algorithm_params.get("max_step_size", 1)
        self.n_iter = algorithm_params.get("n_iter", 100)
        self.threshold = algorithm_params.get("threshold", 0.1)
        self.winsorize = algorithm_params.get("winsorize", 0)
        if algorithm_params.get("beta0") is None:
            self.beta_curr = np.zeros(self.p)
        else:
            self.beta_curr = algorithm_params.get("beta0")
        self.random_displace = algorithm_params.get("random_displace", 0)
        if self.random_displace!= 0:
            displacement = np.random.uniform(-1, 1, self.p)
            displacement *= self.random_displace / LA.norm(displacement)
            self.beta_curr += displacement
        # trust_params
        self.info = trust_params.get("info", "Both")
        self.accelerate = trust_params.get("accelerate", True)
        self.cosine_recompute = trust_params.get("cosine_recompute", False)
        self.include = algorithm_params.get("include", 0)
        # adversary_params
        self.adversary_type = "friendly"
        self.corrupt_fraction = 0
        self.jitter = 0
        # workspace
        self.betas_temp = {j: np.copy(self.beta_curr) for j in self.neighbors}
        self.gradients = {j: np.zeros_like(self.beta_curr) for j in self.neighbors}
        self.betas_old = {j: np.copy(self.beta_curr) for j in self.neighbors}
        self.curr_iter = 0
        # trust
        self.a_trust = {j: 1 for j in self.neighbors}
        self.b_trust = {j: 1 for j in self.neighbors}
        self.trust = {j: 1 for j in self.neighbors}
        # history
        self.cosine_history = {j: [] for j in self.neighbors}
        self.distance_history = {j: [] for j in self.neighbors}
    
    # MATH
    
    def compute_gradient(self):
        self.gradients[self.client_id] = np.transpose(self.X) @ (self.X @ self.betas_temp[self.client_id] - self.Y) / self.n
        
    def normalize_magnitude(self, old_vec, new_norm):
        if LA.norm(old_vec) == 0:
            return old_vec
        return old_vec * np.minimum(1, LA.norm(new_norm) / LA.norm(old_vec))
    
    def objective_function(self):
        return 1 / (2 * np.shape(self.X)[0]) * LA.norm(self.Y - self.X @ self.betas_temp[self.client_id]) ** 2 + self.threshold * LA.norm(self.betas_temp[self.client_id], ord = 1)
    
    # TRUST
    
    def cosine_similarity(self):
        if self.cosine_recompute:
            out = {}
            for j in self.neighbors:
                gradient_ij = np.transpose(self.X) @ (self.X @ self.betas_temp[j] - self.Y) / self.n
                out[j] = 0.5 * (np.dot(gradient_ij, self.gradients[j]) / (LA.norm(gradient_ij) * LA.norm(self.gradients[j])) + 1)
            return out
        else:
            return {j: 0.5 * (np.dot(self.gradients[self.client_id], self.gradients[j]) / (LA.norm(self.gradients[self.client_id]) * LA.norm(self.gradients[j])) + 1) for j in self.neighbors}
    
    def distance_similarity(self):
        return {j: LA.norm(self.betas_temp[self.client_id] - self.betas_temp[j]) / LA.norm(self.betas_old[self.client_id] - self.betas_old[j]) for j in self.neighbors}
    
    def rank_dict(self, d, desc):
        # Sort the items by value
        sorted_items = sorted(d.items(), key=lambda x: x[1], reverse = desc)
        # Assign ranks
        ranks = {k: (rank + 1) / len(d) for rank, (k, v) in enumerate(sorted_items)}
        return ranks
    
    # TODO: can speed this up by ignoring norms
    def judge_cosines(self):
        cosines = self.cosine_similarity()
        cosine_ranks = self.rank_dict(cosines, desc = True)
        if self.include < 0:
            return {j: cosines[j] > 0.5 and cosine_ranks[j] < -1 * self.include for j in list(set(self.neighbors) - set([self.client_id]))}
        else:
            return {j: cosines[j] > 0.5 or cosine_ranks[j] < self.include for j in list(set(self.neighbors) - set([self.client_id]))}
        
    def judge_distances(self):
        distances = self.distance_similarity()
        distance_ranks = self.rank_dict(distances, desc = False)
        if self.include < 0:
            return {j: distances[j] < 1 and distance_ranks[j] < -1 * self.include for j in list(set(self.neighbors) - set([self.client_id]))}
        else:
            return {j: distances[j] < 1 or distance_ranks[j] < self.include for j in list(set(self.neighbors) - set([self.client_id]))}
     
    def update_gradient_based_trust(self):
        cosine_judgement = self.judge_cosines()
        if self.accelerate:
            for j in list(set(self.neighbors) - set([self.client_id])):
                if len(self.cosine_history[j]) == 0:
                    if cosine_judgement[j]:
                        self.a_trust[j] += 1
                        self.cosine_history[j].append(1)
                    else:
                        self.b_trust[j] += 1
                        self.cosine_history[j].append(-1)
                else:
                    if cosine_judgement[j] and self.cosine_history[j][-1] > 0:
                        self.a_trust[j] += self.cosine_history[j][-1] + 1
                        self.cosine_history[j].append(self.cosine_history[j][-1] + 1)
                    elif cosine_judgement[j] and self.cosine_history[j][-1] <= 0:
                        self.a_trust[j] += 1
                        self.cosine_history[j].append(1)
                    elif not cosine_judgement[j] and self.cosine_history[j][-1] > 0:
                        self.b_trust[j] += 1
                        self.cosine_history[j].append(-1)
                    elif not cosine_judgement[j] and self.cosine_history[j][-1] <= 0:
                        self.b_trust[j] += -1 * self.cosine_history[j][-1] + 1
                        self.cosine_history[j].append(self.cosine_history[j][-1] - 1)
        else:
            for j in list(set(self.neighbors) - set([self.client_id])):
                if cosine_judgement[j]:
                    self.a_trust[j] += 1
                    self.cosine_history[j].append(1)
                else:
                    self.b_trust[j] += 1
                    self.cosine_history[j].append(-1)
        
    def update_model_based_trust(self):
        distance_judgement = self.judge_distances()
        if self.accelerate:
            for j in list(set(self.neighbors) - set([self.client_id])):
                if len(self.distance_history[j]) == 0:
                    if distance_judgement[j]:
                        self.b_trust[j] += 1
                        self.distance_history[j].append(-1)
                    else:
                        self.a_trust[j] += 1
                        self.distance_history[j].append(1)
                else:
                    if not distance_judgement[j] and self.distance_history[j][-1] > 0:
                        self.b_trust[j] += 1
                        self.distance_history[j].append(-1)
                    elif not distance_judgement[j] and self.distance_history[j][-1] <= 0:
                        self.b_trust[j] += -1 * self.distance_history[j][-1] + 1
                        self.distance_history[j].append(self.distance_history[j][-1] - 1)
                    elif distance_judgement[j] and self.distance_history[j][-1] > 0:
                        self.a_trust[j] += self.distance_history[j][-1] + 1
                        self.distance_history[j].append(self.distance_history[j][-1] + 1)
                    elif distance_judgement[j] and self.distance_history[j][-1] <= 0:
                        self.a_trust[j] += 1
                        self.distance_history[j].append(1)
        else:
            for j in list(set(self.neighbors) - set([self.client_id])):
                if distance_judgement[j]:
                    self.b_trust[j] += 1
                    self.distance_history[j].append(-1)
                else:
                    self.a_trust[j] += 1
                    self.distance_history[j].append(1)
                        
    def update_trust(self):
        if self.info == "Gradient":
            self.update_gradient_based_trust()
        elif self.info == "Model":
            self.update_model_based_trust()
        elif self.info == "Both":
            self.update_gradient_based_trust()
            self.update_model_based_trust()
        for j in list(set(self.neighbors) - set([self.client_id])):
            self.trust[j] = self.a_trust[j] / (self.a_trust[j] + self.b_trust[j])
            
    # ALGORITHM STEPS
    
    # TODO: should I be dividing by number of neighbors or sum of trust?
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
        
    def THRESHOLD(self, step_size):
        self.betas_temp[self.client_id] = np.sign(self.betas_temp[self.client_id]) * np.maximum(np.abs(self.betas_temp[self.client_id]) - step_size * self.threshold, 0.0)
        
    def WINSORIZE(self):
        for i in range(self.p):
            grad_distances = {j: abs(self.gradients[self.client_id][i] - self.gradients[j][i]) for j in self.neighbors}
            grad_ranks = self.rank_dict(grad_distances, desc = True)
            good_grad_coords = {j: self.gradients[j][i] for j in self.neighbors if grad_ranks[j] > self.winsorize}
            max_good_grad_coord = max(good_grad_coords.values())
            min_good_grad_coord = min(good_grad_coords.values())
            for j in list(set(self.neighbors) - set([self.client_id])):
                if self.gradients[j][i] > max_good_grad_coord:
                    self.gradients[j][i] = max_good_grad_coord
                elif self.gradients[j][i] < min_good_grad_coord:
                    self.gradients[j][i] = min_good_grad_coord
               
    # MISC
    
    def reset_beta_temp(self):
        self.betas_temp[self.client_id] = np.copy(self.beta_curr)
        
    def update_beta_curr(self):
        self.beta_curr = np.copy(self.betas_temp[self.client_id])
        
    def update_betas_old(self):
        self.betas_old = {j : np.copy(self.betas_temp[j]) for j in self.neighbors}
        
    def select_step_size(self):
        invphi = (5 ** 0.5 - 1) / 2
        a = 0
        b = self.max_step_size
        
        if self.scheme == "G":
            while b - a > 1 / (self.curr_iter + 1):
                c = b - (b - a) * invphi
                self.GRADIENT(c)
                self.THRESHOLD(c)
                fc = self.objective_function()
                self.reset_beta_temp()
                
                d = a + (b - a) * invphi
                self.GRADIENT(d)
                self.THRESHOLD(d)
                fd = self.objective_function()
                self.reset_beta_temp()
                
                if fc < fd:
                    b = d
                else:
                    a = c        
            self.GRADIENT((a + b) / 2)
            self.THRESHOLD((a + b) / 2)
            self.update_beta_curr()
            self.reset_beta_temp()
        elif self.scheme == "A":
            while b - a > 1 / (self.curr_iter + 1):
                c = b - (b - a) * invphi
                self.AGGREGATE(c)
                self.THRESHOLD(c)
                fc = self.objective_function()
                self.reset_beta_temp()
                
                d = a + (b - a) * invphi
                self.AGGREGATE(d)
                self.THRESHOLD(d)
                fd = self.objective_function()
                self.reset_beta_temp()
                
                if fc < fd:
                    b = d
                else:
                    a = c   
            self.AGGREGATE((a + b) / 2)
            self.THRESHOLD((a + b) / 2)
            self.update_beta_curr()
            self.reset_beta_temp()
        elif self.scheme == "(AC)":
            while b - a > 1 / (self.curr_iter + 1):
                c = b - (b - a) * invphi
                self.AGGREGATE_CONSENSUS(c)
                self.THRESHOLD(c)
                fc = self.objective_function()
                self.reset_beta_temp()
                
                d = a + (b - a) * invphi
                self.AGGREGATE_CONSENSUS(d)
                self.THRESHOLD(d)
                fd = self.objective_function()
                self.reset_beta_temp()
                
                if fc < fd:
                    b = d
                else:
                    a = c        
            self.AGGREGATE_CONSENSUS((a + b) / 2)
            self.THRESHOLD((a + b) / 2)
            self.update_beta_curr()
            self.reset_beta_temp()
        elif self.scheme == "CG":
            while b - a > 1 / (self.curr_iter + 1):
                c = b - (b - a) * invphi
                self.CONSENSUS(c)
                self.compute_gradient()
                self.GRADIENT(c)
                self.THRESHOLD(c)
                fc = self.objective_function()
                self.reset_beta_temp()
                
                d = a + (b - a) * invphi
                self.CONSENSUS(d)
                self.compute_gradient()
                self.GRADIENT(d)
                self.THRESHOLD(d)
                fd = self.objective_function()
                self.reset_beta_temp()
                
                if fc < fd:
                    b = d
                else:
                    a = c        
            self.CONSENSUS((a + b) / 2)
            self.compute_gradient()
            self.GRADIENT((a + b) / 2)
            self.THRESHOLD((a + b) / 2)
            self.update_beta_curr()
            self.reset_beta_temp()
        elif self.scheme == "GC":
            while b - a > 1 / (self.curr_iter + 1):
                c = b - (b - a) * invphi
                self.GRADIENT(c, True)
                self.CONSENSUS(c)
                self.THRESHOLD(c)
                fc = self.objective_function()
                self.reset_beta_temp()
                
                d = a + (b - a) * invphi
                self.GRADIENT(d, True)
                self.CONSENSUS(d)
                self.THRESHOLD(d)
                fd = self.objective_function()
                self.reset_beta_temp()
                
                if fc < fd:
                    b = d
                else:
                    a = c        
            self.GRADIENT((a + b) / 2, True)
            self.CONSENSUS((a + b) / 2)
            self.THRESHOLD((a + b) / 2)
            self.update_beta_curr()
            self.reset_beta_temp()
        elif self.scheme == "(GC)":
            while b - a > 1 / (self.curr_iter + 1):
                c = b - (b - a) * invphi
                self.GRADIENT_CONSENSUS(c)
                self.THRESHOLD(c)
                fc = self.objective_function()
                self.reset_beta_temp()
                
                d = a + (b - a) * invphi
                self.GRADIENT_CONSENSUS(d)
                self.THRESHOLD(d)
                fd = self.objective_function()
                self.reset_beta_temp()
                
                if fc < fd:
                    b = d
                else:
                    a = c        
            self.GRADIENT_CONSENSUS((a + b) / 2)
            self.THRESHOLD((a + b) / 2)
            self.update_beta_curr()
            self.reset_beta_temp()
        
        self.curr_iter += 1
        
class Adversary(Client):
    # INITIALIZER
    
    # data_params: X, Y
    # topology_params: client_id, neighbors
    # algorithm_params: scheme, max_step_size, n_iter, threshold, beta0, random_displace, 
    # trust_params: info, accelerate, include
    # adversary_params: adversary_type, corrupt_fraction, jitter
    def __init__(self, data_params, topology_params, algorithm_params = {}, trust_params = {}, adversary_params = {}): 
        # adversary_params
        adversary_type = adversary_params.get("adversary_type", "X")
        corrupt_fraction = adversary_params.get("corrupt_fraction", 1)
        jitter = adversary_params.get("jitter", 0)        
        n = np.shape(data_params["X"])[0]
        p = np.shape(data_params["X"])[1]
        X_jitter = np.random.multivariate_normal(np.zeros(p), jitter * np.identity(p), n)
        if adversary_type == "X":
            corrupt_index = random.sample(range(n * p), round(n * p * corrupt_fraction))
            data_params["X"].flat[corrupt_index] *= -1
            data_params["X"].flat[corrupt_index] += X_jitter.flat[corrupt_index]
        elif adversary_type == "X_cols":
            corrupt_index = random.sample(range(p), round(p * corrupt_fraction))
            data_params["X"][:, corrupt_index] *= -1
            data_params["X"][:, corrupt_index] += X_jitter[:, corrupt_index]
        elif adversary_type == "X_rows":
            corrupt_index = random.sample(range(n), round(n * corrupt_fraction))
            data_params["X"][corrupt_index, :] *= -1
            data_params["X"][corrupt_index, :] += X_jitter[corrupt_index, :]
        super().__init__(data_params, topology_params, algorithm_params, trust_params)
        # adversary_params
        self.adversary_type = adversary_type
        self.corrupt_fraction = corrupt_fraction
        self.jitter = jitter
        
    def select_step_size(self):
        invphi = (5 ** 0.5 - 1) / 2
        a = 0
        b = self.max_step_size

        while b - a > 1 / (self.curr_iter + 1):
            c = b - (b - a) * invphi
            self.GRADIENT(c)
            self.THRESHOLD(c)
            fc = self.objective_function()
            self.reset_beta_temp()
            
            d = a + (b - a) * invphi
            self.GRADIENT(d)
            self.THRESHOLD(d)
            fd = self.objective_function()
            self.reset_beta_temp()
            
            if fc < fd:
                b = d
            else:
                a = c        
        self.GRADIENT((a + b) / 2)
        self.THRESHOLD((a + b) / 2)
        self.update_beta_curr()
        self.reset_beta_temp()
        
        self.curr_iter += 1