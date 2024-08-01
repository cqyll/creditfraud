import numpy as np

class GiniCriterion:
    def __call__(self, y):
        m = len(y)
        if m == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / m
        return 1.0 - np.sum(probabilities ** 2)
    
    def calculate_gain(self, y_left, y_right):
        total_size = len(y_left) + len(y_right)
        gini_left = self(y_left)
        gini_right = self(y_right)
        weighted_gini = (len(y_left) / total_size) * gini_left + (len(y_right) / total_size) * gini_right
        return -weighted_gini
    

class EntropyCriterion:
    def __call__(self, y):
        m = len(y)
        if m == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / m
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))
    
    def calculate_gain(self, y_left, y_right):
        total_size = len(y_left) + len(y_right)
        entropy_left = self(y_left)
        entropy_right = self(y_right)
        weighted_entropy = (len(y_left) / total_size) * entropy_left + (len(y_right) / total_size) * entropy_right
        return -weighted_entropy
    
class MeanSquaredErrorCriterion:
    def __call__(self, y):
        if len(y) == 0:
            return 0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def calculate_gain(self, y_left, y_right):
        total_size = len(y_left) + len(y_right)
        mse_left = self(y_left)
        mse_right = self(y_right)
        weighted_mse = (len(y_left) / total_size) * mse_left + (len(y_right) / total_size) * mse_right
        return -weighted_mse  
