import numpy as np
import heapq
from creditfraud.models.dt.criterion import GiniCriterion, EntropyCriterion, MeanSquaredErrorCriterion

class Splitter:
    def __init__(self, criterion, reg_lambda=1.0, gamma=0.0):
        if criterion == 'gini':
            self.criterion = GiniCriterion()
        elif criterion == 'entropy':
            self.criterion = EntropyCriterion()
        elif criterion == 'mse':
            self.criterion = MeanSquaredErrorCriterion()
        else:
            raise ValueError("Unknown criterion")
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        

    def best_split(self, X, y, grad, hess, feature_types):
        heap = []
        best_split = None

        # collect potential splits in a heap
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold

                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue

                gain = self._calculate_split_gain(grad[left_indices], hess[left_indices], grad[right_indices], hess[right_indices])
                left_purity = self.criterion(y[left_indices])
                right_purity = self.criterion(y[right_indices])

                # push the current split candidate onto the heap
                heapq.heappush(heap, (-gain, -left_purity - right_purity, feature_index, threshold, left_indices, right_indices))

        # extract the best split from the heap only once
        if heap:
            gain, purity, feature_index, threshold, left_indices, right_indices = heapq.heappop(heap)
            best_split = {
                'feature_index': feature_index,
                'threshold': threshold,
                'left_indices': left_indices,
                'right_indices': right_indices,
                'left_purity': -purity / 2,   # dividing by 2 as purity was stored as the sum of both purities negated
                'right_purity': -purity / 2   
            }

        return best_split
    
    def _calculate_split_gain(self, grad_left, hess_left, grad_right, hess_right):
        grad_total = np.sum(grad_left) + np.sum(grad_right)
        hess_total = np.sum(hess_left) + np.sum(hess_right)
        
        gain_left = (np.sum(grad_left) ** 2) / (np.sum(hess_left) + self.reg_lambda)
        gain_right = (np.sum(grad_right) ** 2) / (np.sum(hess_right) + self.reg_lambda)
        gain_total = (grad_total ** 2) / (hess_total + self.reg_lambda)
        
        gain = 0.5 * (gain_left + gain_right - gain_total) - self.gamma
        return gain
