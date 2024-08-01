from creditfraud.models.dt.splitter import Splitter
import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2, reg_lambda=1.0, gamma=0.0, criterion = 'gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.splitter = Splitter(criterion=criterion, reg_lambda=reg_lambda, gamma=gamma)
        self.tree = None
        self.classes_ = None

    def fit(self, X, y, grad, hess, feature_types):
        self.classes_ = np.unique(y)
        self.tree = self._build_tree(X, y, grad, hess, feature_types, depth=0)

    def _build_tree(self, X, y, grad, hess, feature_types, depth):
        num_samples, num_features = X.shape
        if depth >= self.max_depth or num_samples < self.min_samples_split:
            leaf_value = self._leaf_value(y)
            return leaf_value

        best_split = self.splitter.best_split(X, y, grad, hess, feature_types)
        if best_split is None:
            leaf_value = self._leaf_value(y)
            return leaf_value

        left_indices = best_split['left_indices']
        right_indices = best_split['right_indices']

        left_tree = self._build_tree(X[left_indices], y[left_indices], grad[left_indices], hess[left_indices], feature_types, depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], grad[right_indices], hess[right_indices], feature_types, depth + 1)
        
        return {
            'feature_index': best_split['feature_index'],
            'threshold': best_split['threshold'],
            'left': left_tree,
            'right': right_tree
        }

    def _leaf_value(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        return np.array([self._predict(inputs, self.tree) for inputs in X])

    def _predict(self, inputs, tree):
        if isinstance(tree, dict):
            feature_index = tree['feature_index']
            threshold = tree['threshold']
            if inputs[feature_index] <= threshold:
                return self._predict(inputs, tree['left'])
            else:
                return self._predict(inputs, tree['right'])
        else:
            return tree
