import logging
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from scipy.sparse import csc_matrix, issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import cProfile
import pstats
from memory_profiler import memory_usage


class XGHistClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_child_weight=1, subsample=1.0, colsample_bytree=1.0, 
                 reg_lambda=1.0, reg_alpha=0.0, gamma=0, random_state=None,
                 n_bins=256, categorical_features = None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.random_state = check_random_state(random_state)
        self.n_bins = n_bins
        self.categorical_features = categorical_features

    def _to_csc(self, X):
        if issparse(X):
            return X.tocsc()
        else:
            return csc_matrix(X)

    def _preprocess_data(self, X):
        X_csc = self._to_csc(X)
        blocks = []
        for i in range(X_csc.shape[1]):
            col = X_csc.getcol(i)
            sorted_indices = np.argsort(col.data)
            sorted_col = csc_matrix((col.data[sorted_indices], 
                                     (col.indices[sorted_indices], np.zeros_like(col.indices))),
                                    shape=(X_csc.shape[0], 1))
            blocks.append(sorted_col)
        return blocks

    def _create_histograms(self, X_block, grad, hess):
        histograms = []
        for col in X_block:
            if issparse(col):
                col_data = col.data
            else:
                col_data = col.ravel()
            
            hist, bin_edges = np.histogram(col_data, bins=self.n_bins)
            grad_hist = np.histogram(col_data, bins=bin_edges, weights=grad)[0]
            hess_hist = np.histogram(col_data, bins=bin_edges, weights=hess)[0]
            histograms.append((hist, grad_hist, hess_hist, bin_edges))
        return histograms

    def _find_best_split(self, histograms):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for feature, (hist, grad_hist, hess_hist, bin_edges) in enumerate(histograms):
            G_left, H_left = 0, 0
            G_right, H_right = np.sum(grad_hist), np.sum(hess_hist)

            for i in range(1, len(hist)):
                G_left += grad_hist[i-1]
                H_left += hess_hist[i-1]
                G_right -= grad_hist[i-1]
                H_right -= hess_hist[i-1]

                if H_left < self.min_child_weight or H_right < self.min_child_weight:
                    continue

                gain = self._calculate_gain(G_left, H_left, G_right, H_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = bin_edges[i]

        return best_feature, best_threshold, best_gain

    def _calculate_gain(self, G_left, H_left, G_right, H_right):
        gain = (G_left**2 / (H_left + self.reg_lambda) + 
                G_right**2 / (H_right + self.reg_lambda) - 
                (G_left + G_right)**2 / (H_left + H_right + self.reg_lambda)) / 2 - self.gamma
        return gain

    def _build_tree(self, X_blocks, grad, hess, depth=0):
        if depth >= self.max_depth:
            return self._calculate_leaf_weight(grad, hess)

        histograms = self._create_histograms(X_blocks, grad, hess)
        best_feature, best_threshold, best_gain = self._find_best_split(histograms)

        if best_gain <= 0:
            return self._calculate_leaf_weight(grad, hess)

        left_mask = X_blocks[best_feature].toarray().ravel() <= best_threshold
        right_mask = ~left_mask

        left_blocks = [block[left_mask] for block in X_blocks]
        right_blocks = [block[right_mask] for block in X_blocks]

        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(left_blocks, grad[left_mask], hess[left_mask], depth+1),
            'right': self._build_tree(right_blocks, grad[right_mask], hess[right_mask], depth+1)
        }

    def _calculate_leaf_weight(self, grad, hess):
        return -np.sum(grad) / (np.sum(hess) + self.reg_lambda)

    def _predict_tree(self, tree, x):
        if isinstance(tree, dict):
            if x[tree['feature']] <= tree['threshold']:
                return self._predict_tree(tree['left'], x)
            else:
                return self._predict_tree(tree['right'], x)
        else:
            return tree

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        self.classes_ = np.unique(y)
        y = (y == self.classes_[1]).astype(int)

        self.X_blocks = self._preprocess_data(X)
        self.trees = []

        pred = np.zeros(len(y))

        for _ in range(self.n_estimators):
            grad, hess = self._calculate_gradient_hessian(y, pred)

            subsample_mask = self.random_state.rand(len(y)) < self.subsample
            X_subset = [block[subsample_mask] for block in self.X_blocks]
            grad_subset = grad[subsample_mask]
            hess_subset = hess[subsample_mask]

            tree = self._build_tree(X_subset, grad_subset, hess_subset)
            self.trees.append(tree)

            pred += self.learning_rate * np.apply_along_axis(lambda x: self._predict_tree(tree, x), 1, X)

        return self

    def _calculate_gradient_hessian(self, y, pred):
        prob = self._sigmoid(pred)
        grad = prob - y
        hess = prob * (1 - prob)
        return grad, hess

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict_proba(self, X):
        X = check_array(X, accept_sparse=True)
        if issparse(X):
            X = X.tocsr() 
        pred = np.zeros(X.shape[0])
        for tree in self.trees:
            pred += self.learning_rate * np.apply_along_axis(lambda x: self._predict_tree(tree, x), 1, X)
        prob = self._sigmoid(pred)
        return np.vstack([1-prob, prob]).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def run_xghist_test():
    random_state = 21
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=random_state)
    
    # Convert to DataFrame without any categorical variables
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    # Display the dtypes before any processing
    print("Data types before processing:")
    print(X_df.dtypes)
    
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled data back to DataFrame
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Display the dtypes after scaling
    print("\nData types after scaling:")
    print(X_train_scaled_df.dtypes)

    xghist = XGHistClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, 
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=0, n_bins=255, reg_alpha=0, random_state=random_state)
    
    xghist.fit(X_train_scaled_df, y_train)
    y_pred = xghist.predict(X_test_scaled_df)
    y_pred_proba = xghist.predict_proba(X_test_scaled_df)[:, 1]

    print("\nXGHistClassifier Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.10f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.10f}")

def run_hist_gb_test():
    random_state = 21
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=random_state)
    
    # Convert to DataFrame without any categorical variables
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    # Display the dtypes before any processing
    print("Data types before processing:")
    print(X_df.dtypes)
    
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled data back to DataFrame
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Display the dtypes after scaling
    print("\nData types after scaling:")
    print(X_train_scaled_df.dtypes)

    hist_gb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=3, random_state=random_state)
    hist_gb.fit(X_train_scaled_df, y_train)
    y_pred_hist = hist_gb.predict(X_test_scaled_df)
    y_pred_proba_hist = hist_gb.predict_proba(X_test_scaled_df)[:, 1]

    print("\nHistGradientBoostingClassifier Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_hist):.10f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_hist):.10f}")

if __name__ == "__main__":
    print("Running XGHistClassifier test...")
    run_xghist_test()

    print("\nRunning HistGradientBoostingClassifier test...")
    run_hist_gb_test()
    
    
    
    
# def profile_xghist():
#     random_state = 21
#     X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=random_state)
#     X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
#     X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=random_state)

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     xghist = XGHistClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, 
#                               subsample=0.8, colsample_bytree=0.8, reg_lambda=0, n_bins=255, reg_alpha=0, random_state=random_state)
    
#     xghist.fit(X_train_scaled, y_train)
#     y_pred = xghist.predict(X_test_scaled)
#     y_pred_proba = xghist.predict_proba(X_test_scaled)[:, 1]
    
#     accuracy = accuracy_score(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_pred_proba)
    
#     return accuracy, roc_auc

# def profile_hist_gb():
#     random_state = 21
#     X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=random_state)
#     X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
#     X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=random_state)

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     hist_gb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=3, random_state=random_state)
#     hist_gb.fit(X_train_scaled, y_train)
#     y_pred_hist = hist_gb.predict(X_test_scaled)
#     y_pred_proba_hist = hist_gb.predict_proba(X_test_scaled)[:, 1]
    
#     accuracy = accuracy_score(y_test, y_pred_hist)
#     roc_auc = roc_auc_score(y_test, y_pred_proba_hist)
    
#     return accuracy, roc_auc

# if __name__ == "__main__":
#     # Redirect output to a file
#     with open('profiling_output.txt', 'w') as f:
#         sys.stdout = f

#         print("Profiling XGHistClassifier...")
#         cProfile.run('profile_xghist()', 'xghist_prof')

#         print("Profiling HistGradientBoostingClassifier...")
#         cProfile.run('profile_hist_gb()', 'histgb_prof')

#         # Memory usage profiling
#         print("Memory usage profiling...")
#         mem_usage_xghist = memory_usage(profile_xghist, interval=0.1)
#         mem_usage_histgb = memory_usage(profile_hist_gb, interval=0.1)

#         print(f"XGHistClassifier - Max memory usage: {max(mem_usage_xghist)} MiB")
#         print(f"HistGradientBoostingClassifier - Max memory usage: {max(mem_usage_histgb)} MiB")

#         # Display profiling results
#         print("\nXGHistClassifier Profiling Results:")
#         p = pstats.Stats('xghist_prof')
#         p.strip_dirs().sort_stats('cumulative').print_stats(10)

#         print("\nHistGradientBoostingClassifier Profiling Results:")
#         p = pstats.Stats('histgb_prof')
#         p.strip_dirs().sort_stats('cumulative').print_stats(10)

#         # Getting accuracy and ROC AUC scores
#         xghist_accuracy, xghist_roc_auc = profile_xghist()
#         histgb_accuracy, histgb_roc_auc = profile_hist_gb()

#         print("\nXGHistClassifier Results:")
#         print(f"Accuracy: {xghist_accuracy:.10f}")
#         print(f"ROC AUC: {xghist_roc_auc:.10f}")

#         print("\nHistGradientBoostingClassifier Results:")
#         print(f"Accuracy: {histgb_accuracy:.10f}")
#         print(f"ROC AUC: {histgb_roc_auc:.10f}")

#     # Reset stdout back to default
#     sys.stdout = sys.__stdout__

#     print("Profiling complete. Results saved to profiling_output.txt.")