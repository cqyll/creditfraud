import numpy as np
from scipy.sparse import issparse


def create_histograms(X_block, grad, hess, n_bins):
    histograms = []
    for col in X_block:
        if issparse(col):
            col_data = col.data
        else:
            col_data = col.ravel()

        hist, bin_edges = np.histogram(col_data, bins=n_bins)
        grad_hist = np.histogram(col_data, bins=bin_edges, weights=grad)[0]
        hess_hist = np.histogram(col_data, bins=bin_edges, weights=hess)[0]
        histograms.append((hist, grad_hist, hess_hist, bin_edges))
    return histograms


def find_best_split(histograms, min_child_weight, gamma, reg_lambda):
    best_gain = -np.inf
    best_feature = None
    best_threshold = None

    for feature, (hist, grad_hist, hess_hist, bin_edges) in enumerate(histograms):
        G_left, H_left = 0, 0
        G_right, H_right = np.sum(grad_hist), np.sum(hess_hist)

        for i in range(1, len(hist)):
            G_left += grad_hist[i - 1]
            H_left += hess_hist[i - 1]
            G_right -= grad_hist[i - 1]
            H_right -= hess_hist[i - 1]

            if H_left < min_child_weight or H_right < min_child_weight:
                continue
            gain = calculate_gain(G_left, H_left, G_right, H_right, gamma, reg_lambda)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = bin_edges[i]
    return best_feature, best_threshold, best_gain


def calculate_gain(G_left, H_left, G_right, H_right, gamma, reg_lambda):
    gain = (
        G_left**2 / (H_left + reg_lambda)
        + G_right**2 / (H_right + reg_lambda)
        - (G_left + G_right) ** 2 / (H_left + H_right + reg_lambda)
    ) / 2 - gamma
    return gain
