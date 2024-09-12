import numpy as np

from .histogram import create_histograms, find_best_split


def build_tree(X_blocks, grad, hess, max_depth, min_child_weight, gamma, reg_lambda, depth=0):
    if depth >= max_depth:
        return calculate_leaf_weight(grad, hess, reg_lambda)

    histograms = create_histograms(X_blocks, grad, hess, n_bins=255)
    best_feature, best_threshold, best_gain = find_best_split(
        histograms, min_child_weight, gamma, reg_lambda
    )

    if best_gain <= 0:
        return calculate_leaf_weight(grad, hess, reg_lambda)

    left_mask = X_blocks[best_feature].toarray().ravel() <= best_threshold
    right_mask = ~left_mask

    left_blocks = [block[left_mask] for block in X_blocks]
    right_blocks = [block[right_mask] for block in X_blocks]

    return {
        "feature": best_feature,
        "threshold": best_threshold,
        "left": build_tree(
            left_blocks,
            grad[left_mask],
            hess[left_mask],
            max_depth,
            min_child_weight,
            gamma,
            reg_lambda,
            depth + 1,
        ),
        "right": build_tree(
            right_blocks,
            grad[right_mask],
            hess[right_mask],
            max_depth,
            min_child_weight,
            gamma,
            reg_lambda,
            depth + 1,
        ),
    }


def predict_tree(tree, x):
    if isinstance(tree, dict):
        if x[tree["feature"]] <= tree["threshold"]:
            return predict_tree(tree["left"], x)
        else:
            return predict_tree(tree["right"], x)
    else:
        return tree


def calculate_leaf_weight(grad, hess, reg_lambda):
    return -np.sum(grad) / (np.sum(hess) + reg_lambda)
