import numpy as np
from scipy.sparse import csc_matrix, issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_X_y

from .tree1 import build_tree, predict_tree
from .utils import setup_logger




class XGHistClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_child_weight=1,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        gamma=0,
        random_state=None,
        n_bins=255,
    ):
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
        self.logger = setup_logger(self.__class__.__name__)

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

            tree = build_tree(
                X_subset,
                grad_subset,
                hess_subset,
                self.max_depth,
                self.min_child_weight,
                self.gamma,
                self.reg_lambda,
            )
            self.trees.append(tree)

            pred += self.learning_rate * np.apply_along_axis(lambda x: predict_tree(tree, x), 1, X)

        return self

    def predict_proba(self, X):
        X = check_array(X, accept_sparse=True)
        if issparse(X):
            X = X.tocsr()
        pred = np.zeros(X.shape[0])
        for tree in self.trees:
            pred += self.learning_rate * np.apply_along_axis(lambda x: predict_tree(tree, x), 1, X)
        prob = self._sigmoid(pred)
        return np.vstack([1 - prob, prob]).T

    def predict(self, X):
        self.logger.info("Predicting classes")
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def _preprocess_data(self, X):
        X_csc = self._to_csc(X)
        blocks = []
        for i in range(X_csc.shape[1]):
            col = X_csc.getcol(i)
            sorted_indices = np.argsort(col.data)
            sorted_col = csc_matrix(
                (
                    col.data[sorted_indices],
                    (col.indices[sorted_indices], np.zeros_like(col.indices)),
                ),
                shape=(X_csc.shape[0], 1),
            )
            blocks.append(sorted_col)
        return blocks

    def _to_csc(self, X):
        if issparse(X):
            return X.tocsc()
        else:
            return csc_matrix(X)

    def _calculate_gradient_hessian(self, y, pred):
        prob = self._sigmoid(pred)
        grad = prob - y
        hess = prob * (1 - prob)
        return grad, hess

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
