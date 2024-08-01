import numpy as np
from creditfraud.models.dt.decisiontree import DecisionTree
from creditfraud.models.dt.loss_functions import LossFunction, MeanSquaredError, LogisticLoss, HuberLoss, AbsoluteError

class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, reg_lambda=1.0, gamma=0.0, loss='mse', criterion='mse'):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.gamma = gamma

        if isinstance(loss, str):
            if loss == 'mse':
                self.loss = MeanSquaredError()
            elif loss == 'logistic':
                self.loss = LogisticLoss()
            elif loss == 'huber':
                self.loss = HuberLoss()
            elif loss == 'absolute':
                self.loss = AbsoluteError()
            else:
                raise ValueError("Unknown loss function")
        elif isinstance(loss, LossFunction):
            self.loss = loss
        else:
            raise ValueError("Loss function must be a string or an instance of LossFunction")

        self.criterion = criterion
        self.trees = []
        self.classes = None
        self.is_multi_class = False

    def fit(self, X, y, feature_types):
        self.classes = np.unique(y)
        self.is_multi_class = len(self.classes) > 2
        self.trees = {cls: [] for cls in self.classes}
        m, n = X.shape
        y_pred = np.zeros((m, len(self.classes)))

        for i in range(self.n_estimators):
            for cls in self.classes:
                binary_y = (y == cls).astype(int)
                grad = self.loss.gradient(binary_y, y_pred[:, cls])
                hess = self.loss.hessian(binary_y, y_pred[:, cls])
                tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, reg_lambda=self.reg_lambda, gamma=self.gamma, criterion=self.criterion)
                tree.fit(X, binary_y, grad, hess, feature_types)
                y_pred[:, cls] += self.learning_rate * tree.predict(X)
                self.trees[cls].append(tree)

    def predict(self, X):
        m = X.shape[0]
        y_pred = np.zeros((m, len(self.classes)))
        
        for cls in self.classes:
            for tree in self.trees[cls]:
                y_pred[:, cls] += self.learning_rate * tree.predict(X)
        
        if self.is_multi_class:
            return np.argmax(y_pred, axis=1)
        else:
            return (y_pred[:, 1] > 0.5).astype(int)  # For binary classification return 0 or 1
