import numpy as np

class LossFunction:
    def gradient(self, y_true, y_pred):
        raise NotImplementedError

    def hessian(self, y_true, y_pred):
        raise NotImplementedError

class MeanSquaredError(LossFunction):
    def gradient(self, y_true, y_pred):
        return y_pred - y_true

    def hessian(self, y_true, y_pred):
        return np.ones_like(y_true)

class LogisticLoss(LossFunction):
    def gradient(self, y_true, y_pred):
        pred = 1 / (1 + np.exp(-y_pred))
        return pred - y_true

    def hessian(self, y_true, y_pred):
        pred = 1 / (1 + np.exp(-y_pred))
        return pred * (1 - pred)

class HuberLoss(LossFunction):
    def __init__(self, delta=1.0):
        self.delta = delta

    def gradient(self, y_true, y_pred):
        diff = y_pred - y_true
        return np.where(np.abs(diff) <= self.delta, diff, self.delta * np.sign(diff))

    def hessian(self, y_true, y_pred):
        diff = y_pred - y_true
        return np.where(np.abs(diff) <= self.delta, np.ones_like(y_true), np.zeros_like(y_true))

class AbsoluteError(LossFunction):
    def gradient(self, y_true, y_pred):
        return np.sign(y_pred - y_true)

    def hessian(self, y_true, y_pred):
        return np.zeros_like(y_true)

