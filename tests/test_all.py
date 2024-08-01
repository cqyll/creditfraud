import cProfile
import pstats
import unittest
import numpy as np
import os
import sys
from memory_profiler import profile

# Dynamically add the project directory to PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from creditfraud.models.dt.decisiontree import DecisionTree
from creditfraud.models.dt.gradient_boosting import GradientBoosting
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split

class TestGradientBoosting(unittest.TestCase):
    def setUp(self):
        self.X_binary, self.y_binary = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
        self.X_multi, self.y_multi = make_classification(n_samples=100, n_features=20, n_classes=3, n_informative=3, n_redundant=0, random_state=42)
        self.feature_types = ['continuous'] * 20

    def test_gradient_boosting_mse_binary(self):
        self._test_gradient_boosting(self.X_binary, self.y_binary, loss='mse', criterion='mse')

    def test_gradient_boosting_logistic_binary(self):
        self._test_gradient_boosting(self.X_binary, self.y_binary, loss='logistic', criterion='gini')

    def test_gradient_boosting_mse_multi(self):
        self._test_gradient_boosting(self.X_multi, self.y_multi, loss='mse', criterion='mse')

    def test_gradient_boosting_logistic_multi(self):
        self._test_gradient_boosting(self.X_multi, self.y_multi, loss='logistic', criterion='gini')

    def _test_gradient_boosting(self, X, y, loss, criterion):
        gb = GradientBoosting(n_estimators=10, learning_rate=0.1, max_depth=3, min_samples_split=2, reg_lambda=1.0, loss=loss, criterion=criterion)
        gb.fit(X, y, feature_types=self.feature_types)
        predictions = gb.predict(X)
        self.assertEqual(len(predictions), len(y), "Predictions should have the same length as the input data")
        if len(np.unique(y)) > 2:
            self.assertTrue(np.all(np.isin(predictions, np.unique(y))), "Predictions should only contain valid class labels")
        else:
            self.assertTrue(np.all(np.isin(predictions, [0, 1])), "Predictions should only contain 0 or 1")

    def test_iris_dataset_gini(self):
        self._test_iris_dataset(criterion='gini')

    def test_iris_dataset_entropy(self):
        self._test_iris_dataset(criterion='entropy')

    def test_iris_dataset_mse(self):
        self._test_iris_dataset(criterion='mse')

    def _test_iris_dataset(self, criterion):
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
        gb = GradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, reg_lambda=1.0, criterion=criterion)
        
        feature_types = ['continuous'] * X_train.shape[1]
        gb.fit(X_train, y_train, feature_types=feature_types)
        
        predictions = gb.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        print(f"Accuracy for criterion {criterion}: {accuracy}")
        self.assertGreater(accuracy, 0.8, f"Accuracy should be greater than 0.8 for criterion {criterion}")


def run_tests_with_profiling():
    profiler = cProfile.Profile()
    profiler.enable()
    
    unittest.main(exit=False)
    
    profiler.disable()
    
    return profiler
    

if __name__ == '__main__':
    profiler = run_tests_with_profiling()
    
    with open('profiler_output_optMem.txt', 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs()
        stats.sort_stats('cumtime').print_stats(50)
        stats.sort_stats('tottime').print_stats(50)
        stats.sort_stats('ncalls').print_stats(50)
