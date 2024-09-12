from sklearn.datasets import make_classification
from frauddetection.models.dt.xghist_classifier import XGHistClassifier

# Generate a small dataset
X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_redundant=1, random_state=42)

# Create and fit the classifier
xghist = XGHistClassifier(n_estimators=5, learning_rate=0.1, max_depth=3, min_child_weight=1e-3, random_state=42)
xghist.fit(X, y)