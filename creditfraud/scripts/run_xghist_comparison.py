from frauddetection.models.dt.utils import setup_logger
from frauddetection.models.dt.xghist_classifier import XGHistClassifier
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

logger = setup_logger(__name__)


def run_comparison():
    logger.info("Starting comparison between XGHistClassifier and HistGradientBoostingClassifier")

    # Generate sample data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=42
    )

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define scoring metrics
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1": make_scorer(f1_score),
        "roc_auc": make_scorer(roc_auc_score),
    }

    # XGHistClassifier
    xghist = XGHistClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    xghist_scores = cross_validate(xghist, X_scaled, y, cv=5, scoring=scoring)

    # HistGradientBoostingClassifier
    hist_gb = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.1, max_depth=3, random_state=42)
    hist_gb_scores = cross_validate(hist_gb, X_scaled, y, cv=5, scoring=scoring)

    # Log results
    for metric in scoring.keys():
        logger.info(
            f"XGHistClassifier {metric}: {np.mean(xghist_scores[f'test_{metric}']):.4f} (+/- {np.std(xghist_scores[f'test_{metric}']):.4f})"
        )
        logger.info(
            f"HistGradientBoostingClassifier {metric}: {np.mean(hist_gb_scores[f'test_{metric}']):.4f} (+/- {np.std(hist_gb_scores[f'test_{metric}']):.4f})"
        )


if __name__ == "__main__":
    run_comparison()
