import numpy as np

from fraud_detection.evaluation import classification_metrics, select_f1_threshold


def test_metrics_and_threshold_are_calculated():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.05, 0.4, 0.6, 0.9])
    threshold, best_f1 = select_f1_threshold(y, scores)
    metrics = classification_metrics(y, scores, threshold)
    assert threshold == 0.6
    assert best_f1 == 1.0
    assert metrics["f1"] == 1.0
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (2, 0, 0, 2)
