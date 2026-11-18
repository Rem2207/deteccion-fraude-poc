import numpy as np

from fraud_detection.alerts import AlertThresholds, alert_summary, calibrate_alert_thresholds, classify_risk


def test_risk_levels_and_capacity():
    scores = np.linspace(0, 1, 1000)
    thresholds = calibrate_alert_thresholds(scores, high_threshold=0.995, review_capacity=0.01)
    middle_score = (thresholds.medium + thresholds.high) / 2
    levels = classify_risk([0.1, middle_score, 0.999], thresholds)
    assert levels.tolist() == ["bajo", "medio", "alto"]
    assert (scores >= thresholds.medium).mean() <= 0.011


def test_alert_summary_counts_errors():
    thresholds = AlertThresholds(medium=0.2, high=0.8, review_capacity=0.5)
    summary = alert_summary([0.9, 0.5, 0.1], [1, 0, 1], thresholds).set_index("nivel")
    assert summary.loc["alto", "fraudes"] == 1
    assert summary.loc["medio", "alertas_incorrectas"] == 1
    assert summary.loc["bajo", "fraudes"] == 1
