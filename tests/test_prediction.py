from fraud_detection.alerts import AlertThresholds
from fraud_detection.config import FEATURE_COLUMNS
from fraud_detection.prediction import predict_records
from fraud_detection.training import build_models, fit_model


def test_prediction_on_new_records(sample_frame):
    X = sample_frame[FEATURE_COLUMNS]
    y = sample_frame["Class"]
    model = fit_model(build_models(rf_estimators=5, n_jobs=1)["logistic_regression"], X, y)
    artifact = {
        "model": model,
        "model_name": "logistic_regression",
        "thresholds": AlertThresholds(medium=0.2, high=0.7, review_capacity=0.005).to_dict(),
    }
    result = predict_records(artifact, X.iloc[:3])
    assert result.shape == (3, 3)
    assert result["puntaje_riesgo"].between(0, 1).all()
    assert set(result["nivel_riesgo"]) <= {"bajo", "medio", "alto"}
