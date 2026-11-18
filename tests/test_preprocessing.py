import numpy as np

from fraud_detection.preprocessing import build_preprocessor


def test_scaler_is_fitted_only_with_training_rows(sample_frame):
    train = sample_frame.iloc[:120].drop(columns="Class")
    validation = sample_frame.iloc[120:].drop(columns="Class")
    preprocessor = build_preprocessor().fit(train)
    amount_scaler = preprocessor.named_transformers_["amount"].named_steps["scale"]
    expected_train_mean = np.log1p(train["Amount"]).mean()
    combined_mean = np.log1p(sample_frame["Amount"]).mean()
    assert np.isclose(amount_scaler.mean_[0], expected_train_mean)
    assert not np.isclose(amount_scaler.mean_[0], combined_mean)
    assert preprocessor.transform(validation).shape == (80, 30)
