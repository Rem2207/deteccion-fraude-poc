"""Construccion y ajuste de los tres modelos del experimento."""

from __future__ import annotations

import inspect
from collections.abc import Mapping

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from .config import RANDOM_STATE
from .preprocessing import build_preprocessor

MODEL_LABELS = {
    "logistic_regression": "Regresion logistica",
    "random_forest": "Random Forest",
    "mlp": "Perceptron multicapa",
}


def build_models(
    random_state: int = RANDOM_STATE,
    rf_estimators: int = 300,
    n_jobs: int = -1,
) -> dict[str, Pipeline]:
    estimators = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=rf_estimators,
            class_weight="balanced_subsample",
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=n_jobs,
            random_state=random_state,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            batch_size=512,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=12,
            max_iter=250,
            random_state=random_state,
        ),
    }
    return {
        name: Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        for name, estimator in estimators.items()
    }


def fit_model(pipeline: Pipeline, X, y) -> Pipeline:
    """Ajusta un clon; usa sample_weight en MLP si la version lo admite."""
    fitted = clone(pipeline)
    estimator = fitted.named_steps["model"]
    fit_parameters: dict[str, np.ndarray] = {}
    if isinstance(estimator, MLPClassifier) and "sample_weight" in inspect.signature(estimator.fit).parameters:
        fit_parameters["model__sample_weight"] = compute_sample_weight("balanced", y)
    fitted.fit(X, y, **fit_parameters)
    return fitted


def train_models(models: Mapping[str, Pipeline], X, y) -> dict[str, Pipeline]:
    return {name: fit_model(model, X, y) for name, model in models.items()}


def risk_scores(model: Pipeline, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    values = np.asarray(model.decision_function(X), dtype=float)
    return 1.0 / (1.0 + np.exp(-values))
