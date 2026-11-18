"""Carga del artefacto y prediccion de registros nuevos."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .alerts import AlertThresholds, classify_risk
from .config import FEATURE_COLUMNS
from .training import risk_scores


def load_artifact(path: str | Path) -> dict:
    artifact = joblib.load(path)
    required = {"model", "model_name", "thresholds"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Artefacto incompleto: {sorted(missing)}")
    return artifact


def predict_records(artifact: dict, records: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(FEATURE_COLUMNS) - set(records.columns))
    if missing:
        raise ValueError(f"Faltan variables para predecir: {missing}")
    X = records.loc[:, FEATURE_COLUMNS]
    scores = risk_scores(artifact["model"], X)
    thresholds = AlertThresholds(**artifact["thresholds"])
    levels = classify_risk(scores, thresholds)
    return pd.DataFrame(
        {
            "puntaje_riesgo": scores,
            "nivel_riesgo": levels,
            "requiere_revision": levels != "bajo",
        },
        index=records.index,
    )
