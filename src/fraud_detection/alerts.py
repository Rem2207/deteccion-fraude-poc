"""Reglas operativas para niveles de riesgo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlertThresholds:
    medium: float
    high: float
    review_capacity: float

    def to_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def calibrate_alert_thresholds(
    scores,
    high_threshold: float,
    review_capacity: float = 0.005,
) -> AlertThresholds:
    if not 0 < review_capacity < 1:
        raise ValueError("review_capacity debe estar entre 0 y 1")
    values = np.asarray(scores, dtype=float)
    capacity_cutoff = float(np.quantile(values, 1.0 - review_capacity, method="higher"))
    medium = min(float(high_threshold), capacity_cutoff)
    return AlertThresholds(medium=medium, high=float(high_threshold), review_capacity=review_capacity)


def classify_risk(scores, thresholds: AlertThresholds) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    return np.where(
        values >= thresholds.high,
        "alto",
        np.where(values >= thresholds.medium, "medio", "bajo"),
    )


def alert_summary(scores, y_true, thresholds: AlertThresholds) -> pd.DataFrame:
    levels = classify_risk(scores, thresholds)
    frame = pd.DataFrame({"nivel": levels, "fraude": np.asarray(y_true, dtype=int)})
    rows = []
    total_fraud = max(int(frame["fraude"].sum()), 1)
    for level in ("alto", "medio", "bajo"):
        subset = frame.loc[frame["nivel"] == level]
        fraud = int(subset["fraude"].sum())
        operations = len(subset)
        rows.append(
            {
                "nivel": level,
                "operaciones": operations,
                "fraudes": fraud,
                "alertas_incorrectas": int(operations - fraud) if level != "bajo" else 0,
                "porcentaje_fraudes": float(fraud / total_fraud),
                "porcentaje_operaciones": float(operations / len(frame)),
            }
        )
    return pd.DataFrame(rows)
