"""Seleccion de umbrales, metricas y persistencia de resultados."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)


def select_f1_threshold(y_true, scores) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return 0.5, 0.0
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    best = int(np.nanargmax(f1))
    return float(thresholds[best]), float(f1[best])


def classification_metrics(y_true, scores, threshold: float) -> dict[str, object]:
    predictions = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_samples": int(len(predictions)),
        "n_fraud": int(np.asarray(y_true).sum()),
    }


def select_model(validation_metrics: dict[str, dict[str, object]]) -> str:
    """Prioriza AP, luego F1, precision y menos falsos positivos en validacion."""
    return max(
        validation_metrics,
        key=lambda name: (
            validation_metrics[name]["average_precision"],
            validation_metrics[name]["f1"],
            validation_metrics[name]["precision"],
            -validation_metrics[name]["fp"],
        ),
    )


def metrics_table(metrics: dict[str, dict[str, object]]) -> pd.DataFrame:
    columns = [
        "model",
        "average_precision",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "threshold",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    rows = [{"model": name, **values} for name, values in metrics.items()]
    return pd.DataFrame(rows).loc[:, columns]


def save_json(data: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
