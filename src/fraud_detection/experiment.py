"""Orquestacion reproducible del experimento completo."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .alerts import alert_summary, calibrate_alert_thresholds
from .config import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN, project_root
from .data import audit_and_deduplicate, load_dataset, stratified_split, temporal_split
from .evaluation import (
    classification_metrics,
    metrics_table,
    save_json,
    select_f1_threshold,
    select_model,
)
from .plotting import (
    plot_alert_distribution,
    plot_architecture,
    plot_class_distribution,
    plot_confusion,
    plot_precision_recall,
)
from .training import MODEL_LABELS, build_models, fit_model, risk_scores, train_models

REFERENCE_TEST = {
    "logistic_regression": {"average_precision": 0.73, "precision": 0.89, "recall": 0.75, "f1": 0.81},
    "random_forest": {"average_precision": 0.86, "precision": 0.95, "recall": 0.77, "f1": 0.85},
    "mlp": {"average_precision": 0.86, "precision": 0.88, "recall": 0.79, "f1": 0.83},
}


def _split_summary(splits) -> dict[str, dict[str, int]]:
    return {
        "train": {"rows": len(splits.X_train), "fraud": int(splits.y_train.sum())},
        "validation": {"rows": len(splits.X_validation), "fraud": int(splits.y_validation.sum())},
        "test": {"rows": len(splits.X_test), "fraud": int(splits.y_test.sum())},
    }


def _comparison_markdown(test_metrics: dict[str, dict[str, object]]) -> str:
    lines = [
        "# Comparacion con los resultados del documento academico",
        "",
        "Los valores del documento se usan solo como referencia. La tabla siguiente contiene los resultados calculados por esta ejecucion.",
        "",
        "| Modelo | AP real | AP referencia | Precision real | Recall real | F1 real |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("logistic_regression", "random_forest", "mlp"):
        actual = test_metrics[name]
        reference = REFERENCE_TEST[name]
        lines.append(
            f"| {MODEL_LABELS[name]} | {actual['average_precision']:.4f} | {reference['average_precision']:.2f} | "
            f"{actual['precision']:.4f} | {actual['recall']:.4f} | {actual['f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Las diferencias son esperables por la semilla, la version de scikit-learn, la implementacion exacta de los pesos, la convergencia del MLP y la seleccion discreta del umbral. No se alteraron resultados para hacerlos coincidir.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_experiment(
    data_path: str | Path | None = None,
    output_root: str | Path | None = None,
    rf_estimators: int = 300,
    n_jobs: int = -1,
) -> dict:
    root = Path(output_root) if output_root else project_root()
    data_path = Path(data_path) if data_path else root / "data" / "raw" / "creditcard.csv"
    figures = root / "reports" / "figures"
    metrics_dir = root / "reports" / "metrics"
    models_dir = root / "models"
    figures.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Cargando, auditando y deduplicando el dataset real...", flush=True)
    raw = load_dataset(data_path)
    clean, audit = audit_and_deduplicate(raw)
    if audit.missing_values:
        raise ValueError(f"El dataset contiene {audit.missing_values} valores faltantes")
    plot_class_distribution(clean[TARGET_COLUMN], figures / "01_distribucion_clases.png")
    plot_architecture(figures / "02_arquitectura_sistema.png")

    print("[2/7] Creando particiones estratificadas 60/20/20...", flush=True)
    splits = stratified_split(clean)
    split_info = _split_summary(splits)
    assert not (set(splits.X_train.index) & set(splits.X_validation.index))
    assert not (set(splits.X_train.index) & set(splits.X_test.index))
    assert not (set(splits.X_validation.index) & set(splits.X_test.index))

    print("[3/7] Entrenando regresion logistica, Random Forest y MLP...", flush=True)
    fitted = train_models(build_models(rf_estimators=rf_estimators, n_jobs=n_jobs), splits.X_train, splits.y_train)

    validation_metrics: dict[str, dict[str, object]] = {}
    validation_scores: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    for name, model in fitted.items():
        scores = risk_scores(model, splits.X_validation)
        threshold, _ = select_f1_threshold(splits.y_validation, scores)
        validation_scores[name] = scores
        thresholds[name] = threshold
        validation_metrics[name] = classification_metrics(splits.y_validation, scores, threshold)
    selected = select_model(validation_metrics)
    print(f"[4/7] Modelo seleccionado exclusivamente con validacion: {MODEL_LABELS[selected]}", flush=True)

    test_metrics: dict[str, dict[str, object]] = {}
    test_scores: dict[str, np.ndarray] = {}
    for name, model in fitted.items():
        scores = risk_scores(model, splits.X_test)
        test_scores[name] = scores
        test_metrics[name] = classification_metrics(splits.y_test, scores, thresholds[name])

    plot_precision_recall(splits.y_test, test_scores, figures / "03_curvas_precision_recall.png")
    plot_confusion(
        splits.y_test,
        test_scores[selected],
        thresholds[selected],
        figures / "04_matriz_confusion_modelo_seleccionado.png",
        MODEL_LABELS[selected],
    )

    print("[5/7] Calibrando niveles de alerta con capacidad de revision del 0,5 %...", flush=True)
    alert_thresholds = calibrate_alert_thresholds(validation_scores[selected], thresholds[selected], review_capacity=0.005)
    alerts = alert_summary(test_scores[selected], splits.y_test, alert_thresholds)
    alerts.to_csv(metrics_dir / "alertas_prueba.csv", index=False)
    plot_alert_distribution(alerts, figures / "05_distribucion_alertas.png")

    print("[6/7] Ejecutando comprobacion temporal sin fuga hacia el 20 % final...", flush=True)
    temporal_initial, temporal_test = temporal_split(clean)
    internal_boundary = int(len(temporal_initial) * 0.80)
    temporal_train_internal = temporal_initial.iloc[:internal_boundary]
    temporal_validation = temporal_initial.iloc[internal_boundary:]
    base_temporal_model = build_models(rf_estimators=rf_estimators, n_jobs=n_jobs)[selected]
    preliminary = fit_model(
        base_temporal_model,
        temporal_train_internal[FEATURE_COLUMNS],
        temporal_train_internal[TARGET_COLUMN],
    )
    preliminary_scores = risk_scores(preliminary, temporal_validation[FEATURE_COLUMNS])
    temporal_threshold, _ = select_f1_threshold(temporal_validation[TARGET_COLUMN], preliminary_scores)
    temporal_model = fit_model(
        base_temporal_model,
        temporal_initial[FEATURE_COLUMNS],
        temporal_initial[TARGET_COLUMN],
    )
    temporal_scores = risk_scores(temporal_model, temporal_test[FEATURE_COLUMNS])
    temporal_metrics = classification_metrics(temporal_test[TARGET_COLUMN], temporal_scores, temporal_threshold)
    temporal_metrics["model"] = selected
    temporal_metrics["train_rows"] = len(temporal_initial)
    temporal_metrics["test_rows"] = len(temporal_test)
    temporal_metrics["train_max_time"] = float(temporal_initial["Time"].max())
    temporal_metrics["test_min_time"] = float(temporal_test["Time"].min())

    print("[7/7] Guardando modelos, metricas, tablas y reporte comparativo...", flush=True)
    artifact = {
        "model": fitted[selected],
        "model_name": selected,
        "model_label": MODEL_LABELS[selected],
        "decision_threshold": thresholds[selected],
        "thresholds": alert_thresholds.to_dict(),
        "feature_columns": FEATURE_COLUMNS,
        "random_state": RANDOM_STATE,
        "score_note": "Puntaje de riesgo no calibrado; no interpretar como probabilidad.",
    }
    joblib.dump(artifact, models_dir / "modelo_seleccionado.joblib", compress=3)
    for name, model in fitted.items():
        joblib.dump(model, models_dir / f"{name}.joblib", compress=3)

    metrics_table(validation_metrics).to_csv(metrics_dir / "metricas_validacion.csv", index=False)
    metrics_table(test_metrics).to_csv(metrics_dir / "metricas_prueba.csv", index=False)
    save_json(validation_metrics, metrics_dir / "metricas_validacion.json")
    save_json(test_metrics, metrics_dir / "metricas_prueba.json")
    save_json(temporal_metrics, metrics_dir / "metricas_temporales.json")
    save_json(audit.to_dict(), metrics_dir / "auditoria_dataset.json")
    save_json(split_info, metrics_dir / "particiones.json")
    save_json(alert_thresholds.to_dict(), metrics_dir / "umbrales_alertas.json")
    (metrics_dir / "comparacion_documento.md").write_text(_comparison_markdown(test_metrics), encoding="utf-8")

    run_metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "random_state": RANDOM_STATE,
        "rf_estimators": rf_estimators,
        "selected_model": selected,
        "selection_rule": "Mayor AP; desempate por F1, precision y menor FP, todo en validacion.",
    }
    save_json(run_metadata, metrics_dir / "metadata_ejecucion.json")
    summary = {
        "audit": audit.to_dict(),
        "splits": split_info,
        "selected_model": selected,
        "validation": validation_metrics,
        "test": test_metrics,
        "alerts": json.loads(alerts.to_json(orient="records")),
        "temporal": temporal_metrics,
        "artifact": str(models_dir / "modelo_seleccionado.joblib"),
    }
    save_json(summary, metrics_dir / "resumen_experimento.json")
    print(json.dumps({"selected_model": selected, "test": test_metrics[selected], "temporal": temporal_metrics}, indent=2), flush=True)
    return summary
