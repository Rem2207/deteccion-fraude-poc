"""Graficos listos para el informe academico."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve

from .training import MODEL_LABELS


def _save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_class_distribution(y, path: str | Path) -> None:
    counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.barh(["Legitimas", "Fraudulentas"], counts, color=["#167d8d", "#c06d56"])
    ax.set_xscale("log")
    ax.set_xlabel("Cantidad de transacciones (escala logaritmica)")
    ax.set_title("Distribucion de operaciones por clase")
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, counts):
        ax.text(value * 1.08, bar.get_y() + bar.get_height() / 2, f"{value:,}".replace(",", "."), va="center")
    _save(fig, path)


def plot_architecture(path: str | Path) -> None:
    labels = [
        ("1", "DATOS", "Transacciones\nanonimizadas"),
        ("2", "PREPARACION", "Limpieza, log y\nescalado"),
        ("3", "MODELOS", "Estimacion del\nriesgo"),
        ("4", "REGLAS", "Umbrales y\npriorizacion"),
        ("5", "SALIDA", "Alerta y revision\nmanual"),
    ]
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    for index, (number, title, subtitle) in enumerate(labels):
        x = 0.25 + index * 2.4
        box = FancyBboxPatch((x, 0.65), 2.0, 1.65, boxstyle="round,pad=0.08", fc="#eef6f7", ec="#9bc6cb", lw=1.4)
        ax.add_patch(box)
        ax.text(x + 0.18, 2.02, number, color="#167d8d", weight="bold", fontsize=12)
        ax.text(x + 0.18, 1.58, title, color="#173a4d", weight="bold", fontsize=10)
        ax.text(x + 0.18, 1.15, subtitle, color="#526b78", fontsize=9, va="top")
        if index < len(labels) - 1:
            ax.annotate("", xy=(x + 2.36, 1.48), xytext=(x + 2.05, 1.48), arrowprops={"arrowstyle": "->", "color": "#167d8d", "lw": 1.6})
    ax.set_title("Arquitectura del sistema de deteccion de fraude", fontsize=14, weight="bold")
    _save(fig, path)


def plot_precision_recall(y_true, model_scores: dict[str, np.ndarray], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    colors = ["#7f8f98", "#167d8d", "#c06d56"]
    from sklearn.metrics import average_precision_score

    for color, (name, scores) in zip(colors, model_scores.items()):
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(recall, precision, lw=2, color=color, label=f"{MODEL_LABELS[name]} (AP = {ap:.3f})")
    baseline = float(np.mean(y_true))
    ax.axhline(baseline, color="#999999", ls="--", lw=1, label=f"Azar (AP = {baseline:.4f})")
    ax.set(xlabel="Recall (fraudes detectados)", ylabel="Precision (alertas correctas)", xlim=(0, 1), ylim=(0, 1.02))
    ax.set_title("Curvas Precision-Recall en el conjunto de prueba")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    _save(fig, path)


def plot_confusion(y_true, scores, threshold: float, path: str | Path, model_label: str) -> None:
    predictions = (np.asarray(scores) >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=["Legitima", "Fraude"],
        cmap="Blues",
        colorbar=False,
        values_format=",d",
        ax=ax,
    )
    display.ax_.set_title(f"Matriz de confusion - {model_label}\nUmbral elegido en validacion: {threshold:.4f}")
    display.ax_.set_xlabel("Prediccion")
    display.ax_.set_ylabel("Valor real")
    _save(fig, path)


def plot_alert_distribution(summary, path: str | Path) -> None:
    order = ["alto", "medio", "bajo"]
    frame = summary.set_index("nivel").loc[order]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    colors = ["#b9443f", "#e5a84b", "#4f8f72"]
    bars = ax.bar(["Alto", "Medio", "Bajo"], frame["operaciones"], color=colors)
    ax.set_yscale("log")
    ax.set_ylim(55, float(frame["operaciones"].max()) * 2.2)
    ax.set_ylabel("Operaciones (escala logaritmica)")
    ax.set_title("Distribucion de operaciones por nivel de riesgo", pad=14)
    ax.grid(axis="y", alpha=0.25)
    for bar, (_, row) in zip(bars, frame.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.08, f"{int(row['operaciones']):,}\n({int(row['fraudes'])} fraudes)".replace(",", "."), ha="center", va="bottom")
    _save(fig, path)
