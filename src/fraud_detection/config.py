"""Configuracion compartida del experimento."""

from pathlib import Path

RANDOM_STATE = 42
TARGET_COLUMN = "Class"
TIME_COLUMN = "Time"
AMOUNT_COLUMN = "Amount"
PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = [TIME_COLUMN, *PCA_COLUMNS, AMOUNT_COLUMN]
REQUIRED_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
