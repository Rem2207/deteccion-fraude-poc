"""Carga, validacion, limpieza y particion de datos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURE_COLUMNS, RANDOM_STATE, REQUIRED_COLUMNS, TARGET_COLUMN


@dataclass(frozen=True)
class DatasetAudit:
    rows_original: int
    fraud_original: int
    missing_values: int
    duplicate_rows: int
    duplicate_fraud: int
    rows_clean: int
    fraud_clean: int

    def to_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class DataSplits:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def validate_columns(frame: pd.DataFrame) -> None:
    """Valida el esquema y la etiqueta sin modificar el DataFrame."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")
    unexpected_target = set(frame[TARGET_COLUMN].dropna().unique()) - {0, 1}
    if unexpected_target:
        raise ValueError(f"Class debe contener solo 0 y 1: {unexpected_target}")
    non_numeric = [name for name in REQUIRED_COLUMNS if not pd.api.types.is_numeric_dtype(frame[name])]
    if non_numeric:
        raise ValueError(f"Las columnas deben ser numericas: {non_numeric}")


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro {path}. Ejecute: python scripts/download_data.py"
        )
    frame = pd.read_csv(path)
    validate_columns(frame)
    return frame.loc[:, REQUIRED_COLUMNS]


def audit_and_deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, DatasetAudit]:
    """Elimina duplicados exactos antes de cualquier particion."""
    validate_columns(frame)
    duplicates = frame.duplicated(keep="first")
    clean = frame.loc[~duplicates].copy().reset_index(drop=True)
    audit = DatasetAudit(
        rows_original=len(frame),
        fraud_original=int(frame[TARGET_COLUMN].sum()),
        missing_values=int(frame[REQUIRED_COLUMNS].isna().sum().sum()),
        duplicate_rows=int(duplicates.sum()),
        duplicate_fraud=int(frame.loc[duplicates, TARGET_COLUMN].sum()),
        rows_clean=len(clean),
        fraud_clean=int(clean[TARGET_COLUMN].sum()),
    )
    return clean, audit


def stratified_split(
    frame: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> DataSplits:
    """Particion estratificada 60/20/20 con indices disjuntos."""
    validate_columns(frame)
    X = frame[FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN].astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        stratify=y,
        random_state=random_state,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state,
    )
    return DataSplits(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )


def temporal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve el 80 % inicial y el 20 % final, ordenados por Time."""
    validate_columns(frame)
    ordered = frame.sort_values("Time", kind="mergesort").reset_index(drop=True)
    boundary = int(len(ordered) * 0.80)
    return ordered.iloc[:boundary].copy(), ordered.iloc[boundary:].copy()
