"""Transformaciones sin fuga de informacion."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .config import AMOUNT_COLUMN, PCA_COLUMNS, TIME_COLUMN


def build_preprocessor() -> ColumnTransformer:
    """Log1p + escalado para Amount, escalado para Time y paso directo de V1..V28."""
    amount_pipeline = Pipeline(
        steps=[
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("amount", amount_pipeline, [AMOUNT_COLUMN]),
            ("time", StandardScaler(), [TIME_COLUMN]),
            ("pca", "passthrough", PCA_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
