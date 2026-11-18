from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import PCA_COLUMNS


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 200
    frame = pd.DataFrame({"Time": np.arange(rows, dtype=float) * 10, "Amount": rng.lognormal(3, 1, rows)})
    for column in PCA_COLUMNS:
        frame[column] = rng.normal(size=rows)
    frame["Class"] = 0
    frame.loc[[3, 20, 51, 80, 111, 140, 170, 199], "Class"] = 1
    return frame
