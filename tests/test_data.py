import pandas as pd
import pytest

from fraud_detection.data import audit_and_deduplicate, stratified_split, validate_columns


def test_validate_columns_rejects_missing(sample_frame):
    with pytest.raises(ValueError, match="Faltan columnas"):
        validate_columns(sample_frame.drop(columns="V28"))


def test_deduplication_happens_exactly(sample_frame):
    duplicated = pd.concat([sample_frame, sample_frame.iloc[[3, 4]]], ignore_index=True)
    clean, audit = audit_and_deduplicate(duplicated)
    assert len(clean) == len(sample_frame)
    assert audit.duplicate_rows == 2
    assert audit.duplicate_fraud == 1


def test_stratified_splits_are_disjoint_and_60_20_20(sample_frame):
    splits = stratified_split(sample_frame)
    assert (len(splits.X_train), len(splits.X_validation), len(splits.X_test)) == (120, 40, 40)
    assert set(splits.X_train.index).isdisjoint(splits.X_validation.index)
    assert set(splits.X_train.index).isdisjoint(splits.X_test.index)
    assert set(splits.X_validation.index).isdisjoint(splits.X_test.index)
    assert (splits.y_train.sum(), splits.y_validation.sum(), splits.y_test.sum()) == (5, 2, 1)
