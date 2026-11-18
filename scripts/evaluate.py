#!/usr/bin/env python
"""Evalua el artefacto seleccionado sobre la particion de prueba reproducible."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraud_detection.data import audit_and_deduplicate, load_dataset, stratified_split
from fraud_detection.evaluation import classification_metrics
from fraud_detection.prediction import load_artifact
from fraud_detection.training import risk_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/creditcard.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/modelo_seleccionado.joblib"))
    args = parser.parse_args()
    clean, _ = audit_and_deduplicate(load_dataset(args.data))
    splits = stratified_split(clean)
    artifact = load_artifact(args.model)
    scores = risk_scores(artifact["model"], splits.X_test)
    result = classification_metrics(splits.y_test, scores, artifact["decision_threshold"])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
