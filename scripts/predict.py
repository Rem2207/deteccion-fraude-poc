#!/usr/bin/env python
"""Evalua una o mas transacciones desde un CSV sin la columna Class."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fraud_detection.prediction import load_artifact, predict_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/modelo_seleccionado.joblib"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = pd.read_csv(args.input_csv)
    result = predict_records(load_artifact(args.model), records)
    combined = records.join(result)
    if args.output:
        combined.to_csv(args.output, index=False)
        print(f"Resultados guardados en {args.output}")
    else:
        print(combined.to_csv(index=False))


if __name__ == "__main__":
    main()
