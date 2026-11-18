#!/usr/bin/env python
"""Entrena y evalua el experimento completo."""

from __future__ import annotations

import argparse
from pathlib import Path

from fraud_detection.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/creditcard.csv"))
    parser.add_argument("--rf-estimators", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    run_experiment(data_path=args.data, rf_estimators=args.rf_estimators, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
