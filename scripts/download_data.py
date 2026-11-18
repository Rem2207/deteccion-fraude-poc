#!/usr/bin/env python
"""Descarga el dataset publico mediante la API oficial de Kaggle y lo extrae."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import requests

API_URL = "https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "creditcard.csv"
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    auth = (username, key) if username and key else None
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        with requests.get(API_URL, auth=auth, stream=True, timeout=120) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                temporary.write(chunk)
    try:
        with zipfile.ZipFile(temporary_path) as archive:
            member = next(name for name in archive.namelist() if name.endswith("creditcard.csv"))
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Dataset descargado en {target} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
