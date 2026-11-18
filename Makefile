.PHONY: install download train evaluate test notebook clean

PYTHON ?= python

install:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt

download:
	$(PYTHON) scripts/download_data.py

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate.py

test:
	$(PYTHON) -m pytest

notebook:
	$(PYTHON) -m jupyter lab notebooks/01_analisis_entrenamiento.ipynb

clean:
	$(PYTHON) -c "from pathlib import Path; [p.unlink() for p in Path('reports/metrics').glob('*') if p.is_file()]"
