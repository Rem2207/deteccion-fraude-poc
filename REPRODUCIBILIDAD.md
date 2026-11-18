# Reproduccion de los experimentos

## Condiciones registradas

- Python 3.11 o posterior.
- Semilla global: `42`.
- Dataset: `Credit Card Fraud Detection` de Machine Learning Group - ULB.
- Duplicados exactos eliminados antes de separar datos.
- Particion principal estratificada: 60 % entrenamiento, 20 % validacion y 20 % prueba.
- Umbrales elegidos solo con validacion; prueba se usa una unica vez para el resultado final.

## Ejecucion limpia

En PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe scripts\train.py
.\.venv\Scripts\python.exe -m pytest
```

En Linux/macOS:

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python scripts/download_data.py
./.venv/bin/python scripts/train.py
./.venv/bin/python -m pytest
```

La ejecucion vuelve a generar `reports/metrics/`, `reports/figures/` y los artefactos locales en `models/`. Los modelos y el dataset se excluyen de Git porque son reproducibles y voluminosos.

## Criterio de seleccion

Se selecciona el mayor Average Precision en validacion. Los desempates se resuelven por F1, precision y menor cantidad de falsos positivos, siempre en validacion. Para cada modelo, el umbral binario maximiza F1 en validacion.

## Comprobacion temporal

Las filas se ordenan por `Time`. El 80 % inicial se usa para entrenar y el 20 % final para evaluar. Dentro del tramo inicial se reserva una ventana interna para elegir el umbral; luego el modelo se reajusta con el 80 % completo. No se ajusta ninguna transformacion con el 20 % final.

## Actualizacion del informe

1. Copiar la tabla de `reports/metrics/metricas_prueba.csv`.
2. Copiar la matriz y curvas de `reports/figures/`.
3. Consultar `comparacion_documento.md` para explicar diferencias.
4. Registrar las versiones de `metadata_ejecucion.json`.
5. No copiar los valores de referencia como si fueran resultados de una nueva ejecucion.
