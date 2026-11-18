# Comparacion con los resultados del documento academico

Los valores del documento se usan solo como referencia. La tabla siguiente contiene los resultados calculados por esta ejecucion.

| Modelo | AP real | AP referencia | Precision real | Recall real | F1 real |
|---|---:|---:|---:|---:|---:|
| Regresion logistica | 0.7253 | 0.73 | 0.8875 | 0.7474 | 0.8114 |
| Random Forest | 0.8525 | 0.86 | 0.9487 | 0.7789 | 0.8555 |
| Perceptron multicapa | 0.8291 | 0.86 | 0.7917 | 0.8000 | 0.7958 |

Las diferencias son esperables por la semilla, la version de scikit-learn, la implementacion exacta de los pesos, la convergencia del MLP y la seleccion discreta del umbral. No se alteraron resultados para hacerlos coincidir.
