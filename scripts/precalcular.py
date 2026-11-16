"""Recalcula los resultados que muestra la app y los guarda en data/.

Uso:  python scripts/precalcular.py
Tarda unos minutos. Evalúa nueve modelos ligeros con origen móvil (walk-forward) y guarda:
  data/series.csv        series semanales (mercado real + costos internos)
  data/predicciones.csv  predicción de cada modelo en cada semana, con su rango calibrado
  data/metricas.csv      métricas en validación y en prueba
  data/ejemplo_erp.csv   archivo de ejemplo con el formato que debe exportar el ERP
"""
import sys, warnings
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ)); warnings.filterwarnings("ignore")
import pandas as pd
from motor_costos import datos, backtest as B
from motor_costos.config import FAMILIAS
from motor_costos.servicio import PARAMS, MODELOS_APP, empaquetar

df = datos.generar_sintetico()
CAL, T = {}, []
for fam in FAMILIAS:
    res, _ = B.walk_forward(df, fam, MODELOS_APP, params=PARAMS[fam])
    tab, CAL[fam] = B.evaluar(res, fam); T.append(tab)
    print(fam, "listo")
pred = empaquetar(CAL)
df.to_csv(RAIZ / "data/series.csv", index=False)
pred.to_csv(RAIZ / "data/predicciones.csv", index=False)
pd.concat(T).to_csv(RAIZ / "data/metricas.csv", index=False)
df.rename(columns={"date": "fecha_semana"})[list(datos.COLUMNAS_ERP)].round(4).to_csv(RAIZ / "data/ejemplo_erp.csv", index=False)
print("Archivos guardados en data/")
