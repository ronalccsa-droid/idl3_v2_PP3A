"""Capa de servicio de la app: empaqueta resultados, calcula una cotización y recalcula con datos del ERP."""
import numpy as np
import pandas as pd
from . import backtest as B, contingencia as C, datos
from .config import FAMILIAS, INICIO_TEST, CONTINGENCIA_BASE

# Hiperparámetros elegidos en validación (paso 3 del cuaderno)
PARAMS = {"costo_mac": {"xgb": {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 150},
                        "lgbm": {"num_leaves": 8, "learning_rate": 0.03, "n_estimators": 150, "min_child_samples": 5}},
          "costo_tierras": {"xgb": {"max_depth": 2, "learning_rate": 0.1, "n_estimators": 400},
                            "lgbm": {"num_leaves": 8, "learning_rate": 0.1, "n_estimators": 300, "min_child_samples": 10}}}
MODELOS_APP = ["ultimo", "pm4", "ets", "sarimax", "arx", "qr", "xgb", "lgbm"]   # el ensamble se deriva de qr + xgb + lgbm
CAMPEON = {"costo_mac": "lgbm", "costo_tierras": "qr"}          # elegidos en validación por puntaje de Winkler
NOMBRE_MODELO = {"ultimo": "Práctica actual (último costo)", "pm4": "Promedio móvil", "ets": "Suavizamiento exponencial",
                 "sarimax": "SARIMAX", "arx": "ARX (lineal)", "qr": "Regresión cuantílica", "xgb": "XGBoost",
                 "lgbm": "LightGBM", "ens": "Ensamble"}
UNIDAD = {"costo_mac": "t", "costo_tierras": "m³"}
NOMBRE_CORTO = {"costo_mac": "Mezcla asfáltica (MAC)", "costo_tierras": "Movimiento de tierras"}
MIN_SEMANAS = 100


def empaquetar(cal: dict) -> pd.DataFrame:
    """Convierte los resultados calibrados {familia: {modelo: tabla}} en una sola tabla plana."""
    filas = []
    for fam, modelos in cal.items():
        for m, r in modelos.items():
            d = pd.DataFrame({"familia": fam, "modelo": m, "t": r.index, "date": r.date.values, "y": r.y.values,
                              "q50": r.q50.values,
                              "lo": r.lo.values if "lo" in r else np.nan, "hi": r.hi.values if "hi" in r else np.nan})
            filas.append(d)
    return pd.concat(filas, ignore_index=True)


def tabla(pred: pd.DataFrame, fam: str, modelo: str) -> pd.DataFrame:
    return pred[(pred.familia == fam) & (pred.modelo == modelo)].set_index("t").sort_index()


def cotizar(pred, fam, t, tipo, meses, tope, modelo=None, base=CONTINGENCIA_BASE) -> dict:
    """Cotización de la semana t: costo esperado, rango, contingencia del motor y práctica actual."""
    r = tabla(pred, fam, modelo or CAMPEON[fam]).loc[t]; ult = tabla(pred, fam, "ultimo").loc[t].q50
    k = float(C.kappa(r.q50, r.hi)); f = C.FACTOR_CONTRATO[tipo]; plazo = float(np.sqrt(max(meses, 1)))
    sin_tope = k * f * plazo; cont = min(sin_tope, tope)
    return dict(fecha=pd.Timestamp(r.date), p10=float(r.lo), p50=float(r.q50), p90=float(r.hi), riesgo=k, factor=f, plazo=plazo,
                sin_tope=sin_tope, contingencia=cont, topado=sin_tope > tope, precio=float(r.q50 * (1 + cont)),
                ultimo=float(ult), precio_base=float(ult * (1 + base)), real=float(r.y) if pd.notna(r.y) else None)


def comparacion(pred, fam, modelo=None) -> dict:
    """Práctica actual frente al motor en el periodo de prueba."""
    r = tabla(pred, fam, modelo or CAMPEON[fam]); te = r[r.index >= INICIO_TEST]
    return C.backtest(te, tabla(pred, fam, "ultimo").q50)


def recalcular(archivo, progreso=None):
    """Repite la evaluación con un historial del ERP (formato de data/plantilla_erp.csv).
    Usa modelos ligeros para responder en poco tiempo. Devuelve series, predicciones, métricas e informe de calidad."""
    df = datos.cargar_erp(archivo)
    if len(df) < MIN_SEMANAS:
        raise ValueError(f"Se necesitan al menos {MIN_SEMANAS} semanas de historia; el archivo tiene {len(df)}.")
    calidad = datos.validar_esquema(df)
    cal, tabs = {}, []
    for i, fam in enumerate(FAMILIAS):
        if progreso:
            progreso(i / len(FAMILIAS), f"Evaluando {NOMBRE_CORTO[fam]}…")
        modelos = ["ultimo", "pm4", "arx", "qr"] + ([CAMPEON[fam]] if CAMPEON[fam] not in ("qr",) else [])
        res, _ = B.walk_forward(df, fam, modelos, params=PARAMS[fam])
        tab, cal[fam] = B.evaluar(res, fam); tabs.append(tab)
    if progreso:
        progreso(1.0, "Listo")
    return df, empaquetar(cal), pd.concat(tabs, ignore_index=True), calidad
