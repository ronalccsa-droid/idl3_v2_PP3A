"""Motor de costos predictivos · Tablero para el analista de presupuestos.

Ejecutar:  streamlit run app.py
"""
import sys
import warnings
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Comprobación de instalación: avisa con claridad si al repositorio le falta algo
_faltan = [r for r in ("motor_costos/servicio.py", "data/predicciones.csv", "data/metricas.csv", "data/series.csv")
           if not (RAIZ / r).exists()]
if _faltan:
    st.error("Al repositorio le faltan archivos. Suba también estas rutas, junto a app.py:\n\n"
             + "\n".join(f"- `{r}`" for r in _faltan))
    st.stop()

from motor_costos import contingencia as C, drift, economico as E, features, servicio as S
from motor_costos.config import CONTINGENCIA_BASE, INICIO_TEST

# ----------------------------------------------------------------------------------------------
# Estilo
# ----------------------------------------------------------------------------------------------
AZUL, NARANJA, GRIS, TINTA, SUAVE, VERDE = "#1F5FBF", "#D9601E", "#B9B8B2", "#14171F", "#5B6270", "#1E8E5A"
st.set_page_config(page_title="Motor de costos predictivos", page_icon="📐", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 3.4rem; padding-bottom: 3rem; max-width: 1280px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.marca {font-size: 0.78rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #1F5FBF;}
.titulo {font-size: 2.0rem; font-weight: 700; line-height: 1.15; margin: 0.15rem 0 0.3rem 0; color: #14171F;}
.bajada {color: #5B6270; font-size: 1.0rem; margin-bottom: 0.6rem;}
.chip {display: inline-block; padding: 0.22rem 0.7rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
       margin-right: 0.4rem; border: 1px solid transparent;}
.chip-demo {background: #FDF1EA; color: #A8460F; border-color: #F3C9B0;}
.chip-real {background: #E7F5EE; color: #17694A; border-color: #B5DFCB;}
.chip-info {background: #EEF3FB; color: #1F4E8C; border-color: #C9D9F2;}
.kpi {background: #FFFFFF; border: 1px solid #E3E6EC; border-radius: 14px; padding: 1.0rem 1.1rem; min-height: 134px;}
.kpi .et {font-size: 0.74rem; font-weight: 600; color: #5B6270; text-transform: uppercase; letter-spacing: 0.05em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.kpi .va {font-size: 1.5rem; font-weight: 700; color: #14171F; line-height: 1.25; margin-top: 0.3rem; white-space: nowrap; font-variant-numeric: tabular-nums;}
.kpi .no {font-size: 0.85rem; color: #5B6270; margin-top: 0.2rem;}
.kpi.destacado {background: #1F5FBF; border-color: #1F5FBF;}
.kpi.destacado .et, .kpi.destacado .no {color: #DCE8FA;}
.kpi.destacado .va {color: #FFFFFF;}
.paso {display: flex; align-items: baseline; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #EEF0F4;}
.paso:last-child {border-bottom: none;}
.paso .n {color: #5B6270; font-size: 0.92rem;}
.paso .v {font-weight: 700; color: #14171F; font-variant-numeric: tabular-nums;}
.paso.total .n, .paso.total .v {color: #1F5FBF; font-size: 1.02rem; font-weight: 700;}
.nota {color: #5B6270; font-size: 0.88rem;}
div[data-testid="stTabs"] button p {font-size: 0.98rem; font-weight: 600;}
#MainMenu, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def num(x, dec=2, signo=False):
    """Número con formato peruano: coma decimal y espacio fino de miles."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{x:+,.{dec}f}" if signo else f"{x:,.{dec}f}"
    return s.replace(",", " ").replace(".", ",")


def soles(x, dec=2):
    return f"S/ {num(x, dec)}"


def pct(x, dec=1, signo=False):
    return f"{num(x, dec, signo)} %"


def kpi(etiqueta, valor, nota="", destacado=False):
    cls = "kpi destacado" if destacado else "kpi"
    st.markdown(f'<div class="{cls}"><div class="et">{etiqueta}</div><div class="va">{valor}</div>'
                f'<div class="no">{nota}</div></div>', unsafe_allow_html=True)


def figura(fig, alto=360, fechas=True):
    fig.update_layout(template="simple_white", height=alto, margin=dict(l=10, r=10, t=30, b=10),
                      font=dict(family="Inter, system-ui, sans-serif", size=13, color=TINTA),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
                      hoverlabel=dict(bgcolor="white", font_size=13), separators=", ")
    fig.update_xaxes(showgrid=False, linecolor="#D5D9E0", tickcolor="#D5D9E0")
    if fechas:
        fig.update_xaxes(tickformat="%m/%Y", hoverformat="%d/%m/%Y")
    fig.update_yaxes(gridcolor="#EEF0F4", linecolor="rgba(0,0,0,0)", tickcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ----------------------------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar_demo():
    series = pd.read_csv(RAIZ / "data/series.csv", parse_dates=["date"])
    pred = pd.read_csv(RAIZ / "data/predicciones.csv", parse_dates=["date"])
    met = pd.read_csv(RAIZ / "data/metricas.csv")
    return series, pred, met


@st.cache_data(show_spinner=False)
def montecarlo(cont_t: tuple, cont_m: tuple):
    return E.montecarlo({"costo_tierras": dict(cont_t), "costo_mac": dict(cont_m)}, n=4000)


if "erp" in st.session_state:
    series, pred, met, calidad = st.session_state["erp"]
    fuente_real = True
else:
    series, pred, met = cargar_demo()
    calidad, fuente_real = None, False

TIPOS = list(C.FACTOR_CONTRATO)

# ----------------------------------------------------------------------------------------------
# Encabezado
# ----------------------------------------------------------------------------------------------
st.markdown('<div class="marca">INDUMAQ S.R.L. · Presupuestos</div>'
            '<div class="titulo">Motor de costos predictivos</div>'
            '<div class="bajada">Cuánto costará producir en las próximas cuatro semanas, con qué margen de error, '
            'y qué contingencia corresponde a cada contrato.</div>', unsafe_allow_html=True)
if fuente_real:
    st.markdown(f'<span class="chip chip-real">Datos del ERP · {len(series)} semanas</span>'
                '<span class="chip chip-info">Modelos recalculados en esta sesión</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="chip chip-demo">Datos de demostración</span>'
                '<span class="chip chip-info">Brent y tipo de cambio: reales</span>'
                '<span class="chip chip-info">Costos internos: simulados</span>', unsafe_allow_html=True)
st.write("")

t_cot, t_des, t_mon, t_eco, t_dat = st.tabs(["Cotizador", "Desempeño", "Monitoreo", "Análisis económico", "Datos y método"])

# ----------------------------------------------------------------------------------------------
# 1. Cotizador
# ----------------------------------------------------------------------------------------------
with t_cot:
    izq, der = st.columns([1, 2.4], gap="large")
    with izq:
        st.subheader("Datos de la cotización")
        fam = st.radio("Producto", list(S.NOMBRE_CORTO), format_func=S.NOMBRE_CORTO.get, key="fam_cot")
        un = S.UNIDAD[fam]
        tipo = st.selectbox("Tipo de contrato", TIPOS, index=2,
                            help="Define quién asume el riesgo de precio. Con fórmula polinómica, buena parte pasa al cliente.")
        meses = st.slider("Plazo de ejecución (meses)", 1, 12, 3,
                          help="A mayor plazo, mayor incertidumbre: la contingencia crece con la raíz del plazo.")
        cantidad = st.number_input(f"Cantidad ({un})", min_value=0.0, value=5000.0 if fam == "costo_mac" else 100000.0,
                                   step=500.0, format="%.0f")
        r_camp = S.tabla(pred, fam, S.CAMPEON[fam])
        fechas = {t: pd.Timestamp(d).strftime("%d/%m/%Y") for t, d in r_camp.date.items()}
        t_sel = st.select_slider("Semana de la cotización", options=list(fechas), value=list(fechas)[-1],
                                 format_func=fechas.get,
                                 help="Se muestran semanas ya cerradas para poder comparar con el costo que realmente ocurrió.")
        with st.expander("Política de la gerencia"):
            tope = st.slider("Tope de contingencia (%)", 10, 40, 25) / 100
            st.caption("El piso es 1 %. Los factores por contrato están en la pestaña Datos y método.")

    q = S.cotizar(pred, fam, t_sel, tipo, meses, tope)
    with der:
        st.subheader(f"{S.NOMBRE_CORTO[fam]} · semana del {q['fecha'].strftime('%d/%m/%Y')}")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi("Costo esperado · P50", soles(q["p50"]), f"por {un}, próximas 4 semanas")
        with c2:
            kpi("Rango P10 – P90", f"{num(q['p10'])} – {num(q['p90'])}", "8 de cada 10 casos")
        with c3:
            kpi("Contingencia", pct(100 * q["contingencia"]), "tope aplicado" if q["topado"] else "según riesgo, contrato y plazo")
        with c4:
            kpi("Costo protegido", soles(q["precio"]), f"por {un}, antes de utilidad", destacado=True)

        st.write("")
        # Gráfico: historia reciente + rango de las próximas 4 semanas
        h = series[series.index <= t_sel].tail(34)
        paso = (series.date.iloc[1] - series.date.iloc[0])
        f0, f1 = q["fecha"] + paso, q["fecha"] + 4 * paso
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h.date, y=h[fam], mode="lines", name="Costo semanal observado",
                                 line=dict(color=TINTA, width=2), hovertemplate="%{y:,.2f}<extra>Observado</extra>"))
        fig.add_trace(go.Scatter(x=[f0, f1, f1, f0], y=[q["p10"], q["p10"], q["p90"], q["p90"]], fill="toself",
                                 mode="lines", fillcolor="rgba(31,95,191,0.16)", line=dict(width=0), name="Rango P10–P90", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[f0, f1], y=[q["p50"]] * 2, mode="lines", name="Costo esperado (P50)",
                                 line=dict(color=AZUL, width=3), hovertemplate="%{y:,.2f}<extra>P50</extra>"))
        fig.add_trace(go.Scatter(x=[f0, f1], y=[q["precio"]] * 2, mode="lines", name="Precio de costo protegido",
                                 line=dict(color=AZUL, width=2, dash="dot"), hovertemplate="%{y:,.2f}<extra>Protegido</extra>"))
        fig.add_trace(go.Scatter(x=[f0, f1], y=[q["precio_base"]] * 2, mode="lines", name="Práctica actual (último + 5 %)",
                                 line=dict(color=NARANJA, width=2, dash="dash"), hovertemplate="%{y:,.2f}<extra>Práctica actual</extra>"))
        if q["real"] is not None:
            fig.add_trace(go.Scatter(x=[f0 + (f1 - f0) / 2], y=[q["real"]], mode="markers", name="Costo real posterior",
                                     marker=dict(color=VERDE, size=12, symbol="diamond", line=dict(color="white", width=2)),
                                     hovertemplate="%{y:,.2f}<extra>Costo real posterior</extra>"))
        fig.update_yaxes(title=f"S/ por {un}")
        figura(fig, 350)

        a, b = st.columns([1.05, 1], gap="large")
        with a:
            st.markdown("**Cómo se calculó la contingencia**")
            filas = [("Riesgo de precio · (P90 − P50) / P50", pct(100 * q["riesgo"])),
                     (f"× Factor del contrato", f"× {num(q['factor'], 1)}"),
                     (f"× Raíz del plazo · √{meses}", f"× {num(q['plazo'], 2)}"),
                     ("= Resultado", pct(100 * q["sin_tope"]))]
            if q["topado"]:
                filas.append((f"Tope de la gerencia", pct(100 * tope)))
            html = "".join(f'<div class="paso"><span class="n">{n}</span><span class="v">{v}</span></div>' for n, v in filas)
            html += f'<div class="paso total"><span class="n">Contingencia aplicada</span><span class="v">{pct(100 * q["contingencia"])}</span></div>'
            st.markdown(html, unsafe_allow_html=True)
        with b:
            st.markdown("**Frente a la práctica actual**")
            dif = 100 * (q["precio"] / q["precio_base"] - 1)
            filas = [("Último costo conocido", soles(q["ultimo"])),
                     ("Práctica actual · último costo + 5 %", soles(q["precio_base"])),
                     ("Motor · P50 + contingencia", soles(q["precio"])),
                     ("Diferencia del motor", pct(dif, 1, signo=True))]
            st.markdown("".join(f'<div class="paso"><span class="n">{n}</span><span class="v">{v}</span></div>' for n, v in filas),
                        unsafe_allow_html=True)
            st.markdown(f'<div class="paso total"><span class="n">Costo protegido de la partida</span>'
                        f'<span class="v">{soles(q["precio"] * cantidad, 0)}</span></div>', unsafe_allow_html=True)

        if q["real"] is not None:
            ok_m, ok_b = q["precio"] >= q["real"], q["precio_base"] >= q["real"]
            st.write("")
            st.markdown(f"**Qué pasó después.** El costo real de las cuatro semanas siguientes fue **{soles(q['real'])}** por {un}. "
                        f"El precio del motor {'lo cubrió' if ok_m else 'no lo cubrió'} "
                        f"({pct(100 * (q['precio'] / q['real'] - 1), 1, signo=True)}); "
                        f"la práctica actual {'lo cubrió' if ok_b else 'no lo cubrió'} "
                        f"({pct(100 * (q['precio_base'] / q['real'] - 1), 1, signo=True)}).")

        ficha = (f"SUSTENTO DE CONTINGENCIA\n"
                 f"Producto: {S.NOMBRE_CORTO[fam]}\nSemana de cotización: {q['fecha'].date()}\n"
                 f"Tipo de contrato: {tipo}\nPlazo: {meses} meses\nCantidad: {num(cantidad, 0)} {un}\n\n"
                 f"Costo esperado (P50): {soles(q['p50'])} por {un}\nRango P10-P90: {num(q['p10'])} a {num(q['p90'])}\n"
                 f"Riesgo de precio: {pct(100 * q['riesgo'])}\nFactor del contrato: {num(q['factor'], 1)}\n"
                 f"Raíz del plazo: {num(q['plazo'], 2)}\nTope de la gerencia: {pct(100 * tope, 0)}\n"
                 f"Contingencia aplicada: {pct(100 * q['contingencia'])}\n"
                 f"Precio de costo protegido: {soles(q['precio'])} por {un}\n"
                 f"Costo protegido de la partida: {soles(q['precio'] * cantidad, 0)}\n\n"
                 f"Referencia (práctica actual): último costo {soles(q['ultimo'])} + 5 % = {soles(q['precio_base'])}\n"
                 f"Modelo: {S.NOMBRE_MODELO[S.CAMPEON[fam]]} con rango calibrado (CQR + ACI al 80 %)\n"
                 f"Fuente de datos: {'historial del ERP' if fuente_real else 'demostración (costos internos simulados)'}\n")
        st.download_button("Descargar sustento de la cotización", ficha, file_name=f"sustento_{fam}_{q['fecha'].date()}.txt")
        if fam == "costo_mac":
            st.markdown('<p class="nota">En MAC el motor no predice el costo mejor que el último valor conocido: su aporte es '
                        'medir el riesgo. Si la contingencia sale alta, conviene negociar una cláusula de reajuste.</p>',
                        unsafe_allow_html=True)

# ----------------------------------------------------------------------------------------------
# 2. Desempeño
# ----------------------------------------------------------------------------------------------
with t_des:
    fam2 = st.radio("Producto", list(S.NOMBRE_CORTO), format_func=S.NOMBRE_CORTO.get, horizontal=True, key="fam_des")
    un2, camp = S.UNIDAD[fam2], S.CAMPEON[fam2]
    mt = met[(met.familia == fam2) & (met.periodo == "prueba")].set_index("modelo")
    cmp_ = S.comparacion(pred, fam2)
    n_prueba = int(mt.n.iloc[0])
    st.caption(f"Resultados en las {n_prueba} semanas de prueba, que no intervinieron en ninguna decisión del modelo.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Error práctica actual", pct(mt.MAPE["ultimo"], 2), "MAPE del último costo")
    with c2:
        kpi("Error del motor", pct(mt.MAPE[camp], 2), S.NOMBRE_MODELO[camp], destacado=True)
    with c3:
        p = mt.p_DM[camp]
        kpi("Mejora del error", pct(mt.mejora_MAE[camp], 1),
            "estadísticamente significativa" if p < 0.05 else f"no significativa (p = {num(p, 2)})")
    with c4:
        kpi("Cobertura del rango", pct(mt.cob_cal[camp], 1), "meta: 75 a 85 %")
    st.write("")

    a, b = st.columns([1, 1.35], gap="large")
    with a:
        st.markdown("**Error por modelo** · MAPE, menor es mejor")
        o = mt.sort_values("MAPE", ascending=False)
        col = [NARANJA if m == "ultimo" else AZUL if m == camp else GRIS for m in o.index]
        fig = go.Figure(go.Bar(x=o.MAPE, y=[S.NOMBRE_MODELO[m] for m in o.index], orientation="h", marker_color=col,
                               text=[num(v, 2) for v in o.MAPE], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{x:.2f} %<extra></extra>"))
        fig.update_xaxes(title="MAPE (%)", range=[0, o.MAPE.max() * 1.15])
        figura(fig, 380, fechas=False)
        st.markdown('<p class="nota">Naranja: práctica actual. Azul: modelo campeón, elegido en validación. '
                    'Las redes neuronales y el modelo fundacional se evalúan en el cuaderno.</p>', unsafe_allow_html=True)
    with b:
        st.markdown("**Predicción frente al costo real**")
        r = S.tabla(pred, fam2, camp); te = r[r.index >= INICIO_TEST]; ub = S.tabla(pred, fam2, "ultimo").loc[te.index]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(te.date) + list(te.date[::-1]), y=list(te.hi) + list(te.lo[::-1]), fill="toself",
                                 mode="lines", fillcolor="rgba(31,95,191,0.15)", line=dict(width=0), name="Rango P10–P90", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=te.date, y=te.y, name="Costo real", line=dict(color=TINTA, width=2.2)))
        fig.add_trace(go.Scatter(x=te.date, y=te.q50, name="Motor (P50)", line=dict(color=AZUL, width=2.2)))
        fig.add_trace(go.Scatter(x=te.date, y=ub.q50, name="Práctica actual", line=dict(color=NARANJA, width=1.8, dash="dash")))
        fig.update_yaxes(title=f"S/ por {un2}"); fig.update_layout(hovermode="x unified")
        figura(fig, 380)

    st.markdown("**¿Protege mejor el margen?** · Práctica actual frente al motor")
    tabla_c = pd.DataFrame({
        "Métrica": ["Contingencia media", "Semanas con el costo real cubierto", f"Déficit medio (S/ por {un2})",
                    "Contingencia para cubrir 90 % de las semanas"],
        "Práctica actual": [pct(cmp_["cont_media_base"]), pct(cmp_["cobertura_base"]), num(cmp_["deficit_medio_base"], 3), pct(cmp_["cont_base90"])],
        "Motor": [pct(cmp_["cont_media_motor"]), pct(cmp_["cobertura_motor"]), num(cmp_["deficit_medio_motor"], 3), pct(cmp_["cont_motor90"])]})
    st.dataframe(tabla_c, hide_index=True, width="stretch")
    v = cmp_["ventaja_precio_pp"]
    st.markdown(f"A igual protección, el motor permite ofertar **{num(abs(v), 1)} puntos "
                f"{'más barato' if v > 0 else 'más caro'}** que la práctica actual en {'MAC' if fam2 == 'costo_mac' else 'movimiento de tierras'}."
                + ("" if fuente_real else " Esta cifra supone que el plan de obra se cumple; con un plan que falla 20 %, "
                   "la ventaja en tierras baja a unos 4 puntos (ver informe, sección 13)."))

    with st.expander("Tabla completa de métricas"):
        cols = {"MAE": "MAE", "MAPE": "MAPE (%)", "mejora_MAE": "Mejora (%)", "IC95_inf": "IC 95 % inf.", "IC95_sup": "IC 95 % sup.",
                "p_DM": "p", "p_Holm": "p Holm", "cob_cal": "Cobertura (%)", "winkler": "Winkler"}
        tt = mt[[c for c in cols if c in mt]].rename(columns=cols)
        tt.index = [S.NOMBRE_MODELO[m] for m in tt.index]; tt.index.name = "Modelo"
        st.dataframe(tt.round(3), width="stretch")

# ----------------------------------------------------------------------------------------------
# 3. Monitoreo
# ----------------------------------------------------------------------------------------------
with t_mon:
    st.caption("Un modelo se degrada cuando cambia el mercado o la operación. Estas dos alertas avisan cuándo recalibrar o reentrenar.")
    estado = []
    mon = {}
    for f in S.NOMBRE_CORTO:
        r = S.tabla(pred, f, S.CAMPEON[f]); val = r[r.index < INICIO_TEST]
        mon[f] = drift.monitor_desempeno(r[r.index >= INICIO_TEST], errores_ref=(val.y - val.q50).values)
    X = features.construir(series, "costo_mac")
    cols_psi = {"r_brent4": "Variación del Brent", "r_fx4": "Variación del tipo de cambio", "r_ca4": "Variación del cemento asfáltico"}
    psi = drift.psi_movil(X, fin_ref=INICIO_TEST, columnas=list(cols_psi))
    psi.index = series.date.iloc[psi.index].values
    c1, c2, c3 = st.columns(3)
    for cc, f in zip((c1, c2), S.NOMBRE_CORTO):
        ult = mon[f].cob_movil.dropna()
        with cc:
            v = ult.iloc[-1] if len(ult) else np.nan
            kpi(f"Cobertura · {'MAC' if f == 'costo_mac' else 'Tierras'}", pct(v, 0), "en rango" if v >= 70 else "bajo el mínimo de 70 %: recalibrar")
    with c3:
        vmax = psi.iloc[-1].max()
        kpi("Cambio en los datos · PSI", num(vmax, 2), "PSI estable" if vmax < 0.25 else "PSI sobre 0,25: cambio importante")
    st.write("")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**¿El rango sigue acertando?** · cobertura móvil de 12 semanas")
        fig = go.Figure()
        for f, colr in zip(S.NOMBRE_CORTO, (AZUL, NARANJA)):
            m = mon[f].dropna(subset=["cob_movil"])
            fig.add_trace(go.Scatter(x=m.date, y=m.cob_movil, name=S.NOMBRE_CORTO[f], line=dict(color=colr, width=2.2)))
        fig.add_hline(y=70, line_dash="dot", line_color=SUAVE, annotation_text="mínimo 70 %", annotation_position="bottom right")
        fig.add_hrect(y0=75, y1=85, fillcolor="rgba(31,95,191,0.08)", line_width=0)
        fig.update_yaxes(title="Cobertura (%)", range=[30, 102]); fig.update_layout(hovermode="x unified")
        figura(fig, 340)
    with b:
        st.markdown("**¿Cambiaron los datos de entrada?** · PSI en ventanas de 26 semanas")
        fig = go.Figure()
        for (c, n), colr in zip(cols_psi.items(), (AZUL, NARANJA, VERDE)):
            fig.add_trace(go.Scatter(x=psi.index, y=psi[c], name=n, line=dict(color=colr, width=2.2)))
        fig.add_hline(y=0.25, line_dash="dot", line_color=SUAVE, annotation_text="alerta 0,25", annotation_position="top left")
        fig.update_yaxes(title="PSI"); fig.update_layout(hovermode="x unified")
        figura(fig, 340)
    st.markdown('<p class="nota">Regla de operación: ante una alerta se recalibra el rango; si la alerta persiste cuatro semanas, '
                'se reentrena el modelo y se vuelve a evaluar al campeón frente a los retadores.</p>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------------------------
# 4. Análisis económico
# ----------------------------------------------------------------------------------------------
with t_eco:
    cont = {f: S.comparacion(pred, f) for f in S.NOMBRE_CORTO}
    st.caption("Empresa con proyecto frente a empresa sin proyecto, a tres años. Mueva los supuestos para ver cuánto cambia el resultado.")
    izq, der = st.columns([1, 2.2], gap="large")
    with izq:
        st.subheader("Supuestos")
        D = E.SUPUESTOS
        vol = st.slider("Tierras cotizadas al año (millones de m³)", 1.0, 5.0, D["vol_t_cot"] / 1e6, 0.25)
        exito = st.slider("Tasa de adjudicación (%)", 10, 50, int(100 * D["tasa_exito"]))
        sa = st.slider("Contratos a suma alzada (%)", 10, 80, int(100 * D["share_sa"]))
        margen = st.slider("Margen de contribución (%)", 4, 20, int(100 * D["margen_contrib"]))
        captura = st.slider("Ventaja trasladada al precio (%)", 0, 60, int(100 * D["captura"]),
                            help="Con 0 % no se cuenta ninguna obra adicional ganada: es el escenario prudente.")
        horas = st.slider("Horas ahorradas por cotización", 0.0, 5.0, float(D["horas_ahorro"]), 0.5)
        tasa = st.slider("Tasa de descuento (%)", 8, 20, int(100 * D["tasa"]))
    s = dict(D, vol_t_cot=vol * 1e6, tasa_exito=exito / 100, share_sa=sa / 100, margen_contrib=margen / 100,
             captura=captura / 100, horas_ahorro=horas, tasa=tasa / 100)
    ev = E.evaluar(cont, s)
    with der:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi("VAN a 3 años", soles(ev["VAN"], 0), "rentable" if ev["VAN"] > 0 else "no recupera la inversión", destacado=True)
        with c2:
            kpi("Beneficio / costo", num(ev["BC"], 2), "mayor a 1 conviene")
        with c3:
            kpi("TIR", pct(100 * ev["TIR"], 0) if not np.isnan(ev["TIR"]) else "—", f"frente a {tasa} % de descuento")
        with c4:
            rec = ev["recuperacion"]
            kpi("Recuperación", f"{num(rec, 1)} años" if not np.isnan(rec) else "más de 3 años", f"inversión: {soles(ev['inversion'], 0)}")
        st.write("")
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("**De dónde sale el beneficio** · S/ por año, en régimen")
            nb = {"perdidas_evitadas": "Pérdidas evitadas a suma alzada", "competitividad": "Más obras ganadas", "horas": "Horas de analista"}
            fig = go.Figure(go.Bar(x=[ev["beneficios"][k] for k in nb], y=list(nb.values()), orientation="h", marker_color=AZUL,
                                   text=[num(ev["beneficios"][k], 0) for k in nb], textposition="outside", cliponaxis=False,
                                   hovertemplate="%{y}: S/ %{x:,.0f}<extra></extra>"))
            fig.update_yaxes(autorange="reversed"); fig.update_xaxes(range=[0, max(ev["beneficios"].values()) * 1.7 + 1], title="S/ por año", showticklabels=False)
            figura(fig, 270, fechas=False)
        with b:
            st.markdown("**Flujo de caja** · S/")
            fl = ev["flujo"]
            fig = go.Figure(go.Bar(x=[f"Año {i}" for i in range(len(fl))], y=fl, marker_color=[NARANJA if v < 0 else AZUL for v in fl],
                                   text=[num(v, 0) for v in fl], textposition="outside", cliponaxis=False,
                                   hovertemplate="%{x}: S/ %{y:,.0f}<extra></extra>"))
            fig.update_yaxes(range=[min(fl) * 1.35, max(max(fl), 1) * 1.25], showticklabels=False)
            figura(fig, 270, fechas=False)
        st.markdown("**Riesgo** · VAN en 4 000 combinaciones de supuestos (simulación de Monte Carlo con los rangos del informe)")
        mc, vans = montecarlo(tuple(sorted(cont["costo_tierras"].items())), tuple(sorted(cont["costo_mac"].items())))
        fig = go.Figure(go.Histogram(x=np.asarray(vans) / 1000, nbinsx=50, marker_color=AZUL, opacity=0.85,
                                     hovertemplate="VAN %{x} mil: %{y} casos<extra></extra>"))
        fig.add_vline(x=0, line_color=NARANJA, line_width=2)
        fig.add_vline(x=ev["VAN"] / 1000, line_color=TINTA, line_dash="dot", annotation_text="sus supuestos", annotation_position="top")
        fig.update_xaxes(title="VAN (miles de S/)"); fig.update_yaxes(title="Casos"); fig.update_layout(bargap=0.04)
        figura(fig, 280, fechas=False)
        st.markdown(f"El VAN es positivo en **{pct(mc['prob_van_pos'], 0)}** de los casos; la mediana es **{soles(mc['p50'], 0)}**. "
                    + ("" if fuente_real else "Estas cifras parten de costos internos simulados y de un plan de obra que se cumple: "
                       "son un techo. El informe estima un VAN de S/\u00a060\u202f000 a S/\u00a0157\u202f000 con supuestos prudentes."))

# ----------------------------------------------------------------------------------------------
# 5. Datos y método
# ----------------------------------------------------------------------------------------------
with t_dat:
    a, b = st.columns([1.25, 1], gap="large")
    with a:
        st.subheader("Qué es real y qué es simulado")
        if fuente_real:
            st.success(f"Está viendo resultados calculados con el historial del ERP ({len(series)} semanas).")
        else:
            st.markdown("El precio del petróleo Brent y el tipo de cambio son **datos reales** (julio de 2023 a marzo de 2026). "
                        "Los costos internos son **simulados** con la estructura de costos de la empresa. "
                        "El tablero demuestra que el método funciona; las cifras deben recalcularse con el historial real.")
        fig = go.Figure()
        v = st.radio("Serie", ["costo_mac", "costo_tierras", "brent", "fx"], horizontal=True,
                     format_func={"costo_mac": "Costo MAC", "costo_tierras": "Costo tierras", "brent": "Brent", "fx": "Tipo de cambio"}.get)
        fig.add_trace(go.Scatter(x=series.date, y=series[v], line=dict(color=AZUL if v in ("brent", "fx") else NARANJA, width=2), name=v))
        fig.add_vrect(x0=series.date.iloc[min(INICIO_TEST, len(series) - 1)], x1=series.date.iloc[-1], fillcolor="rgba(20,23,31,0.05)",
                      line_width=0, annotation_text="periodo de prueba", annotation_position="top left")
        fig.update_layout(showlegend=False)
        figura(fig, 300)
        st.subheader("Cómo funciona, en cuatro pasos")
        st.markdown("""
1. **Aprende solo del pasado.** Cada semana el modelo se reentrena con lo ocurrido hasta ese día y predice el costo medio de las cuatro semanas siguientes.
2. **Entrega un rango, no un número.** P10 (optimista), P50 (esperado) y P90 (pesimista).
3. **Corrige el rango.** Si el rango falla más de lo prometido, se ensancha; si acierta de sobra, se angosta (calibración conformal adaptativa).
4. **Convierte el rango en contingencia.** Riesgo de precio × factor del contrato × raíz del plazo, con piso de 1 % y tope de la gerencia.
""")
        st.dataframe(pd.DataFrame({"Tipo de contrato": TIPOS, "Factor": [C.FACTOR_CONTRATO[t] for t in TIPOS]}),
                     hide_index=True, width="stretch")
    with b:
        st.subheader("Usar el historial del ERP")
        st.markdown("Suba un archivo CSV con una fila por semana y estas columnas. El tablero repite la evaluación completa "
                    "con sus datos (toma cerca de un minuto).")
        st.code("fecha_semana, costo_mac, costo_tierras,\nprecio_ca, prod_mac, diesel,\ndist_acarreo, utilizacion, lluvia", language=None)
        d1, d2 = st.columns(2)
        d1.download_button("Plantilla vacía", (RAIZ / "data/plantilla_erp.csv").read_bytes(), "plantilla_erp.csv", width="stretch")
        d2.download_button("Archivo de ejemplo", (RAIZ / "data/ejemplo_erp.csv").read_bytes(), "ejemplo_erp.csv", width="stretch")
        archivo = st.file_uploader("Historial del ERP (.csv)", type="csv")
        if archivo is not None and st.button("Recalcular con mis datos", type="primary", width="stretch"):
            barra = st.progress(0.0, "Leyendo el archivo…")
            try:
                st.session_state["erp"] = S.recalcular(archivo, progreso=lambda p, t: barra.progress(p, t))
                st.rerun()
            except Exception as e:  # el mensaje indica qué corregir en el archivo
                barra.empty()
                st.error(f"No se pudo procesar el archivo: {e}")
        if fuente_real:
            st.markdown("**Control de calidad del archivo**")
            st.json({k: v for k, v in calidad.items()}, expanded=False)
            if st.button("Volver a los datos de demostración", width="stretch"):
                del st.session_state["erp"]
                st.rerun()
        st.markdown(f'<p class="nota">Se necesitan al menos {S.MIN_SEMANAS} semanas. Los datos se procesan en esta sesión y no se guardan. '
                    'El archivo de ejemplo reproduce los datos de demostración.</p>', unsafe_allow_html=True)

st.markdown('<p class="nota" style="margin-top:2rem">Proyecto Productivo IIIA · Instituto Continental · Ronal Ronaldo Ccasa Lipe. '
            'El motor apoya al analista y no fija precios sin revisión humana.</p>', unsafe_allow_html=True)
