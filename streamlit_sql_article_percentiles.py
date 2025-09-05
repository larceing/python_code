# streamlit_sql_article_percentiles.py
# ------------------------------------------------------------
# Dashboard Streamlit para analizar un ARTÍCULO concreto en SQL Server (sin SQLAlchemy)
# - Conexión directa con pyodbc usando cadena embebida
# - Filtra por fecha y artículo
# - Calcula percentiles por cliente (unidades / importe)
# - Segmenta clientes por percentiles
# - KPIs + tabla percentiles + clientes segmentados + histogram/boxplot + Top N
# - Descarga a CSV
#
# Ejecuta con: streamlit run streamlit_sql_article_percentiles.py
# Requisitos: pip install streamlit pandas numpy pyodbc plotly
# ------------------------------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import date
from typing import List

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pyodbc

# ================================
# Configuración de página
# ================================
st.set_page_config(
    page_title="Artículo – Percentiles por Cliente",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; padding-bottom: 1rem; }
      div[data-testid="stMetricValue"] { font-size: 1.6rem; }
      .note { color: #6b7280; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ================================
# Conexión SQL Server (simple, como tu helper)
# ================================
import pyodbc

# Config mínima en código (ajústala a tu entorno)
CONFIG = {
    "conexion_sql": {
        "host": "*****",        # o r"PC\SQLEXPRESS"
        "base_datos": "****",
        "usuario": "*****",
        "clave": "*****",
    }
}

def construir_conexion_sql(config: dict) -> str:
    sql_cfg = config.get("conexion_sql", {})
    conn_str = (
        f"DRIVER={{SQL Server}};"
        f"SERVER={sql_cfg.get('host', 'localhost')};"
        f"DATABASE={sql_cfg.get('base_datos', 'SG')};"
        f"UID={sql_cfg.get('usuario', 'user')};"
        f"PWD={sql_cfg.get('clave', '')};"
    )
    return conn_str

@st.cache_resource(show_spinner=False)
def get_connection():
    try:
        return pyodbc.connect(construir_conexion_sql(CONFIG))
    except Exception as e:
        st.error(f"❌ Error de conexión ODBC: {e}")
        st.stop()

# ================================
# Parámetros de usuario (UI)
# ================================
st.sidebar.header("🔎 Parámetros")

# Estado para forzar recarga y log
if "__nonce" not in st.session_state:
    st.session_state.__nonce = 0
if "__log" not in st.session_state:
    st.session_state.__log = []

def log(msg: str):
    from datetime import datetime as _dt
    ts = _dt.now().strftime("%H:%M:%S")
    st.session_state.__log.append(f"[{ts}] {msg}")

# Controles
start_date = st.sidebar.date_input("Fecha desde", value=date(2024, 1, 1))
article_code = st.sidebar.text_input("Código de artículo", value="8675").strip()
match_mode = st.sidebar.selectbox("Filtro de artículo", ["Contiene", "Igual exacto"], index=0)
measure = st.sidebar.selectbox("Métrica a analizar", ["unidades", "importe"], index=0)

col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("🔄 Forzar recarga"):
    st.session_state.__nonce += 1
if col_btn2.button("🧪 Probar conexión"):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        log("Conexión OK")
        st.sidebar.success("Conexión OK")
    except Exception as e:
        log(f"Conexión FALLÓ: {e}")
        st.sidebar.error("Conexión falló, revisa cadena ODBC")

# Percentiles / segmentos
pct_list_default = [5, 10, 50, 75, 90]
pct_list: List[int] = st.sidebar.multiselect(
    "Percentiles a calcular",
    options=[5, 10, 25, 50, 75, 90, 95, 99],
    default=pct_list_default,
)

seg_edges_default = [0, 10, 50, 75, 90, 100]
seg_edges_str = st.sidebar.text_input(
    "Segmentos por percentil (coma)",
    value=",".join(map(str, seg_edges_default)),
    help="0,10,50,75,90,100 → (Muy bajos, Bajos, Medios, Altos, Muy altos)",
)

def parse_edges(s: str) -> List[int]:
    try:
        arr = [int(x.strip()) for x in s.split(",") if x.strip()]
        arr = sorted(list(set(arr)))
        if arr[0] != 0:
            arr = [0] + arr
        if arr[-1] != 100:
            arr = arr + [100]
        return arr
    except Exception:
        return seg_edges_default

seg_edges = parse_edges(seg_edges_str)

# ================================
# SQL builders (no mutar templates globales)
# ================================

def build_where_article(codart: str, mode: str) -> str:
    codart = codart.replace("'", "''")
    field = "LTRIM(RTRIM(CAST(LINE.CODART AS VARCHAR(50))))"
    if not codart:
        return ""
    if mode == "Igual exacto":
        return f" AND {field} = '{codart}'\n"
    else:
        return f" AND {field} LIKE '%{codart}%'\n"

BASE_SQL_PREFIX = """
SELECT 
    CABE.CODCLI,
    CLI.NOMCLI,
    CABE.CODREP, 
    CABE.NUMDOC, 
    LINE.CODART, 
    ART.CAR1 AS FAMILIA, 
    LINE.UNIDADES,
    LINE.BASE,
    CABE.FECHA
FROM CABEOFEV  CABE 
LEFT JOIN REPRESEN REP ON CABE.CODREP = REP.CODREP
LEFT JOIN CLIENTES CLI ON CABE.CODCLI = CLI.CODCLI
LEFT JOIN LINEOFER LINE ON LINE.IDOFEV = CABE.IDOFEV 
LEFT JOIN ARTICULO ART ON ART.CODART = LINE.CODART
WHERE CONVERT(DATE, CABE.FECHA, 103) >= '{fecha_desde}'
  AND CABE.SERIE IS NULL 
  AND CABE.TOTDOC > 0
"""

MATCHES_SQL_PREFIX = """
SELECT DISTINCT TOP 50 LTRIM(RTRIM(CAST(LINE.CODART AS VARCHAR(50)))) AS CODART
FROM CABEOFEV  CABE 
LEFT JOIN LINEOFER LINE ON LINE.IDOFEV = CABE.IDOFEV 
WHERE CONVERT(DATE, CABE.FECHA, 103) >= '{fecha_desde}'
"""

@st.cache_data(show_spinner=False)
def load_article_df(start_date: date, codart: str, mode: str, nonce: int) -> pd.DataFrame:
    fecha_desde = start_date.strftime("%d/%m/%Y")
    sql = BASE_SQL_PREFIX.format(fecha_desde=fecha_desde) + build_where_article(codart, mode)
    log(f"SQL main preparada (len={len(sql)}): {sql[:200]}...")
    conn = get_connection()
    try:
        df = pd.read_sql(sql, conn)
        log(f"Filas obtenidas: {len(df)}")
    except Exception as e:
        log(f"Error ejecutando consulta principal: {e}")
        st.error(f"❌ Error ejecutando la consulta: {e}")
        st.stop()
    if df.empty:
        return df
    df = df.rename(columns={
        "CODCLI": "codcli",
        "NOMCLI": "nomcli",
        "CODREP": "codrep",
        "NUMDOC": "numdoc",
        "CODART": "codart",
        "FAMILIA": "familia",
        "UNIDADES": "unidades",
        "BASE": "importe",
    })
    for c in ["unidades", "importe"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

@st.cache_data(show_spinner=False)
def find_matches(start_date: date, codart: str, mode: str, nonce: int) -> pd.DataFrame:
    fecha_desde = start_date.strftime("%d/%m/%Y")
    sql = MATCHES_SQL_PREFIX.format(fecha_desde=fecha_desde) + build_where_article(codart, mode)
    log(f"SQL matches preparada: {sql}")
    conn = get_connection()
    try:
        d = pd.read_sql(sql, conn)
        log(f"Coincidencias encontradas: {len(d)}")
        return d
    except Exception as e:
        log(f"Error ejecutando consulta matches: {e}")
        return pd.DataFrame()

# ================================
# Carga de datos
# ================================
with st.spinner("Cargando datos del artículo…"):
    df_raw = load_article_df(start_date, article_code, match_mode, st.session_state.__nonce)

# Debug: SQL efectiva y coincidencias
with st.expander("🔧 Debug / SQL efectiva", expanded=False):
    fecha_desde = start_date.strftime("%d/%m/%Y")
    sql_preview = BASE_SQL_PREFIX.format(fecha_desde=fecha_desde) + build_where_article(article_code, match_mode)
    st.code(sql_preview, language="sql")
    m = find_matches(start_date, article_code, match_mode, st.session_state.__nonce)
    if not m.empty:
        st.write("Coincidencias de CODART (TOP 50):", m)
    if st.session_state.__log:
        st.text("\n".join(st.session_state.__log))

st.title("📦 Análisis por Artículo – Percentiles por Cliente")
st.caption("Fuente: SQL Server (pyodbc) • Segmentación por percentiles • Métrica seleccionable")

if df_raw.empty:
    st.warning("No hay datos para el artículo o rango indicado.")
    st.stop()

# ================================
# Agregado por cliente
# ================================
agg = (
    df_raw.groupby(["codcli", "nomcli"], as_index=False)
          .agg(total_unidades=("unidades", "sum"), total_importe=("importe", "sum"))
)
agg["total_unidades"] = agg["total_unidades"].astype(float)
agg["total_importe"] = agg["total_importe"].astype(float)

metric_col = "total_unidades" if measure == "unidades" else "total_importe"
metric_label = "Unidades" if measure == "unidades" else "Importe (€)"
vals = agg[metric_col].to_numpy()

# Máx / Mín unidades en un solo pedido (suma de líneas del artículo por NUMDOC)
per_pedido = df_raw.groupby("numdoc", dropna=False)["unidades"].sum() if not df_raw.empty else pd.Series(dtype=float)
max_un_pedido = int(per_pedido.max()) if not per_pedido.empty else 0
min_un_pedido = int(per_pedido.min()) if not per_pedido.empty else 0

def _fmt_int(x: int) -> str:
    return f"{x:,}".replace(",", ".")


# --- Inactivos >90 días (por artículo) ---
def _fmt_int(x: int) -> str:
    return f"{x:,}".replace(",", ".")

ref_date = pd.Timestamp.today().normalize()
fechas = pd.to_datetime(df_raw["FECHA"], errors="coerce")

# última compra del ARTÍCULO por cliente (en el rango actual)
last_by_client = df_raw.assign(_fecha=fechas).groupby("codcli")["_fecha"].max()

inactive_mask = (ref_date - last_by_client) > pd.Timedelta(days=90)
inactive_clients = int(inactive_mask.sum())

base_clientes_art = int(last_by_client.shape[0])  # clientes con alguna compra del artículo
pct_inactive = (inactive_clients / base_clientes_art * 100) if base_clientes_art else 0.0

# ================================
# KPIs + Percentiles
# ================================
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
col1.metric("Clientes con compras", f"{len(agg):,}".replace(",", "."))
col2.metric("Media", f"{np.nanmean(vals):,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
col3.metric("Mediana", f"{np.nanmedian(vals):,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
col4.metric("Suma", f"{np.nansum(vals):,.0f}".replace(",", "X").replace(".", ",").replace("X", "."))
col5.metric("Inactivos >90d (clientes)", _fmt_int(inactive_clients), f"{pct_inactive:.1f}%")
col6.metric("Máx unidades (1 pedido)", _fmt_int(max_un_pedido))
col7.metric("Mín unidades (1 pedido)", _fmt_int(min_un_pedido))

if len(pct_list) == 0:
    pct_list = pct_list_default
pcts = np.percentile(vals, pct_list) if len(vals) else np.array([0.0]*len(pct_list))
ptbl = pd.DataFrame({
    "Percentil": [f"p{p}" for p in pct_list],
    metric_label: [round(x, 2) for x in pcts]
})

st.subheader("📈 Percentiles del artículo por cliente")
colP1, colP2 = st.columns(2)

with colP1:
    st.dataframe(ptbl, use_container_width=True, hide_index=True)

with colP2:
    stats = pd.Series(vals, dtype=float)
    stats_df = pd.DataFrame({
        "Métrica": ["Clientes","Media","Mediana","Desv. típica","Mín","P5","P10","P75","P90","Máx"],
        metric_label: [
            int(len(stats)),
            round(float(stats.mean()) if len(stats) else 0, 2),
            round(float(stats.median()) if len(stats) else 0, 2),
            round(float(stats.std(ddof=1)) if len(stats) > 1 else 0, 2),
            round(float(stats.min()) if len(stats) else 0, 2),
            round(float(np.percentile(stats, 5)) if len(stats) else 0, 2),
            round(float(np.percentile(stats,10)) if len(stats) else 0, 2),
            round(float(np.percentile(stats,75)) if len(stats) else 0, 2),
            round(float(np.percentile(stats,90)) if len(stats) else 0, 2),
            round(float(stats.max()) if len(stats) else 0, 2),
        ]
    })
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

st.markdown("""
<style>
/* KPIs (st.metric) */
div[data-testid="stMetricValue"] { font-size: 2.2rem !important; }   /* valor grande */
div[data-testid="stMetricLabel"] { font-size: 1.05rem !important; }  /* etiqueta */
div[data-testid="stMetricDelta"] { font-size: 1.0rem !important; }   /* delta (+/-) */

/* Tablas (st.dataframe) – p.ej., percentiles y otras */
div[data-testid="stDataFrame"] div[role="gridcell"] { font-size: 16px !important; }
div[data-testid="stDataFrame"] div[role="columnheader"] {
  font-size: 16px !important;
  font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ================================
# Segmentación por percentiles (robusta)
# ================================
agg_sorted = agg.sort_values(metric_col).reset_index(drop=True)

# Bordes de bins a partir de percentiles; elimina duplicados (empates)
edges = np.unique(np.percentile(vals, seg_edges)) if len(vals) else np.array([])

if len(edges) < 2:
    # Sin variación suficiente → un único grupo
    agg_sorted["segmento"] = "Todos"
else:
    # Deja que pandas genere intervalos y luego los renombramos
    seg = pd.cut(
        agg_sorted[metric_col],
        bins=edges,
        include_lowest=True,
        right=True,
    )
    n_bins = len(edges) - 1
    base_labels = ["Muy bajos", "Bajos", "Medios", "Altos", "Muy altos"]
    labels = base_labels[:n_bins] if len(base_labels) >= n_bins else [f"Seg {i+1}" for i in range(n_bins)]
    agg_sorted["segmento"] = seg.cat.rename_categories(labels)

st.subheader("🧩 Clientes segmentados por percentiles")
seg_cols = ["codcli", "nomcli", "total_unidades", "total_importe", "segmento"]
st.dataframe(agg_sorted[seg_cols], use_container_width=True, hide_index=True)

seg_dist = (
    agg_sorted.groupby("segmento", as_index=False)
              .agg(clientes=("codcli", "count"),
                   suma_metric=(metric_col, "sum"),
                   media_metric=(metric_col, "mean"))
)

cA, cB = st.columns(2)
with cA:
    fig_seg = px.bar(seg_dist, x="segmento", y="clientes", text_auto=True,
                     title="Nº de clientes por segmento")
    fig_seg.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_seg, use_container_width=True)

with cB:
    fig_seg2 = px.bar(seg_dist, x="segmento", y="suma_metric", text_auto=".2s",
                      title=f"Suma de {metric_label} por segmento")
    fig_seg2.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_seg2, use_container_width=True)

# ================================
# 📍 Dispersión: Clientes (Unidades vs Importe)
# ================================
st.subheader("📍 Dispersión: Clientes (Unidades vs Importe)")
c1, c2, c3 = st.columns(3)
color_by = c1.selectbox("Color por", ["segmento", "ninguno"], index=0)
size_by  = c2.selectbox("Tamaño por", ["total_importe", "total_unidades"], index=0)
use_log  = c3.checkbox("Escala log (X, Y)", value=True)

# ---- datos base (arrays, para evitar conflictos de metaclases) ----
x = agg_sorted["total_unidades"].to_numpy(dtype=float)
y = agg_sorted["total_importe"].to_numpy(dtype=float)
s = agg_sorted[size_by].to_numpy(dtype=float)
seg = agg_sorted["segmento"].astype(str).fillna("NA").to_numpy()
cod = agg_sorted["codcli"].astype(str).to_numpy()
nom = agg_sorted["nomcli"].astype(str).to_numpy()

# filtra si se usa log (plotly no admite 0 o negativos en escala log)
mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(s)
if use_log:
    mask &= (x > 0) & (y > 0)

x, y, s, seg, cod, nom = x[mask], y[mask], s[mask], seg[mask], cod[mask], nom[mask]

# tamaño visual 6–24 px
def _scale_size(arr, mi=6, ma=24):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return np.array([], dtype=float)
    a, b = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(a) or not np.isfinite(b) or a == b:
        return np.full(arr.shape, (mi + ma) / 2.0, dtype=float)
    return mi + (arr - a) * (ma - mi) / (b - a)

size_px = _scale_size(s)

fig_sc = go.Figure()

if color_by == "ninguno":
    custom = np.stack([cod, nom], axis=-1)
    fig_sc.add_trace(go.Scattergl(
        x=x, y=y,
        mode="markers",
        marker=dict(size=size_px),
        customdata=custom,
        hovertemplate=(
            "Cliente: %{customdata[1]} (COD: %{customdata[0]})<br>"
            "Unidades: %{x:,.0f}<br>"
            "Importe: %{y:,.2f}<extra></extra>"
        ),
        name="Clientes"
    ))
else:
    # traza por segmento (colores separados)
    cats = pd.unique(seg)
    for cval in cats:
        sel = (seg == cval)
        custom = np.stack([cod[sel], nom[sel]], axis=-1)
        fig_sc.add_trace(go.Scattergl(
            x=x[sel], y=y[sel],
            mode="markers",
            marker=dict(size=size_px[sel]),
            customdata=custom,
            hovertemplate=(
                "Cliente: %{customdata[1]} (COD: %{customdata[0]})<br>"
                "Unidades: %{x:,.0f}<br>"
                "Importe: %{y:,.2f}<extra></extra>"
            ),
            name=str(cval)
        ))

if use_log:
    fig_sc.update_xaxes(type="log")
    fig_sc.update_yaxes(type="log")

fig_sc.update_layout(
    title="Clientes: Unidades vs Importe",
    xaxis_title="Unidades",
    yaxis_title="Importe (€)",
    margin=dict(l=10, r=10, t=50, b=10),
    legend_title_text="Segmento" if color_by != "ninguno" else None,
)

st.plotly_chart(fig_sc, use_container_width=True)

# ================================
# Gráficos de distribución
# ================================
st.subheader("📊 Distribuciones")
colG1, colG2 = st.columns(2)

with colG1:
    fig_hist = px.histogram(agg_sorted, x=metric_col, nbins=30,
                            title=f"Histograma de {metric_label} por cliente")
    fig_hist.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_hist, use_container_width=True)

with colG2:
    fig_box = px.box(agg_sorted, y=metric_col, points="outliers",
                     title=f"Boxplot de {metric_label} por cliente")
    fig_box.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_box, use_container_width=True)

# Top N clientes
st.subheader("🏆 Top clientes por métrica")
TopN = st.slider("Top N", min_value=5, max_value=50, value=15, step=5)
ranked = agg_sorted.sort_values(metric_col, ascending=False).head(TopN)
show_cols = ["codcli", "nomcli", "total_unidades", "total_importe", "segmento"]
st.dataframe(ranked[show_cols], use_container_width=True, hide_index=True)

# ================================
# Descarga CSV
# ================================
colD1, colD2 = st.columns(2)
with colD1:
    csv1 = agg_sorted.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar clientes segmentados (CSV)", data=csv1, file_name=f"clientes_segmentados_{article_code}.csv", mime="text/csv")
with colD2:
    csv2 = ptbl.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar percentiles (CSV)", data=csv2, file_name=f"percentiles_{article_code}.csv", mime="text/csv")

st.markdown(
    """
    <div class='note'>
    ℹ️ Tips:
      <ul>
        <li>Si usas <b>instancia nombrada</b>, pon <code>SERVER=r"PC\\SQLEXPRESS"</code> y deja <code>PORT=""</code>.</li>
        <li>Si tienes instalado <b>ODBC Driver 18</b>, cambia <code>DRIVER = "ODBC Driver 18 for SQL Server"</code>.</li>
        <li>Para <b>Windows Auth</b>, usa <code>UID=</code>/<code>PWD=</code> vacíos y añade <code>Trusted_Connection=Yes;</code> a la cadena.</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
