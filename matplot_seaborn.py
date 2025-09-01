# matplot_seaborn.py
# ------------------------------------------------------------
# Dashboard "estilo Power BI" en un solo archivo .py
# - Datos sintéticos de ventas (sin fuentes externas)
# - Filtros por fecha, región, producto y canal
# - KPIs (Ventas, Pedidos, Unidades, AOV, Margen, Variación vs periodo anterior)
# - Gráficos: serie temporal, barras apiladas, treemap, heatmap, Top N tabla
# - Descarga de datos filtrados a CSV
# ------------------------------------------------------------
# Ejecuta con: streamlit run matplot_seaborn.py
# Requisitos: streamlit, plotly, pandas, numpy, python-dateutil
# ------------------------------------------------------------

import numpy as np
import pandas as pd
from datetime import timedelta, date
from dateutil.relativedelta import relativedelta
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -------------------------------
# Configuración de página
# -------------------------------
st.set_page_config(
    page_title="Sales Dashboard (Power BI style)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos sutiles
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    div[data-testid="stMetricValue"] { font-size: 1.75rem; }
    .small-note { color: #6b7280; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Generación de datos sintéticos
# -------------------------------
@st.cache_data(show_spinner=False)
def generate_data(seed: int = 42):
    rng = np.random.default_rng(seed)

    # Catálogos
    regions = ["Norte", "Sur", "Este", "Oeste"]
    channels = ["Retail", "Online", "Distribuidor"]
    categories = ["Electrónica", "Hogar", "Moda", "Deporte"]
    products = {
        "Electrónica": ["Auriculares X1", "Smartwatch Z", "Tablet Pro", "Cámara HD"],
        "Hogar": ["Cafetera Barista", "Aspiradora Max", "Lámpara LED", "Silla Ergo"],
        "Moda": ["Zapatilla Run", "Chaqueta Wind", "Mochila City", "Gafas Sun"],
        "Deporte": ["Bici Urban", "Casco Pro", "Guantes Grip", "Balón Elite"],
    }

    # Fechas (últimos 24 meses, día a día)
    end = date.today()
    start = (end - relativedelta(months=24))
    dates = pd.date_range(start, end, freq="D")

    # Simulación de "clientes"
    customers = [f"CUST-{i:04d}" for i in range(1, 801)]

    rows = []
    for dt in dates:
        # volumen diario base
        base_orders = rng.integers(120, 250)
        # estacionalidad semanal (pico mitad de semana)
        weekday_factor = 1.0 + (0.2 if dt.weekday() in [2, 3] else -0.05 if dt.weekday() in [5, 6] else 0.0)
        # tendencia suave al alza
        months_from_start = (dt.to_pydatetime().date() - start).days / 30
        trend = 1.0 + months_from_start * 0.01  # +1% por mes aprox

        n_orders = int(base_orders * weekday_factor * trend)

        for _ in range(n_orders):
            cat = rng.choice(categories, p=[0.32, 0.28, 0.22, 0.18])
            prod = rng.choice(products[cat])
            region = rng.choice(regions)
            channel = rng.choice(channels, p=[0.45, 0.4, 0.15])
            customer = rng.choice(customers)

            # precio base por categoría (evita negativos extremos)
            raw_base = {
                "Electrónica": rng.normal(180, 40),
                "Hogar":       rng.normal(90, 20),
                "Moda":        rng.normal(60, 15),
                "Deporte":     rng.normal(140, 35),
            }[cat]

            # clamp del precio base para evitar valores patológicos
            base_price = max(10.0, float(raw_base))  # mínimo 10€

            units = max(1, int(abs(rng.normal(1.6, 0.8))))

            # descuento en [0, 0.35]
            discount = float(np.clip(rng.normal(0.06, 0.05), 0.0, 0.35))

            # precio final (aplicando descuento) con suelo de 5€
            price = max(5.0, base_price * (1 - discount))

            # ---- COSTE ----
            # desviación positiva y con suelo (evita scale < 0)
            cost_sd = max(1.0, abs(base_price) * 0.05)
            cost_unit = max(2.0, base_price * 0.7 + rng.normal(0.0, cost_sd))

            revenue = price * units
            cost    = cost_unit * units
            profit  = revenue - cost
            margin_pct = profit / revenue if revenue > 0 else 0.0

            rows.append({
                "date": dt,
                "year": dt.year,
                "month": dt.month,
                "month_name": dt.strftime("%b"),
                "quarter": f"Q{((dt.month-1)//3)+1}",
                "week": dt.isocalendar().week,
                "region": region,
                "channel": channel,
                "category": cat,
                "product": prod,
                "customer": customer,
                "units": units,
                "price": round(price, 2),
                "revenue": round(revenue, 2),
                "cost": round(cost, 2),
                "profit": round(profit, 2),
                "margin_pct": float(margin_pct),
                "discount_pct": float(discount),
            })

    df = pd.DataFrame(rows)
    # ordenar meses para heatmap
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    df["month_abbr"] = pd.Categorical(df["month_name"], categories=month_order, ordered=True)
    return df

df = generate_data()

# -------------------------------
# Utilidades
# -------------------------------
def format_currency(x: float, symbol="€"):
    try:
        return f"{symbol}{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return f"{symbol}{x}"

def period_compare(current_start, current_end):
    """Devuelve el rango anterior con la misma duración."""
    delta = current_end - current_start
    prev_end = current_start - timedelta(days=1)
    prev_start = prev_end - delta
    return prev_start, prev_end

# -------------------------------
# Sidebar: Filtros
# -------------------------------
st.sidebar.header("🔎 Filtros")

min_date, max_date = df["date"].min().date(), df["date"].max().date()
default_start = max_date - relativedelta(months=3)
date_range = st.sidebar.date_input(
    "Rango de fechas",
    value=(default_start, max_date),
    min_value=min_date,
    max_value=max_date
)
if isinstance(date_range, tuple):
    f_start, f_end = date_range
else:
    f_start, f_end = min_date, max_date

regions = sorted(df["region"].unique())
channels = sorted(df["channel"].unique())
categories = sorted(df["category"].unique())
products = sorted(df["product"].unique())

f_region = st.sidebar.multiselect("Región", options=regions, default=regions)
f_channel = st.sidebar.multiselect("Canal", options=channels, default=channels)
f_category = st.sidebar.multiselect("Categoría", options=categories, default=categories)
f_product = st.sidebar.multiselect("Producto", options=products, default=[])

granularity = st.sidebar.selectbox("Granularidad de fecha", ["Día", "Semana", "Mes"], index=1)
topn = st.sidebar.slider("Top N productos (tabla)", min_value=5, max_value=25, value=10, step=5)

st.sidebar.markdown("---")
st.sidebar.caption("💡 Consejo: combina filtros como en Power BI para contar diferentes historias con el mismo dato.")

# -------------------------------
# Aplicar filtros
# -------------------------------
mask = (
    (df["date"].dt.date >= f_start) &
    (df["date"].dt.date <= f_end) &
    (df["region"].isin(f_region)) &
    (df["channel"].isin(f_channel)) &
    (df["category"].isin(f_category))
)
if f_product:
    mask &= df["product"].isin(f_product)

dff = df.loc[mask].copy()

# Rango anterior para comparación
prev_start, prev_end = period_compare(pd.to_datetime(f_start), pd.to_datetime(f_end))
mask_prev = (
    (df["date"] >= pd.to_datetime(prev_start)) &
    (df["date"] <= pd.to_datetime(prev_end)) &
    (df["region"].isin(f_region)) &
    (df["channel"].isin(f_channel)) &
    (df["category"].isin(f_category))
)
if f_product:
    mask_prev &= df["product"].isin(f_product)
dff_prev = df.loc[mask_prev].copy()

# -------------------------------
# Cálculo de KPIs (medidas)
# -------------------------------
def kpis(data: pd.DataFrame):
    sales = float(data["revenue"].sum())
    orders = int(len(data))
    units = int(data["units"].sum())
    aov = sales / orders if orders else 0.0
    profit = float(data["profit"].sum())
    margin = profit / sales if sales else 0.0
    return dict(sales=sales, orders=orders, units=units, aov=aov, profit=profit, margin=margin)

cur = kpis(dff)
prv = kpis(dff_prev)

def delta_pct(cur_val, prv_val):
    if prv_val == 0:
        return None
    return (cur_val - prv_val) / prv_val

delta_sales = delta_pct(cur["sales"], prv["sales"])
delta_orders = delta_pct(cur["orders"], prv["orders"])
delta_units = delta_pct(cur["units"], prv["units"])
delta_aov = delta_pct(cur["aov"], prv["aov"])
delta_margin = delta_pct(cur["margin"], prv["margin"])

# -------------------------------
# Encabezado
# -------------------------------
st.title("📊 Sales Dashboard — estilo Power BI (Python)")
st.markdown(
    f"<span class='small-note'>Periodo: <b>{f_start.strftime('%d-%b-%Y')}</b> a "
    f"<b>{f_end.strftime('%d-%b-%Y')}</b> • Comparado con {prev_start.strftime('%d-%b-%Y')} a {prev_end.strftime('%d-%b-%Y')}</span>",
    unsafe_allow_html=True
)

# -------------------------------
# KPIs
# -------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ventas", format_currency(cur["sales"]), None if delta_sales is None else f"{delta_sales*100:.1f}%")
c2.metric("Pedidos", f"{cur['orders']:,}".replace(",", "."), None if delta_orders is None else f"{delta_orders*100:.1f}%")
c3.metric("Unidades", f"{cur['units']:,}".replace(",", "."), None if delta_units is None else f"{delta_units*100:.1f}%")
c4.metric("AOV (Ticket medio)", format_currency(cur["aov"]), None if delta_aov is None else f"{delta_aov*100:.1f}%")
c5.metric("Margen", f"{cur['margin']*100:.1f}%", None if delta_margin is None else f"{delta_margin*100:.1f}%")

st.markdown("---")

# -------------------------------
# Utilidad de reamostrar por granularidad
# -------------------------------
def resample_by_granularity(data: pd.DataFrame, gran: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=["Fecha", "Ventas"])
    tmp = data.copy().set_index("date").sort_index()
    if gran == "Día":
        g = tmp.resample("D")["revenue"].sum().rename("Ventas")
    elif gran == "Semana":
        g = tmp.resample("W-MON")["revenue"].sum().rename("Ventas")
    else:
        g = tmp.resample("MS")["revenue"].sum().rename("Ventas")
    return g.reset_index().rename(columns={"date": "Fecha"})

# -------------------------------
# Gráfico 1: Serie temporal de ventas
# -------------------------------
if dff.empty:
    st.warning("Sin datos para los filtros seleccionados.")
else:
    ts = resample_by_granularity(dff, granularity)
    if ts.empty:
        st.info("No hay datos para la serie temporal con los filtros actuales.")
    else:
        fig_ts = px.line(ts, x="Fecha", y="Ventas", markers=True, title="Evolución de Ventas")
        fig_ts.update_layout(margin=dict(l=10, r=10, t=60, b=10), yaxis_tickprefix="€")
        st.plotly_chart(fig_ts, use_container_width=True)

# -------------------------------
# Fila de dos gráficos (categorías y canal)
# -------------------------------
colA, colB = st.columns(2)

# Barras por categoría (Top categorías)
with colA:
    if dff.empty:
        st.info("No hay datos para Categorías.")
    else:
        cat_agg = dff.groupby("category", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
        if cat_agg.empty:
            st.info("Sin datos en Categorías para estos filtros.")
        else:
            fig_cat = px.bar(cat_agg, x="category", y="revenue", text_auto=".2s", title="Ventas por Categoría")
            fig_cat.update_layout(margin=dict(l=10, r=10, t=60, b=10), yaxis_tickprefix="€")
            st.plotly_chart(fig_cat, use_container_width=True)

# Barras apiladas por canal y región
with colB:
    if dff.empty:
        st.info("No hay datos para Región/Canal.")
    else:
        ch_agg = dff.groupby(["region", "channel"], as_index=False)["revenue"].sum()
        if ch_agg.empty:
            st.info("Sin datos para Región/Canal con estos filtros.")
        else:
            fig_channel = px.bar(ch_agg, x="region", y="revenue", color="channel",
                                 barmode="stack", title="Ventas por Región y Canal")
            fig_channel.update_layout(margin=dict(l=10, r=10, t=60, b=10), yaxis_tickprefix="€", legend_title_text="Canal")
            st.plotly_chart(fig_channel, use_container_width=True)

# -------------------------------
# Treemap de productos (drill visual)
# -------------------------------
treemap_df = (
    dff.groupby(["category", "product"], as_index=False)["revenue"]
       .sum()
       .sort_values("revenue", ascending=False)
)

if treemap_df.empty:
    st.info("No hay datos para el treemap con los filtros actuales.")
else:
    labels, parents, values = [], [], []

    # Nivel categoría (padres)
    for cat, sub in treemap_df.groupby("category"):
        labels.append(cat)
        parents.append("")
        values.append(sub["revenue"].sum())

        # Nivel producto (hijos)
        for _, row in sub.iterrows():
            labels.append(row["product"])
            parents.append(cat)
            values.append(float(row["revenue"]))

    fig_tree = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            hovertemplate="<b>%{label}</b><br>Ventas: €%{value:,.0f}<extra></extra>"
        )
    )
    fig_tree.update_layout(
        title="Desglose de Ventas por Categoría → Producto",
        margin=dict(l=10, r=10, t=60, b=10)
    )
    st.plotly_chart(fig_tree, use_container_width=True)

# -------------------------------
# Heatmap Mes x Región (intensidad de ventas)
# -------------------------------
if dff.empty:
    st.info("No hay datos para el heatmap con los filtros actuales.")
else:
    heat = dff.groupby(["month_abbr", "region"], as_index=False)["revenue"].sum()
    heat_pivot = heat.pivot(index="month_abbr", columns="region", values="revenue").fillna(0)
    if heat_pivot.empty:
        st.info("Sin datos suficientes para el heatmap.")
    else:
        fig_heat = go.Figure(data=go.Heatmap(
            z=heat_pivot.values,
            x=list(heat_pivot.columns),
            y=list(heat_pivot.index),
            hoverongaps=False,
            coloraxis="coloraxis"
        ))
        fig_heat.update_layout(
            title="Heatmap de Ventas por Mes y Región",
            coloraxis_colorscale="Blues",
            margin=dict(l=10, r=10, t=60, b=10),
            yaxis_title="Mes",
            xaxis_title="Región",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# -------------------------------
# Top N productos (tabla)
# -------------------------------
st.subheader(f"🏆 Top {topn} productos por ventas")
if dff.empty:
    st.info("No hay datos para la tabla de Top productos.")
else:
    top_products = (
        dff.groupby("product", as_index=False)
           .agg(ventas=("revenue", "sum"),
                pedidos=("product", "count"),
                unidades=("units", "sum"),
                margen=("profit", "sum"))
           .sort_values("ventas", ascending=False)
           .head(topn)
    )

    if top_products.empty:
        st.info("Sin datos para el Top N con estos filtros.")
    else:
        # Métricas derivadas
        top_products["AOV"] = (top_products["ventas"] / top_products["pedidos"]).round(2)
        top_products["Margen %"] = (top_products["margen"] / top_products["ventas"] * 100).round(1)

        # Selección y renombrado
        df_show = top_products[["product", "ventas", "pedidos", "unidades", "AOV", "Margen %"]].rename(
            columns={
                "product": "Producto",
                "ventas": "Ventas",
                "pedidos": "Pedidos",
                "unidades": "Unidades",
            }
        ).copy()

        # Formateo sin Styler
        df_show["Ventas"] = df_show["Ventas"].apply(format_currency)
        df_show["AOV"] = df_show["AOV"].apply(format_currency)
        df_show["Margen %"] = df_show["Margen %"].map(lambda x: f"{x:.1f}%")

        # Mostrar
        st.dataframe(df_show, use_container_width=True, hide_index=True)

# -------------------------------
# Descarga de datos filtrados
# -------------------------------
csv = dff.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Descargar datos filtrados (CSV)",
    data=csv,
    file_name=f"ventas_filtradas_{f_start}_{f_end}.csv",
    mime="text/csv",
)

# -------------------------------
# Nota final
# -------------------------------
st.markdown("""
<div class='small-note'>
✨ Tip: Este panel está hecho en Python. Si vienes de Power BI, verás conceptos familiares:
<b>medidas</b> (KPIs), <b>segmentaciones</b> (filtros), y diferentes <b>visualizaciones</b> para una misma historia.
</div>
""", unsafe_allow_html=True)
