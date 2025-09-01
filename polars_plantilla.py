# polars_plantilla.py
# ------------------------------------------------------------
# Ejemplos básicos de Polars (eager + lazy)
# - Genera datos sintéticos de ventas (o lee CSV/Parquet si tienes)
# - Columnas calculadas, filtros, agrupaciones, pivots
# - Window functions, joins, manejo de fechas y nulos
# - Modo lazy con optimización y plan de ejecución
# ------------------------------------------------------------

from __future__ import annotations
import polars as pl
from datetime import date, timedelta
import random

# ---------- Utilidades ----------
def print_h2(t):
    print("\n" + "="*len(t))
    print(t)
    print("="*len(t))

def save_sample_files(df_sales: pl.DataFrame, df_customers: pl.DataFrame):
    df_sales.write_csv("ventas.csv")
    df_sales.write_parquet("ventas.parquet")
    df_customers.write_csv("clientes.csv")

# ---------- 0) Datos sintéticos ----------
def build_sample_sales(n_rows: int = 8000, seed: int = 42) -> tuple[pl.DataFrame, pl.DataFrame]:
    random.seed(seed)
    # pl.enable_string_cache()  # <- opcional; no necesaria aquí

    regions = ["Norte", "Sur", "Este", "Oeste"]
    channels = ["Retail", "Online", "Distribuidor"]
    categories = {
        "Electrónica": ["Auriculares X1", "Smartwatch Z", "Tablet Pro", "Cámara HD"],
        "Hogar": ["Cafetera Barista", "Aspiradora Max", "Lámpara LED", "Silla Ergo"],
        "Moda": ["Zapatilla Run", "Chaqueta Wind", "Mochila City", "Gafas Sun"],
        "Deporte": ["Bici Urban", "Casco Pro", "Guantes Grip", "Balón Elite"],
    }
    # rango de fechas
    start, end = date(2024, 1, 1), date(2025, 8, 31)
    days = (end - start).days

    rows = []
    for _ in range(n_rows):
        d = start + timedelta(days=random.randint(0, days))
        cat = random.choice(list(categories.keys()))
        prod = random.choice(categories[cat])
        region = random.choice(regions)
        channel = random.choices(channels, weights=[0.45, 0.4, 0.15])[0]
        customer = f"CUST-{random.randint(1, 800):04d}"
        price_base = {"Electrónica": 180, "Hogar": 90, "Moda": 60, "Deporte": 140}[cat]
        price = max(5.0, random.gauss(price_base, price_base * 0.15))
        qty = max(1, int(abs(random.gauss(1.8, 0.9))))
        discount = min(max(random.gauss(0.06, 0.05), 0.0), 0.35)

        price_final = price * (1 - discount)
        revenue = price_final * qty
        cost_unit = max(2.0, price_base * 0.7 + random.gauss(0, price_base * 0.05))
        cost = cost_unit * qty
        profit = revenue - cost

        rows.append(
            (d, region, channel, cat, prod, customer, qty,
             round(price_final, 2), round(revenue, 2), round(cost, 2),
             round(profit, 2), discount)
        )

    df_sales = pl.DataFrame(
        rows,
        schema=[
            ("date", pl.Date), ("region", pl.Utf8), ("channel", pl.Utf8),
            ("category", pl.Utf8), ("product", pl.Utf8), ("customer", pl.Utf8),
            ("qty", pl.Int64), ("price", pl.Float64), ("revenue", pl.Float64),
            ("cost", pl.Float64), ("profit", pl.Float64), ("discount", pl.Float64),
        ],
    )

    # tabla de clientes (para ejemplo de join)
    rows_c = []
    for i in range(1, 801):
        cust = f"CUST-{i:04d}"
        seg = random.choice(["A", "B", "C"])
        rows_c.append((cust, seg))
    df_customers = pl.DataFrame(rows_c, schema=[("customer", pl.Utf8), ("segment", pl.Utf8)])

    return df_sales, df_customers

# ---------- 1) Eager: columnas, filtro, group_by ----------
def eager_examples(df: pl.DataFrame):
    print_h2("1) EAGER: columnas calculadas, filtros y agregaciones")
    out = (
        df.with_columns([
            pl.col("revenue").round(2),
            pl.col("date").dt.truncate("1mo").alias("month"),
            (pl.col("revenue") / pl.col("qty")).alias("avg_item_price")
        ])
        .filter((pl.col("region") == "Norte") & (pl.col("discount") <= 0.10))
        .group_by(["month", "channel"])
        .agg([
            pl.sum("revenue").alias("sales"),
            pl.len().alias("orders"),
            pl.mean("profit").alias("avg_profit"),
        ])
        .sort(["month", "channel"])
    )
    print(out.head(10))

# ---------- 2) Pivot (tabla dinámica simple) ----------
def pivot_example(df: pl.DataFrame):
    print_h2("2) Pivot: Ventas por mes x región")
    base = df.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
    piv = base.pivot(values="revenue", index="month", columns="region", aggregate_function="sum").sort("month")
    print(piv.head(12))

# ---------- 3) Window functions ----------
def window_examples(df: pl.DataFrame):
    print_h2("3) Window functions: totales por cliente y % contribución")
    w = df.select([
        "customer", "revenue",
        pl.sum("revenue").over("customer").alias("cust_sales"),
        (pl.col("revenue") / pl.sum("revenue").over("customer")).alias("cust_share")
    ])
    print(w.head(8))

# ---------- 4) Join con dimensión clientes ----------
def join_example(df_sales: pl.DataFrame, df_customers: pl.DataFrame):
    print_h2("4) Join: agregar segmento del cliente")
    j = df_sales.join(df_customers, on="customer", how="left")
    print(j.select("customer", "segment").head(8))

# ---------- 5) Fechas: agrupación mensual con group_by_dynamic ----------
def monthly_timeseries(df: pl.DataFrame):
    print_h2("5) Serie temporal mensual con group_by_dynamic")
    ts = (
        df.sort("date")  # ✅ imprescindible para group_by_dynamic
          .group_by_dynamic(
              "date",
              every="1mo",
              label="left",   # mes etiquetado al inicio (opcional)
              closed="left"   # intervalos [inicio, fin) (opcional)
          )
          .agg(
              pl.sum("revenue").alias("sales"),
              pl.len().alias("orders"),
              (pl.sum("revenue") / pl.len()).alias("aov")
          )
          .rename({"date": "month"})
          .sort("month")
    )
    print(ts.head(6))

# ---------- 6) Nulos: rellenar y derivadas ----------
def nulls_example(df: pl.DataFrame):
    print_h2("6) Manejo de nulos (fill_null / coalesce)")
    sample = df.with_columns(
        pl.when(pl.arange(0, pl.len()) % 10 == 0).then(None).otherwise(pl.col("discount")).alias("discount")
    )
    out = sample.select([
        "discount",
        pl.coalesce([pl.col("discount"), pl.lit(0.0)]).alias("discount_filled")
    ]).head(12)
    print(out)

# ---------- 7) Lazy: pipeline optimizado ----------
def lazy_examples():
    print_h2("7) LAZY: scan_csv + optimización + collect()")
    try:
        lf = pl.scan_csv("ventas.csv")
    except Exception:
        lf = pl.scan_parquet("ventas.parquet")

    q = (
        lf.with_columns([
            pl.col("date").str.strptime(pl.Date, strict=False).alias("date"),
            (pl.col("revenue") - pl.col("cost")).alias("profit"),
        ])
        .filter(pl.col("channel") != "Distribuidor")
        .group_by([pl.col("date").dt.truncate("1mo").alias("month"), "region"])
        .agg([
            pl.sum("revenue").alias("sales"),
            pl.len().alias("orders"),
        ])
        .sort(["month", "region"])
    )

    print("\n-- Plan optimizado --")
    print(q.explain())
    res = q.collect()
    print("\n-- Resultado (head) --")
    print(res.head(10))

# ---------- 8) Exportar / interoperar ----------
def export_examples(df: pl.DataFrame):
    print_h2("8) Exportar a Parquet y CSV + pasar a pandas")
    df.write_parquet("ventas_demo.parquet")
    df.write_csv("ventas_demo.csv")
    pdf = df.head(5).to_pandas()
    print(pdf)

# ================== MAIN ==================
if __name__ == "__main__":
    sales, customers = build_sample_sales()
    save_sample_files(sales, customers)

    eager_examples(sales)
    pivot_example(sales)
    window_examples(sales)
    join_example(sales, customers)
    monthly_timeseries(sales)
    nulls_example(sales)

    lazy_examples()
    export_examples(sales)

    print("\n✅ Listo. Revisa los CSV/Parquet generados en el directorio actual.")
