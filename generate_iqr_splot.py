# Dataset de VENTAS diarias desde 2024, con outliers y categorización por IQR mensual.
# - Genera ventas por día (producto x región)
# - Calcula Q1, Q3, IQR y límites por MES
# - Marca outliers y clasifica el mes por IQR (Bajo / Medio− / Medio+ / Alto)
# - Exporta CSV y dibuja boxplots (visual más usado para IQR)
#
# Reglas de visual: matplotlib, una figura por gráfico, sin estilos/colores fijos.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(7)

# 1) Datos de ventas diarios (2024->hoy)
start = pd.Timestamp("2024-01-01")
end   = pd.Timestamp.today().normalize()
days  = pd.date_range(start, end, freq="D")

productos = ["Alpha", "Beta", "Gamma", "Delta"]
regiones  = ["Norte", "Sur", "Este", "Oeste"]
base_prod   = {"Alpha": 520, "Beta": 430, "Gamma": 610, "Delta": 350}
mult_region = {"Norte":1.05, "Sur":0.95, "Este":1.10, "Oeste":0.90}
seasonality = {1:0.92,2:0.95,3:0.98,4:1.00,5:1.02,6:1.05,7:1.10,8:1.12,9:1.08,10:1.15,11:1.25,12:1.40}

rows, rid = [], 0
for d in days:
    promo = 1.0 + (np.random.uniform(0.15, 0.60) if np.random.rand() < 0.08 else 0.0)
    for p in productos:
        for r in regiones:
            mean = base_prod[p] * mult_region[r] * seasonality[d.month] * promo
            v = np.random.lognormal(mean=np.log(max(mean, 1)), sigma=0.25)
            if np.random.rand() < 0.02:
                v *= np.random.uniform(1.8, 3.0)   # outlier alto
            if np.random.rand() < 0.015:
                v *= np.random.uniform(0.2, 0.5)   # outlier bajo
            rows.append((rid, d, d.to_period("M").strftime("%Y-%m"), p, r, float(v)))
            rid += 1

df = pd.DataFrame(rows, columns=["id","fecha","mes","producto","region","ventas"])

# 2) IQR por mes
q1 = df.groupby("mes")["ventas"].quantile(0.25).rename("q1")
q3 = df.groupby("mes")["ventas"].quantile(0.75).rename("q3")
iqr = (q3 - q1).rename("iqr")
limits = pd.concat([q1, q3, iqr], axis=1).reset_index()
limits["lower"] = limits["q1"] - 1.5 * limits["iqr"]
limits["upper"] = limits["q3"] + 1.5 * limits["iqr"]

df = df.merge(limits, on="mes", how="left")
df["outlier"] = (df["ventas"] < df["lower"]) | (df["ventas"] > df["upper"])

# 3) Últimos 12 meses para los gráficos
meses = sorted(df["mes"].unique())[-12:]
vals_by_month = [df.loc[df["mes"] == m, "ventas"].values for m in meses]

# 4) Boxplot mensual (con cuadrícula, escala lineal)
plt.figure(figsize=(12, 6))
plt.boxplot(vals_by_month, labels=meses, showfliers=True, notch=True, showmeans=True, meanline=True)
plt.title("Ventas – IQR mensual (últimos 12 meses)")
plt.xlabel("Mes"); plt.ylabel("Ventas")
plt.xticks(rotation=30)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# 5) Violin plot mensual (con cuadrícula)
plt.figure(figsize=(12, 6))
plt.violinplot(vals_by_month, showmeans=False, showextrema=True, showmedians=True)
plt.title("Ventas – Distribución mensual (Violin)")
plt.xlabel("Mes"); plt.ylabel("Ventas")
plt.xticks(range(1, len(meses)+1), meses, rotation=30)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# 6) Beeswarm simple + box del último mes (con cuadrícula)
last = meses[-1]
vals = df.loc[df["mes"] == last, "ventas"].values
x = np.random.normal(loc=0.0, scale=0.06, size=len(vals))  # jitter horizontal
plt.figure(figsize=(6, 6))
plt.boxplot([vals], labels=[last], showfliers=False)
plt.plot(x + 1, vals, linestyle="", marker="o", alpha=0.4)
plt.title(f"Ventas – Puntos y IQR ({last})")
plt.xlabel("Mes"); plt.ylabel("Ventas")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

