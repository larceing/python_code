
"""
Cómo ejecutar este ejemplo (instalación rápida)

1) Requisitos: Python 3.10+
2) Instala dependencias (NO uses 'sklearn' en pip, usa 'scikit-learn'):
   pip install flask scikit-learn pandas numpy

3) Ejecuta:
   python sample_flask.py

4) Abre en el navegador:
   http://localhost:8000/

Endpoints:
- GET  /status      → estado del servicio
- GET  /summary     → métricas descriptivas del dataset y del modelo
- POST /predict     → predicción (JSON)
- GET  /metrics     → métricas de evaluación del modelo
- POST /retrain     → regenera datos y reentrena (opcionalmente con parámetros)
- GET  /download.csv→ descarga del dataset sintético
- GET  /docs        → documentación rápida con ejemplos curl
- GET  /dashboard   → placeholder para incrustar un informe (Power BI, etc.)
"""

from __future__ import annotations
from flask import Flask, jsonify, request, Response, render_template_string
import os
import io
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -------------------------
# Config
# -------------------------
PORT = int(os.environ.get("PORT", 8000))
DEBUG = bool(int(os.environ.get("DEBUG", 1)))
# Si quieres proteger /retrain en producción, define una API key en el entorno.
API_KEY = os.environ.get("API_KEY")  # opcional

app = Flask(__name__)

# Estado global simple para la demo
STATE = {
    "df": None,
    "model": None,
    "feature_names": ["marketing_spend", "num_sellers", "season_index"],
    "metrics": {},
}

# -------------------------
# Generación de datos y entrenamiento
# -------------------------

def generate_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    marketing_spend = rng.gamma(shape=9.0, scale=1200.0, size=n)
    num_sellers = rng.poisson(7, n) + 3
    season_index = np.sin(np.linspace(0, 6*np.pi, n)) * 0.5 + 0.5
    noise = rng.normal(0, 5000, n)
    sales = 20000 + 2.1*marketing_spend + 3500*num_sellers + 18000*season_index + noise
    df = pd.DataFrame({
        "marketing_spend": marketing_spend,
        "num_sellers": num_sellers,
        "season_index": season_index,
        "sales": sales,
    })
    return df


def train_model(df: pd.DataFrame):
    X = df[["marketing_spend", "num_sellers", "season_index"]]
    y = df["sales"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    metrics = {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "coefficients": dict(zip(X.columns.tolist(), model.coef_.astype(float).tolist())),
        "intercept": float(model.intercept_),
    }
    return model, metrics


def bootstrap(n: int = 300, seed: int = 42):
    STATE["df"] = generate_data(n=n, seed=seed)
    STATE["model"], STATE["metrics"] = train_model(STATE["df"])


# Inicializa al arrancar
bootstrap()

# -------------------------
# Utilidades
# -------------------------

def require_api_key_if_set():
    if API_KEY:
        provided = request.headers.get("X-API-Key") or request.args.get("api_key")
        if provided != API_KEY:
            return jsonify({"error": "API key inválida o ausente"}), 401
    return None


def data_summary(df: pd.DataFrame) -> dict:
    desc = df.describe().to_dict()
    corr = df.corr(numeric_only=True).round(4).to_dict()
    preview = df.head(5).to_dict(orient="records")
    return {
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "describe": desc,
        "correlation": corr,
        "preview": preview,
    }


# -------------------------
# Rutas
# -------------------------
@app.route("/")
def home():
    html = """
    <!doctype html>
    <html lang='es'>
      <head>
        <meta charset='utf-8'/>
        <meta name='viewport' content='width=device-width, initial-scale=1'/>
        <title>Flask Data Analytics — Demo</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
          body { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; }
          .card { max-width: 1000px; margin: auto; padding: 1.5rem; border: 1px solid #eee; border-radius: 14px; box-shadow: 0 6px 24px rgba(0,0,0,0.06); }
          code { background: #f6f8fa; padding: 2px 6px; border-radius: 6px; }
          pre { background: #f6f8fa; padding: 1rem; border-radius: 12px; overflow:auto; }
          a { text-decoration: none; }
          .grid { display:grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
          .placeholder { height: 420px; border: 2px dashed #bbb; display:flex; align-items:center; justify-content:center; border-radius: 12px; }
          @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
        </style>
      </head>
      <body>
        <div class='card'>
          <h1>Flask Data Analytics — Demo (1 archivo)</h1>
          <p>API minimalista que genera datos sintéticos, entrena un modelo y expone endpoints para resumen y predicción.</p>
          <ul>
            <li><a href='/status'>/status</a></li>
            <li><a href='/summary'>/summary</a></li>
            <li><a href='/metrics'>/metrics</a></li>
            <li><a href='/docs'>/docs</a></li>
            <li><a href='/download.csv'>/download.csv</a></li>
            <li><a href='/dashboard'>/dashboard</a> (placeholder)</li>
          </ul>
          <div class='grid'>
            <div>
              <h2>Distribución de ventas (Chart.js)</h2>
              <canvas id='salesChart'></canvas>
            </div>
            <div>
              <h2>Predicción rápida</h2>
              <form id='predForm'>
                <label>marketing_spend <input type='number' step='0.01' name='marketing_spend' value='12000'></label><br/>
                <label>num_sellers <input type='number' name='num_sellers' value='8'></label><br/>
                <label>season_index <input type='number' step='0.01' name='season_index' value='0.6' min='0' max='1'></label><br/>
                <button type='submit'>Predecir</button>
              </form>
              <pre id='predOut'>Resultado aquí…</pre>
            </div>
          </div>
        </div>
        <script>
          // Cargar summary y dibujar gráfico (preview)
          fetch('/summary').then(r => r.json()).then(js => {
            const sales = js.preview.map(r => r.sales); // solo preview para demo
            const labels = sales.map(function(value, index) { return "muestra " + (index + 1); });
            const ctx = document.getElementById('salesChart');
            new Chart(ctx, {
              type: 'bar',
              data: {
                labels: labels,
                datasets: [{ label: 'Sales (preview)', data: sales }]
              },
              options: { responsive: true, plugins: { legend: { display: true } } }
            });
          });

          // Formulario de predicción
          document.getElementById('predForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const fd = new FormData(e.target);
            const payload = Object.fromEntries(fd.entries());
            payload.marketing_spend = parseFloat(payload.marketing_spend);
            payload.num_sellers = parseInt(payload.num_sellers);
            payload.season_index = parseFloat(payload.season_index);
            const res = await fetch('/predict', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload)
            });
            const js = await res.json();
            document.getElementById('predOut').textContent = JSON.stringify(js, null, 2);
          });
        </script>
      </body>
    </html>
    """
    return render_template_string(html)


@app.route("/status")
def status():
    return jsonify({"status": "ok", "service": "analytics-demo", "version": 1})


@app.route("/summary")
def summary():
    return jsonify(data_summary(STATE["df"]))


@app.route("/metrics")
def metrics():
    return jsonify(STATE["metrics"])


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True, silent=True) or {}
    required = STATE["feature_names"]
    missing = [k for k in required if k not in body]
    if missing:
        return jsonify({"error": f"Faltan features: {missing}", "required": required}), 400

    try:
        X_new = [[
            float(body["marketing_spend"]),
            int(body["num_sellers"]),
            float(body["season_index"]),
        ]]
    except Exception as e:
        return jsonify({"error": f"Entrada inválida: {e}"}), 400

    y_hat = float(STATE["model"].predict(X_new)[0])
    return jsonify({"prediction": y_hat, "units": "sales"})


@app.route("/retrain", methods=["POST"])
def retrain():
    # (Opcional) proteger con API key
    guard = require_api_key_if_set()
    if guard is not None:
        return guard

    payload = request.get_json(force=True, silent=True) or {}
    n = int(payload.get("n", 300))
    seed = int(payload.get("seed", 42))
    bootstrap(n=n, seed=seed)
    return jsonify({"status": "ok", "message": "Modelo reentrenado", "n": n, "seed": seed, "metrics": STATE["metrics"]})


@app.route("/download.csv")
def download_csv():
    buf = io.StringIO()
    STATE["df"].to_csv(buf, index=False)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=synthetic_sales.csv"},
    )


@app.route("/docs")
def docs():
    curl_predict = """
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"marketing_spend":12000, "num_sellers":8, "season_index":0.6}'
    """.strip()
    html = f"""
    <h1>Docs rápidas</h1>
    <p>Ejemplos y descripciones de la API.</p>
    <ul>
      <li><code>GET /status</code> → ping de salud</li>
      <li><code>GET /summary</code> → describe & correlación</li>
      <li><code>GET /metrics</code> → r2, mae, rmse, coeficientes</li>
      <li><code>POST /predict</code> → predicción: JSON con <code>marketing_spend</code>, <code>num_sellers</code>, <code>season_index</code></li>
      <li><code>POST /retrain</code> → regenera datos (opcional body: <code>{{'n': 500, 'seed': 7}}</code>)</li>
      <li><code>GET /download.csv</code> → descarga del dataset</li>
      <li><code>GET /dashboard</code> → placeholder para embed</li>
    </ul>
    <h2>Ejemplo curl</h2>
    <pre>{curl_predict}</pre>
    """
    return render_template_string(html)


@app.route("/dashboard")
def dashboard():
    html = """
    <!doctype html>
    <html lang="es">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Dashboard Embed (Placeholder)</title>
        <style>
          body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; }
          .card { max-width: 1100px; margin: auto; padding: 1.5rem; border: 1px solid #eee; border-radius: 12px; box-shadow: 0 6px 24px rgba(0,0,0,0.06); }
          .placeholder { height: 540px; border: 2px dashed #bbb; display:flex; align-items:center; justify-content:center; border-radius: 12px; }
        </style>
      </head>
      <body>
        <div class="card">
          <h1>Dashboard Embed</h1>
          <p>Este es un <strong>placeholder</strong> para incrustar tu informe (Power BI, Tableau, etc.). Sustituye el bloque inferior por tu <code>&lt;iframe src="..."&gt;</code> o tu script de autenticación.</p>
          <div class="placeholder"><!-- TODO: PEGAR IFRAME AQUÍ --></div>
        </div>
      </body>
    </html>
    """
    return render_template_string(html)


# -------------------------
# Manejo de errores amigable
# -------------------------
@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad Request", "message": str(e)}), 400


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal Server Error", "message": str(e)}), 500


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)
