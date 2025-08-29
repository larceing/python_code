#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Escaneo de columnas que contengan palabras clave en la base de datos.
- Busca columnas con esos términos (case-insensitive).
- Cuenta cuántas filas tienen valor no nulo / >0 (numérico) o no vacío (texto).
- Guarda resultados en CSV y muestra resumen.
"""
"""
    Conexión a la base de datos.
    ⚠️ Recordatorio:
    - Asegúrate de tener instalado el conector correcto:
        pip install mariadb
        pip install mysql-connector-python
    - Anota/usa siempre la versión del conector y del motor (MariaDB/MySQL),
      ya que algunas funciones pueden variar según la versión.
    - Usa un usuario con permisos de SOLO LECTURA en entornos productivos.
"""


import csv, sys

DB_HOST = "****"
DB_USER = "****"
DB_PASS = "****"
DB_NAME = "****"
PORT    = 3306
CSV_OUT = "results.csv"

SEARCH_TERMS = ["Etiqueta", "Centro","Almacen","Calle","Código de Barras"]  # términos a buscar en nombres de columnas

NUMERIC_TYPES = {
    "int","integer","bigint","smallint","tinyint","mediumint",
    "decimal","numeric","float","double","double precision","real"
}
TEXT_TYPES = {"char","varchar","text","tinytext","mediumtext","longtext"}

def connect_any():
    try:
        import mariadb
        return mariadb.connect(user=DB_USER, password=DB_PASS, host=DB_HOST, port=PORT, database=DB_NAME), "mariadb"
    except Exception:
        import mysql.connector as mc
        return mc.connect(user=DB_USER, password=DB_PASS, host=DB_HOST, port=PORT, database=DB_NAME, autocommit=True), "mysql.connector"

def q(conn, sql, params=None):
    cur = conn.cursor()
    cur.execute(sql, params or ())
    rows = cur.fetchall()
    cur.close()
    return rows

def main():
    conn, drv = connect_any()
    print(f"Conectado con {drv}. Schema: {DB_NAME}")

    # Armar condición OR para todos los términos
    like_clauses = " OR ".join(["LOWER(COLUMN_NAME) LIKE %s" for _ in SEARCH_TERMS])
    params = [f"%{t}%" for t in SEARCH_TERMS]
    sql = f"""
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, COLUMN_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s AND ({like_clauses})
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """
    cols = q(conn, sql, [DB_NAME] + params)

    if not cols:
        print(f"No se encontraron columnas que contengan {SEARCH_TERMS} en su nombre.")
        return

    print(f"Encontradas {len(cols)} columnas con {SEARCH_TERMS} en {DB_NAME}.\n")

    results = []
    for table_name, column_name, data_type, column_type in cols:
        fq_table = f"`{DB_NAME}`.`{table_name}`"
        col = f"`{column_name}`"
        total = None
        non_empty = None
        cond = ""
        try:
            total = q(conn, f"SELECT COUNT(*) FROM {fq_table}")[0][0]
            dt = (data_type or "").lower()
            if dt in NUMERIC_TYPES:
                cond = f"{col} IS NOT NULL AND {col} > 0"
            elif dt in TEXT_TYPES:
                cond = f"{col} IS NOT NULL AND TRIM({col}) <> '' AND {col} <> '0'"
            else:
                cond = f"{col} IS NOT NULL"
            non_empty = q(conn, f"SELECT COUNT(*) FROM {fq_table} WHERE {cond}")[0][0]

            results.append({
                "table": table_name,
                "column": column_name,
                "data_type": data_type,
                "column_type": column_type,
                "total_rows": total,
                "matching_rows": non_empty,
                "condition": cond
            })

            print(f"[OK] {DB_NAME}.{table_name}.{column_name} ({data_type}) "
                  f"→ filas={total}, cumplen={non_empty}  cond=({cond})")
        except Exception as e:
            print(f"[ERROR] {DB_NAME}.{table_name}.{column_name} → {type(e).__name__}: {e}")

    # Guardar CSV
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["table","column","data_type","column_type","total_rows","matching_rows","condition"])
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"\nCSV guardado en: {CSV_OUT}")

    # Resumen rápido
    print(f"\n=== COLUMNA(S) CON {SEARCH_TERMS} PRESENTE ===")
    any_found = False
    for r in results:
        if (r["matching_rows"] or 0) > 0:
            any_found = True
            print(f"  - {DB_NAME}.{r['table']}.{r['column']} → {r['matching_rows']} filas")
    if not any_found:
        print("  (No se hallaron filas con valores no nulos / >0 / no vacíos)")

    conn.close()

if __name__ == "__main__":
    main()
