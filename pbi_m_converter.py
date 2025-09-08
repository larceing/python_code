# pbi_m_converter.py
# Convertir Excel/CSV a código M (Power Query) embebido (Base64 + Deflate)
# Uso como librería:
#   from pbi_m_converter import convert_file_to_m, dataframe_to_m, save_m_code
# Uso como script:
#   python pbi_m_converter.py ruta/al/archivo.xlsx --sheet Hoja1 --out codigo_powerbi.txt
#   python pbi_m_converter.py datos.csv --stdout

from __future__ import annotations
import argparse
import base64
import csv
import datetime as _dt
from pathlib import Path
import sys
import zlib
from typing import Optional

import pandas as pd


# ---------------------------- Núcleo reutilizable ---------------------------- #

def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    if df.empty:
        raise ValueError("El DataFrame está vacío.")
    text = df.to_csv(
        index=False,
        sep=",",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    return text.encode("utf-8")


def _csv_bytes_to_base64_deflate(csv_bytes: bytes) -> str:
    # Compresión zlib y recorte para obtener "raw deflate" compatible con Power Query
    comp = zlib.compress(csv_bytes, level=zlib.Z_BEST_COMPRESSION)
    raw_deflate = comp[2:-4]  # quitar cabecera (2) y Adler-32 (4)
    return base64.b64encode(raw_deflate).decode("utf-8")


def dataframe_to_m(df: pd.DataFrame) -> str:
    """Genera código M para cargar el DataFrame como CSV embebido."""
    cols = list(df.columns)
    cols_m = ", ".join([f'{{"{c}", type text}}' for c in cols])

    base64_data = _csv_bytes_to_base64_deflate(_df_to_csv_bytes(df))

    m = f"""
let
    Origen = Csv.Document(
        Binary.Decompress(
            Binary.FromText("{base64_data}", BinaryEncoding.Base64),
            Compression.Deflate
        ),
        [Delimiter=",", Columns={len(cols)}, Encoding=65001, QuoteStyle=QuoteStyle.None]
    ),
    #"Encabezados asignados" = Table.PromoteHeaders(Origen, [PromoteAllScalars=true]),
    #"Tipo cambiado" = Table.TransformColumnTypes(#"Encabezados asignados", {{ {cols_m} }})
in
    #"Tipo cambiado"
""".strip("\n")
    return m


def convert_file_to_m(path: str | Path, *, sheet: Optional[str] = None) -> str:
    """Lee Excel/CSV y devuelve el código M correspondiente."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo: {p}")

    ext = p.suffix.lower()
    if ext in {".xls", ".xlsx", ".xlsm"}:
        df = pd.read_excel(p, sheet_name=sheet) if sheet else pd.read_excel(p)
        # Si devuelve dict (libro con varias hojas + sin sheet), tomar la primera
        if isinstance(df, dict):
            first_key = next(iter(df))
            df = df[first_key]
    elif ext in {".csv"}:
        df = pd.read_csv(p)
    else:
        # Intento flexible: primero CSV, si falla, Excel
        try:
            df = pd.read_csv(p)
        except Exception:
            df = pd.read_excel(p, sheet_name=sheet) if sheet else pd.read_excel(p)
            if isinstance(df, dict):
                first_key = next(iter(df))
                df = df[first_key]

    return dataframe_to_m(df)


def save_m_code(m_code: str, out_path: str | Path, *, append_timestamp: bool = True, overwrite: bool = False) -> Path:
    """Guarda el código M en un archivo. Por defecto añade timestamp y no sobreescribe."""
    out = Path(out_path)
    if out.exists() and not overwrite:
        mode = "a"
    else:
        mode = "w"

    with out.open(mode, encoding="utf-8") as f:
        if append_timestamp:
            ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n\n--- Código generado el {ts} ---\n")
        f.write(m_code)
        if append_timestamp:
            f.write("\n\n" + "=" * 80 + "\n")

    return out


# ----------------------------------- CLI ------------------------------------ #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convierte un Excel/CSV en código M embebido (Base64+Deflate) para Power Query."
    )
    p.add_argument("input", help="Ruta al archivo de entrada (Excel o CSV).")
    p.add_argument("--sheet", help="Nombre de la hoja (para Excel).", default=None)
    p.add_argument("--out", help="Archivo de salida (por defecto: codigo_powerbi.txt).", default="codigo_powerbi.txt")
    p.add_argument("--stdout", help="Imprimir el código M por stdout en lugar de guardar.", action="store_true")
    p.add_argument("--no-timestamp", help="No añadir separadores/timestamp al guardar.", action="store_true")
    p.add_argument("--overwrite", help="Sobrescribir el archivo de salida si existe.", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        m_code = convert_file_to_m(args.input, sheet=args.sheet)
    except Exception as e:
        print(f"[ERROR] No se pudo convertir el archivo: {e}", file=sys.stderr)
        return 1

    if args.stdout:
        print(m_code)
        return 0

    try:
        save_m_code(
            m_code,
            args.out,
            append_timestamp=not args.no_timestamp,
            overwrite=args.overwrite,
        )
        print(f"[OK] Código M guardado en: {args.out}")
        return 0
    except Exception as e:
        print(f"[ERROR] No se pudo guardar el código M: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
