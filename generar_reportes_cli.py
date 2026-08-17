#!/usr/bin/env python3
"""
generar_reportes_cli.py
------------------------
Genera el ZIP de reportes de Sentinel Quant Lab (Excel + PNG por cada
Parquet en data/) SIN abrir Streamlit ni el navegador.

Además, cuando se piden varios archivos, genera un
`reporteTotalConsolidado.md` con el resumen de la pestaña "General"
(las 3 métricas — Volumen/ln(H/L)/Body — SIN filtrar por color de vela)
de todos los símbolos/timeframes pedidos, agrupado por symbol → métrica,
con una fila por timeframe. Así no hay que abrir un Excel por archivo
para comparar.

Reutiliza directamente las funciones del archivo sentinel_quant_lab.py
(glob de parquets, generate_reports_zip, compute_window_stats, etc.), así
que cualquier cambio que hagas en ese archivo (por ejemplo el resumen del
CSV/Excel) se refleja automáticamente aquí también.

USO:
    python generar_reportes_cli.py                # todos los parquets de data/
    python generar_reportes_cli.py BTC_USDT_1d     # solo ese archivo (por nombre)
    python generar_reportes_cli.py BTC_USDT_1d ETH_USDT_1h
    python generar_reportes_cli.py --out /ruta/salida.zip

    # Varios archivos → además del zip, genera reporteTotalConsolidado.md
    python generar_reportes_cli.py BTC_USDT_4h BTC_USDT_1h BTC_USDT_1d BTC_USDT_1m \\
                                    ETH_USDT_4h ETH_USDT_1h ETH_USDT_1d ETH_USDT_1m

    # Solo el consolidado, sin generar el zip (más rápido)
    python generar_reportes_cli.py --solo-consolidado

    # Solo el zip, sin el consolidado (comportamiento original)
    python generar_reportes_cli.py --sin-consolidado

    # Ruta custom para el consolidado
    python generar_reportes_cli.py --out-md /ruta/reporteTotalConsolidado.md
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime

# Importa el módulo original sin ejecutar main() (tiene guard __main__)
import sentinel_quant_lab as sql

# ──────────────────────────────────────────────────────────────────────────
# CONSOLIDADO (.md) — resumen "General" de todos los archivos pedidos,
# agrupado por symbol → métrica, una fila por timeframe.
# ──────────────────────────────────────────────────────────────────────────

# Orden "natural" de timeframes para las filas de la tabla (los que no
# aparezcan acá se listan al final, ordenados alfabéticamente).
TF_ORDER = [
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
]


def _tf_sort_key(tf: str):
    return (TF_ORDER.index(tf), "") if tf in TF_ORDER else (len(TF_ORDER), tf)


def build_consolidated_markdown(file_info: list[dict], progress_cb=None) -> str:
    """
    Recorre cada archivo pedido, calcula las estadísticas de la ventana
    "General (todas)" (sin filtrar por color de vela) para las 3 métricas
    (Volumen/ln(H/L)/Body) usando sql.compute_window_stats (el MISMO motor
    que arma la pestaña "..._General" de cada Excel), y arma un Markdown
    consolidado agrupado por symbol → métrica → fila por timeframe.
    """
    # data[symbol][window_label][timeframe] = (stats_dict | None, window_cfg)
    data: dict[str, dict[str, dict[str, tuple]]] = {}
    total = len(file_info)

    for i, fi in enumerate(file_info, 1):
        if progress_cb:
            progress_cb(i, total, f"Consolidando {fi['display']}")

        try:
            df = sql.load_parquet_for_file(fi["path"])
        except Exception as e:
            for window_label in sql.METRIC_WINDOWS:
                data.setdefault(fi["symbol"], {}).setdefault(window_label, {})[fi["timeframe"]] = (
                    None, sql.METRIC_WINDOWS[window_label], f"Error cargando parquet: {e}",
                )
            continue

        for window_label, window_cfg in sql.METRIC_WINDOWS.items():
            # color_filter=None == "⚪ General (todas)" en CANDLE_FILTERS
            stats = sql.compute_window_stats(df, window_cfg["key"], None)
            data.setdefault(fi["symbol"], {}).setdefault(window_label, {})[fi["timeframe"]] = (
                stats, window_cfg, None,
            )

    lines: list[str] = []
    lines.append("# Reporte Total Consolidado — Sentinel Quant Lab")
    lines.append("")
    lines.append(f"_Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_  ")
    lines.append(f"_Archivos incluidos: {total}_  ")
    lines.append("_Ventana: ⚪ General (todas las velas, sin filtrar por color)_")
    lines.append("")

    header = (
        "| TF | n | min | max | media | mediana | moda | p90 | p95 | p99 "
        "| skew | kurt | entropia | mae | gamma |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"

    for symbol in sorted(data.keys()):
        lines.append(f"## {symbol.replace('/', '')}")
        lines.append("")
        for window_label in sql.METRIC_WINDOWS:  # respeta el orden Volumen/logHL/Body
            tf_map = data[symbol].get(window_label)
            if not tf_map:
                continue

            unit = sql.METRIC_WINDOWS[window_label]["unit_suffix"]
            unit_txt = f" ({unit})" if unit else ""
            lines.append(f"### {window_label}{unit_txt}")
            lines.append("")
            lines.append(header)
            lines.append(sep)

            for tf in sorted(tf_map.keys(), key=_tf_sort_key):
                stats, window_cfg, err = tf_map[tf]
                if stats is None:
                    motivo = err or "sin datos para esta combinación"
                    lines.append(f"| {tf} | _{motivo}_ |" + " |" * 13)
                    continue

                M = window_cfg["unit_divisor"]
                d = window_cfg["decimals"]
                gamma_str = ", ".join(
                    f"{g / M:.{d}f}" for g in stats["gamma_pts"]
                ) or "—"

                lines.append(
                    f"| {tf} | {stats['n']:,} | {stats['ath_min'] / M:.{d}f} | "
                    f"{stats['ath_max'] / M:.{d}f} | {stats['mean'] / M:.{d}f} | "
                    f"{stats['median'] / M:.{d}f} | {stats['mode'] / M:.{d}f} | "
                    f"{stats['p90'] / M:.{d}f} | {stats['p95'] / M:.{d}f} | "
                    f"{stats['p99'] / M:.{d}f} | {stats['skew']:.4f} | "
                    f"{stats['kurtosis']:.4f} | {stats['entropy']:.4f} | "
                    f"{stats['mae']:.6f} | {gamma_str} |"
                )
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera reportes Sentinel por consola.")
    parser.add_argument(
        "nombres", nargs="*",
        help="Nombres de archivo (sin .parquet) a incluir. Si se omite, se incluyen todos."
    )
    parser.add_argument(
        "--data-dir", default=sql.SENTINEL_DIR,
        help=f"Carpeta con los .parquet (default: '{sql.SENTINEL_DIR}')"
    )
    parser.add_argument(
        "--out", default=None,
        help="Ruta del .zip de salida (default: ./sentinel_reportes_<timestamp>.zip)"
    )
    parser.add_argument(
        "--out-md", default=None,
        help="Ruta del .md consolidado (default: ./reporteTotalConsolidado.md)"
    )
    parser.add_argument(
        "--solo-consolidado", action="store_true",
        help="Genera SOLO el reporteTotalConsolidado.md (no arma el .zip con Excel+PNG)."
    )
    parser.add_argument(
        "--sin-consolidado", action="store_true",
        help="No genera el reporteTotalConsolidado.md (comportamiento original, solo .zip)."
    )
    args = parser.parse_args()

    if args.solo_consolidado and args.sin_consolidado:
        print("❌ --solo-consolidado y --sin-consolidado son mutuamente excluyentes.")
        sys.exit(1)

    files = sorted(glob.glob(f"{args.data_dir}/*.parquet"))
    if not files:
        print(f"❌ No se encontraron .parquet en '{args.data_dir}/'")
        sys.exit(1)

    file_info = []
    for f in files:
        name = os.path.basename(f).replace(".parquet", "")
        parts = name.split("_")
        if len(parts) >= 3:
            file_info.append({
                "path": f,
                "name": name,
                "symbol": f"{parts[0]}/{parts[1]}",
                "timeframe": parts[2],
                "display": f"{parts[0]}/{parts[1]} - {parts[2]}",
            })

    if args.nombres:
        file_info = [fi for fi in file_info if fi["name"] in args.nombres]
        if not file_info:
            print(f"❌ Ninguno de los nombres pedidos coincide. Disponibles:")
            for f in files:
                print("   -", os.path.basename(f).replace(".parquet", ""))
            sys.exit(1)

    print(f"📦 Generando reportes para {len(file_info)} archivo(s):")
    for fi in file_info:
        print(f"   - {fi['display']}")

    def _progress(done, total, label):
        print(f"   [{done}/{total}] {label}")

    # ── ZIP (Excel + PNG por archivo) ───────────────────────────────────────
    if not args.solo_consolidado:
        zip_bytes = sql.generate_reports_zip(file_info, progress_cb=_progress)

        out_path = args.out or f"sentinel_reportes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        with open(out_path, "wb") as fh:
            fh.write(zip_bytes)

        print(f"\n✅ Zip listo: {os.path.abspath(out_path)}")

    # ── Consolidado .md (pestaña "General" de todos los archivos pedidos) ──
    if not args.sin_consolidado:
        print(f"\n📝 Generando reporteTotalConsolidado.md ({len(file_info)} archivo(s))...")
        md_text = build_consolidated_markdown(file_info, progress_cb=_progress)

        md_path = args.out_md or "reporteTotalConsolidado.md"
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(md_text)

        print(f"✅ Consolidado listo: {os.path.abspath(md_path)}")


if __name__ == "__main__":
    main()
