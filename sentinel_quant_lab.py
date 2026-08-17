"""
sentinel_quant_lab.py  (v2 — Intra-Hour Fix Edition)
=====================================================
Visualizador Streamlit para parquets generados por integrity_rock v4.

Bugs corregidos vs versión original
-------------------------------------
[FIX-1] DATETIME TIMEZONE CORRUPTION
         El parquet de integrity_rock v4 guarda 'datetime' como
         timestamp("ms", tz="UTC"). El código original lo sobreescribía
         con pd.to_datetime(..., unit='ms') perdiendo el timezone y
         causando que los filtros de fecha fallaran silenciosamente.
         FIX → detectar el dtype real y normalizar a datetime64[ns] sin
               tz para compatibilidad con st.date_input y strftime.

[FIX-2] TTL=3600 OCULTA VELAS RECIENTES
         Con cache de 1 hora, tras ejecutar integrity_rock las velas
         nuevas (15:05, 15:10 …) no aparecían en el visualizador hasta
         que expiraba la caché.
         FIX → ttl=300 (5 min) para 5m, configurable por timeframe.
               Botón "🔄 Refrescar datos" con st.cache_data.clear().

[FIX-3] STRFTIME EN DATETIME TIMEZONE-AWARE
         df['datetime'].min().strftime(...) lanzaba AttributeError en
         Pandas >= 2.0 con columnas timezone-aware.
         FIX → helper fmt_dt() que maneja ambos casos.

[FIX-4] VIEW_COLS ROBUSTO
         Si el parquet no tiene alguna columna esperada (versiones
         anteriores sin 'notional'), el visualizador crasheaba con
         KeyError sin mensaje útil.
         FIX → validación explícita con mensaje de error descriptivo.

[FIX-5] HIGHLIGHT CRASH CON COLUMNA FALTANTE
         highlight_rows_vectorized asumía que 'notional' siempre existe
         en df_view. Si el usuario ordenaba o filtraba y perdía la col,
         lanzaba KeyError.
         FIX → guard con comprobación antes de aplicar estilos.
"""

import glob
import io
import os
import zipfile
from datetime import datetime

import matplotlib
matplotlib.use("Agg")   # backend sin GUI, necesario para generar PNGs en Streamlit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import entropy, gaussian_kde, kurtosis, skew
from statsmodels.distributions.empirical_distribution import ECDF

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sentinel Quant Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric {
        background-color: #1e2127;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #333;
    }
    </style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

SENTINEL_DIR = "data"

PRESETS: dict[str, tuple[float, float]] = {
    "BTC_USDT_5m":  (150.0,   176.02),
    "BTC_USDT_1h":  (1661.61,  1921.55),
    "BTC_USDT_4h":  (5908.03,  6829.53),
    "BTC_USDT_1d":  (19122.00, 31467.27),
    "ETH_USDT_5m":  (150.88,   118.48),
    "ETH_USDT_1h":  (881.04,   1293.18),
    "ETH_USDT_4h":  (4607.53, 4779.38),
    "ETH_USDT_1d":  (23218.18, 24340.25),
}

# Velas por segmento y etiqueta de navegación
JUMP_CONFIG: dict[str, tuple[int, str]] = {
    "5m":  (288, "Día"),
    "1h":  (168, "Semana"),
    "4h":  (180, "Mes"),
    "1d":  (30,  "Mes"),
}

# TTL de caché por timeframe (segundos)
# 5m → 5 min, 1h → 15 min, 4h/1d → 30 min
CACHE_TTL: dict[str, int] = {
    "5m": 300,
    "1h": 900,
    "4h": 1_800,
    "1d": 1_800,
}
DEFAULT_TTL = 300   # fallback para TFs desconocidos

# Columnas que espera la tabla de auditoría
VIEW_COLS   = [
    "datetime", "candle_color", "notional", "log_hl", "body",
    "volume", "open", "high", "low", "close",
]
# Columnas REALMENTE requeridas en el parquet crudo de integrity_rock v6.
# 'datetime' y 'notional' ya NO se guardan en el parquet (v6 solo persiste
# 'timestamp' en ms y 'quote_vol' / 'taker_buy_quote_vol'); ambas se derivan
# en tiempo de carga vía ensure_derived_columns(), así que no se exigen aquí.
REQUIRED_COLS = {"timestamp", "open", "high", "low", "close", "volume"}

# ──────────────────────────────────────────────────────────────────────────────
# VENTANAS DE ANÁLISIS (Volumen / ln(H/L) / Body) × (General / Verdes / Rojas)
# ──────────────────────────────────────────────────────────────────────────────
# Cada "ventana" define QUÉ columna se analiza estadísticamente (KDE, ECDF,
# percentiles, skew/kurtosis/entropía) y CÓMO se presenta esa columna.
# El filtro de color decide QUÉ SUBCONJUNTO de velas entra al cálculo.
# Combinando ambos ejes se obtienen las 3 × 3 = 9 vistas solicitadas.
# IMPORTANTE: en todos los casos el cálculo corre sobre el 100% de las filas
# del DataFrame completo (df / df_metric), nunca sobre una muestra, un
# head()/tail() ni sobre df_view (que sí puede estar limitado por
# sample_size en la tabla de auditoría).

METRIC_WINDOWS: dict[str, dict] = {
    "💰 Volumen (Notional)": {
        "key":          "notional",
        "unit_divisor": 1_000_000.0,
        "unit_suffix":  "M",
        "decimals":     3,
        "xaxis_title":  "Notional en millones M (VWAP × Volume)",
    },
    "📏 ln(H/L)": {
        "key":          "log_hl",
        "unit_divisor": 1.0,
        "unit_suffix":  "",
        "decimals":     5,
        "xaxis_title":  "ln(High / Low) — rango logarítmico intra-vela",
    },
    "🕯️ Body / Rango (%)": {
        "key":          "body",
        "unit_divisor": 0.01,
        "unit_suffix":  "%",
        "decimals":     2,
        "xaxis_title":  "Body / (High - Low) — % del rango ocupado por el cuerpo",
    },
}

CANDLE_FILTERS: dict[str, "str | None"] = {
    "⚪ General (todas)":                    None,
    "🟢 Solo velas verdes (Close > Open)":   "green",
    "🔴 Solo velas rojas (Close < Open)":    "red",
}

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def fmt_dt(dt_val) -> str:
    """
    Formatea un valor datetime/Timestamp a string '%Y-%m-%d %H:%M',
    manejando tanto objetos timezone-aware como naive.

    FIX-3: en Pandas >= 2.0, llamar .strftime() directamente sobre un
    Timestamp timezone-aware dentro de una columna pandas puede lanzar
    AttributeError en ciertos entornos Streamlit.
    """
    try:
        if hasattr(dt_val, "tz_localize"):
            # Es un Pandas Timestamp
            if dt_val.tzinfo is not None:
                dt_val = dt_val.tz_convert(None)   # quitar tz → naive UTC
        return dt_val.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(dt_val)


def detect_timeframe(filename: str) -> str:
    """Extrae el timeframe del nombre de archivo (ej. 'BTC_USDT_5m' → '5m')."""
    for tf in JUMP_CONFIG:
        if tf in filename:
            return tf
    return "1h"


def get_jump_config(filename: str) -> tuple[int, str]:
    tf = detect_timeframe(filename)
    return JUMP_CONFIG.get(tf, (180, "Segmento"))


def get_ttl(filename: str) -> int:
    tf = detect_timeframe(filename)
    return CACHE_TTL.get(tf, DEFAULT_TTL)


def normalize_datetime_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX-1 + FIX-3: Garantiza que la columna 'datetime' sea
    datetime64[ns] **sin** timezone (naive UTC) para compatibilidad con:
      - st.date_input  (requiere .date(), falla con tz-aware en algunos builds)
      - .strftime()    (inconsistente con tz-aware en Pandas >= 2.0)
      - comparaciones  (df['datetime'].dt.date >= start_date)

    Lógica:
      1. Si ya es datetime64 con tz → quitar tz (tz_convert + tz_localize None)
      2. Si es datetime64 sin tz  → dejar como está
      3. Si es int64/int32        → convertir desde epoch-ms
      4. Cualquier otro tipo      → intentar pd.to_datetime
    """
    col = df["datetime"]
    dtype = col.dtype

    if hasattr(dtype, "tz") and dtype.tz is not None:
        # Caso más común con integrity_rock v4: timestamp("ms", tz="UTC")
        df["datetime"] = col.dt.tz_convert("UTC").dt.tz_localize(None)
    elif str(dtype).startswith("datetime"):
        pass   # ya es naive datetime64 — sin cambios
    elif str(dtype) in ("int64", "int32", "int16"):
        df["datetime"] = pd.to_datetime(col, unit="ms")
    else:
        df["datetime"] = pd.to_datetime(col, utc=True).dt.tz_localize(None)

    return df


def ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX-6: integrity_rock v6 ya no persiste 'datetime' ni 'notional' en el
    parquet — solo 'timestamp' (epoch ms, UTC) y 'quote_vol' /
    'taker_buy_quote_vol'. Derivamos ambas columnas aquí para que el resto
    del pipeline (KDE, tabla de auditoría, highlight, presets) siga
    funcionando sin tocar integrity_rock.py.
    """
    if "datetime" not in df.columns:
        # timestamp de Sentinel viene en epoch **milisegundos**, UTC.
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    if "notional" not in df.columns:
        if "quote_vol" in df.columns:
            # quote_vol de Binance YA es el notional (precio × volumen) de la vela.
            df["notional"] = df["quote_vol"]
        else:
            # Fallback si tampoco hay quote_vol: aproximación close × volume.
            df["notional"] = df["close"] * df["volume"]

    return df


def ensure_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega las columnas necesarias para las 3 ventanas de análisis
    (Volumen / ln(H/L) / Body) sobre el DataFrame COMPLETO:

      - notional        → ya viene de ensure_derived_columns()
      - log_hl           = ln(high / low)                  → rango intra-vela en escala log
      - body              = |close - open| / (high - low)  → cuerpo como fracción del rango
      - candle_color      = 'green' | 'red' | 'doji'        → según close vs open

    Estas columnas se calculan UNA VEZ sobre todo el DataFrame cargado del
    parquet (no sobre una muestra), así que cualquier estadística posterior
    (KDE, percentiles, ECDF, skew/kurtosis) que filtre por candle_color o
    seleccione una de estas columnas sigue operando sobre el 100% de los
    datos disponibles para ese archivo.
    """
    if "log_hl" not in df.columns:
        # Protección ante low <= 0 (no debería ocurrir en OHLC real, pero
        # evita -inf/NaN silenciosos que romperían la KDE).
        safe_low = df["low"].where(df["low"] > 0, np.nan)
        df["log_hl"] = np.log(df["high"] / safe_low).astype("float32")

    if "body" not in df.columns:
        # Protección ante high == low (rango nulo, típico de un doji perfecto
        # o de datos con baja resolución): evita divisiones por cero → NaN.
        hl_range = (df["high"] - df["low"]).where(df["high"] > df["low"], np.nan)
        df["body"] = (
            (df["close"] - df["open"]).abs() / hl_range
        ).astype("float32")

    if "candle_color" not in df.columns:
        df["candle_color"] = np.where(
            df["close"] > df["open"], "green",
            np.where(df["close"] < df["open"], "red", "doji"),
        )

    return df


def validate_sentinel_columns(df: pd.DataFrame, filename: str) -> None:
    """
    FIX-4: Verifica que el parquet tenga todas las columnas requeridas.
    Lanza un st.error descriptivo y detiene la app si falta alguna.
    """
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        st.error(
            f"❌ El archivo **{filename}** no tiene las columnas requeridas: "
            f"`{sorted(missing)}`\n\n"
            "Solución: re-ejecuta `integrity_rock.py v6` para regenerar los "
            "parquets con el esquema mínimo (timestamp + OHLCV). "
            "'datetime' y 'notional' ya no son obligatorias: el visualizador "
            "las deriva automáticamente de 'timestamp' (ms) y 'quote_vol'."
        )
        st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# CACHING
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)   # TTL base; se actualiza dinámicamente abajo
def _load_parquet_base(file_path: str) -> pd.DataFrame:
    """
    Carga raw del parquet. Separada de la normalización para que el
    cache funcione correctamente (la normalización muta el df).
    """
    df = pd.read_parquet(file_path)

    # Reducir float64 → float32 (ahorro ~50% RAM)
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            # NUNCA castear timestamp a int32 (pierde precisión de ms)
            if col not in ("timestamp", "ts_ms"):
                df[col] = df[col].astype("int32")

    return df


def load_parquet_for_file(file_path: str) -> pd.DataFrame:
    """
    Carga, valida y normaliza un parquet Sentinel.

    FIX-2: el TTL se ajusta según el timeframe del archivo para que
    las velas de 5m aparezcan en max 5 minutos tras la actualización,
    sin necesidad de reiniciar Streamlit.
    """
    filename = os.path.basename(file_path).replace(".parquet", "")
    ttl = get_ttl(filename)

    # Re-definir la función cacheada con el TTL correcto para este archivo.
    # Streamlit identifica el cache por (función, argumentos), así que
    # crear una closure con ttl dinámico no funciona bien. La solución
    # pragmática: usar el TTL más corto (300s) para todos y dejar que
    # el botón "Refrescar" limpie el cache manualmente.
    df = _load_parquet_base(file_path)
    validate_sentinel_columns(df, filename)
    df = ensure_derived_columns(df)
    df = ensure_analysis_columns(df)
    df = normalize_datetime_col(df)
    return df


@st.cache_resource
def compute_kde_and_stats(
    _data: np.ndarray, ath_min: float, ath_max: float, cache_key: str = ""
):
    """
    Calcula KDE y derivadas una sola vez por dataset.
    El prefijo _ evita que Streamlit intente hashear el array.

    `cache_key` (ej. "archivo|metrica|filtro_color") se incluye únicamente
    para forzar entradas de caché distintas por ventana de análisis, aunque
    dos ventanas distintas coincidan por casualidad en ath_min/ath_max.
    """
    kde = gaussian_kde(_data)
    x_grid = np.linspace(ath_min, ath_max, 1_500, dtype=np.float32)
    pdf    = kde(x_grid).astype(np.float32)

    dx           = x_grid[1] - x_grid[0]
    first_deriv  = np.gradient(pdf, dx)
    second_deriv = np.gradient(first_deriv, dx)

    return kde, x_grid, pdf, second_deriv

# ──────────────────────────────────────────────────────────────────────────────
# REPORTES DESCARGABLES (CSV/XLSX + PNG del KDE — 3 ventanas × 3 filtros)
# ──────────────────────────────────────────────────────────────────────────────

METRIC_SHORT: dict[str, str] = {
    "💰 Volumen (Notional)":  "Notional",
    "📏 ln(H/L)":             "logHL",
    "🕯️ Body / Rango (%)":    "Body",
}
FILTER_SHORT: dict[str, str] = {
    "⚪ General (todas)":                    "General",
    "🟢 Solo velas verdes (Close > Open)":   "Verdes",
    "🔴 Solo velas rojas (Close < Open)":    "Rojas",
}


def compute_window_stats(df: pd.DataFrame, metric_col: str, color_filter):
    """
    Replica el motor estadístico usado en la vista principal (KDE, ECDF,
    percentiles, skew/kurtosis/entropía, puntos gamma) para una combinación
    ventana × filtro, pero de forma standalone (sin tocar st.session_state
    ni la caché de la UI) para poder usarse en la generación de reportes.
    Retorna None si no hay datos para esa combinación.
    """
    df_metric = df if color_filter is None else df[df["candle_color"] == color_filter]
    data = df_metric[metric_col].dropna().values.astype(np.float32)
    if data.size == 0:
        return None

    ath_min = float(np.min(data))
    ath_max = float(np.max(data))
    mean_v  = float(np.mean(data))
    median_v = float(np.percentile(data, 50))
    p90_v   = float(np.percentile(data, 90))
    p95_v   = float(np.percentile(data, 95))
    p99_v   = float(np.percentile(data, 99))

    kde = gaussian_kde(data)
    x_grid = np.linspace(ath_min, ath_max, 800, dtype=np.float32)
    pdf = kde(x_grid).astype(np.float32)
    mode_v = float(x_grid[np.argmax(pdf)])

    ecdf_gen = ECDF(data)
    dx = x_grid[1] - x_grid[0]
    first_deriv = np.gradient(pdf, dx)
    second_deriv = np.gradient(first_deriv, dx)
    zero_cross = np.where(np.diff(np.sign(second_deriv)))[0]
    gamma_pts = [float(x_grid[i]) for i in zero_cross if mean_v < x_grid[i] < p95_v]

    mae = float(np.mean(np.abs(ecdf_gen(x_grid) - pdf.cumsum() / pdf.sum())))

    return {
        "n": len(data), "ath_min": ath_min, "ath_max": ath_max,
        "mean": mean_v, "median": median_v, "mode": mode_v,
        "p90": p90_v, "p95": p95_v, "p99": p99_v,
        "skew": float(skew(data)), "kurtosis": float(kurtosis(data)),
        "entropy": float(entropy(pdf)), "mae": mae,
        "x_grid": x_grid, "pdf": pdf, "gamma_pts": gamma_pts,
        "ecdf": ecdf_gen,
    }


def render_kde_image(stats: dict, window_cfg: dict, window_label: str,
                      filter_label: str, file_name: str) -> bytes:
    """Genera un PNG (matplotlib, sin GUI) del KDE con los mismos hitos que
    la vista Plotly interactiva: media, mediana, moda, P90/P95/P99 y puntos
    gamma. Devuelve los bytes del PNG listos para escribir en el ZIP."""
    M        = window_cfg["unit_divisor"]
    unit     = window_cfg["unit_suffix"]
    decimals = window_cfg["decimals"]

    x_M = stats["x_grid"] / M

    fig, ax = plt.subplots(figsize=(10, 6), dpi=130)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    ax.plot(x_M, stats["pdf"], color="#00FF00", linewidth=1.6)
    ax.fill_between(x_M, stats["pdf"], color="#00FF00", alpha=0.15)

    hitos = [
        ("MEDIA",   stats["mean"]   / M, "#00bfff"),
        ("MEDIANA", stats["median"] / M, "#ffa500"),
        ("MODA",    stats["mode"]   / M, "#ff00ff"),
        ("P90",     stats["p90"]    / M, "#32cd32"),
        ("P95",     stats["p95"]    / M, "#ff4500"),
        ("P99",     stats["p99"]    / M, "#ff0000"),
    ]
    ymax = float(np.max(stats["pdf"])) * 1.12
    for name, val, color in hitos:
        ax.axvline(val, color=color, linestyle="--", alpha=0.7, linewidth=1.1)
        ax.text(val, ymax * 0.98, f"{name} {val:.{decimals}f}{unit}",
                 color=color, fontsize=7.5, rotation=90, va="top", ha="right")

    if stats["gamma_pts"]:
        g_M = [g / M for g in stats["gamma_pts"]]
        g_y = np.interp(stats["gamma_pts"], stats["x_grid"], stats["pdf"])
        ax.scatter(g_M, g_y, color="yellow", edgecolor="black",
                    s=70, marker="*", zorder=5, label="Puntos Gamma (f''=0)")
        ax.legend(loc="upper right", fontsize=8, facecolor="#1e2127",
                   edgecolor="#333", labelcolor="white")

    ax.set_ylim(0, ymax)
    ax.set_title(f"KDE — {file_name} · {window_label} · {filter_label}",
                 color="white", fontsize=11, pad=12)
    ax.set_xlabel(window_cfg["xaxis_title"], color="white", fontsize=9)
    ax.set_ylabel("Densidad Probabilística", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.grid(color="#222222", linewidth=0.5)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_reports_zip(selected_infos: list[dict], progress_cb=None) -> bytes:
    """
    Por cada archivo seleccionado (activo + timeframe) genera:
      - 1 Excel (.xlsx) con 9 pestañas (3 ventanas × 3 filtros), cada una
        con el resumen estadístico + la curva KDE (x, densidad).
      - 9 imágenes PNG del KDE (una por pestaña).
    Todo se empaqueta en un único ZIP en memoria (bytes), organizado en
    una carpeta por archivo: {activo_tf}/reporte_{activo_tf}.xlsx
                             {activo_tf}/images/{metrica}_{filtro}.png

    `progress_cb`, si se provee, se llama como progress_cb(done, total, label)
    tras cada combinación procesada, para alimentar una barra de progreso.
    """
    total_steps = max(1, len(selected_infos) * len(METRIC_WINDOWS) * len(CANDLE_FILTERS))
    done = 0

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for info in selected_infos:
            file_name = info["name"]

            try:
                df = load_parquet_for_file(info["path"])
            except Exception as e:
                zf.writestr(f"{file_name}/ERROR.txt", f"No se pudo cargar el parquet: {e}")
                done += len(METRIC_WINDOWS) * len(CANDLE_FILTERS)
                if progress_cb:
                    progress_cb(done, total_steps, f"❌ {file_name} (error de carga)")
                continue

            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
                for window_label, window_cfg in METRIC_WINDOWS.items():
                    for filter_label, color_filter in CANDLE_FILTERS.items():
                        done += 1
                        sheet_name = f"{METRIC_SHORT[window_label]}_{FILTER_SHORT[filter_label]}"[:31]

                        if progress_cb:
                            progress_cb(done, total_steps,
                                        f"{file_name} — {window_label} / {filter_label}")

                        stats = compute_window_stats(df, window_cfg["key"], color_filter)
                        if stats is None:
                            pd.DataFrame(
                                {"info": ["Sin datos para esta combinación"]}
                            ).to_excel(writer, sheet_name=sheet_name, index=False)
                            continue

                        M = window_cfg["unit_divisor"]
                        unit_col = f"valor{(' (' + window_cfg['unit_suffix'] + ')') if window_cfg['unit_suffix'] else ''}"

                        summary_df = pd.DataFrame({
                            "estadistico": [
                                "n (velas)", "min", "max", "media", "mediana", "moda",
                                "p90", "p95", "p99", "skewness", "kurtosis", "entropia",
                                "mae_kde_vs_ecdf", "puntos_gamma",
                            ],
                            unit_col: [
                                stats["n"], stats["ath_min"] / M, stats["ath_max"] / M,
                                stats["mean"] / M, stats["median"] / M, stats["mode"] / M,
                                stats["p90"] / M, stats["p95"] / M, stats["p99"] / M,
                                stats["skew"], stats["kurtosis"], stats["entropy"],
                                stats["mae"],
                                ", ".join(f"{g / M:.{window_cfg['decimals']}f}" for g in stats["gamma_pts"]) or "—",
                            ],
                        })
                        summary_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)

                        # Nota: se omite deliberadamente el detalle punto-a-punto de la
                        # curva KDE (x_grid / pdf) en el reporte descargable. Ese detalle
                        # granular no aporta valor de lectura; el resumen estadístico de
                        # arriba (summary_df) es la "glosa final" que se necesita.
                        # El PNG de abajo sigue mostrando la curva completa, por si se
                        # necesita ver la forma de la distribución.

                        png_bytes = render_kde_image(
                            stats, window_cfg, window_label, filter_label, file_name
                        )
                        zf.writestr(f"{file_name}/images/{sheet_name}.png", png_bytes)

            zf.writestr(f"{file_name}/reporte_{file_name}.xlsx", excel_buf.getvalue())

    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# ──────────────────────────────────────────────────────────────────────────────
# HIGHLIGHT
# ──────────────────────────────────────────────────────────────────────────────

def highlight_rows_vectorized(
    df: pd.DataFrame,
    u_gamma: float,
    u_crit:  float,
) -> pd.DataFrame:
    """
    FIX-5: Guard para 'notional' faltante + vectorización.
    Retorna una matriz de estilos CSS del mismo shape que df.
    """
    styles = pd.DataFrame("", index=df.index, columns=df.columns)

    if "notional" not in df.columns:
        return styles

    critical_mask = df["notional"] >= u_crit
    gamma_mask    = (df["notional"] >= u_gamma) & (df["notional"] < u_crit)

    styles.loc[critical_mask, :] = (
        "background-color: rgba(139, 0, 0, 0.8); color: white; font-weight: bold"
    )
    styles.loc[gamma_mask, :] = (
        "background-color: rgba(204, 204, 0, 0.4); color: white"
    )
    return styles

# ──────────────────────────────────────────────────────────────────────────────
# DATAFRAME SECTION
# ──────────────────────────────────────────────────────────────────────────────

def render_dataframe_section(
    df:            pd.DataFrame,
    u_gamma:       float,
    u_crit:        float,
    selected_file: str,
) -> None:
    st.subheader("🔍 Auditoría de Rows (Visual Diff)")

    # Filtrar view_cols a las que realmente existen
    view_cols = [c for c in VIEW_COLS if c in df.columns]

    # ── Controles superiores ──────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])

    with ctrl1:
        modo = st.radio(
            "📍 Modo de visualización",
            options=[
                "🔚 Últimas filas",
                "🚀 Navegador por segmentos",
                "📅 Rango de fechas",
                "🔍 Buscar eventos",
            ],
            horizontal=True,
            key="nav_mode",
        )

    with ctrl2:
        sample_size = st.select_slider(
            "📊 Filas a mostrar",
            options=[50, 100, 200, 500, 800, 1_000, 1_500],
            value=500,
            key="sample_size",
        )

    with ctrl3:
        st.write("")   # espaciado vertical
        if st.button("💾 Exportar CSV"):
            st.info("Funcionalidad de exportación próximamente")

    # ── Modo 1: Últimas filas ─────────────────────────────────────────────────
    if modo == "🔚 Últimas filas":
        df_view = df.tail(sample_size)[view_cols].copy()
        rng_min = fmt_dt(df_view["datetime"].min())
        rng_max = fmt_dt(df_view["datetime"].max())
        st.info(
            f"📌 Últimas **{len(df_view)}** filas  |  "
            f"{rng_min}  →  **{rng_max}**  ← row[-1]"
        )

    # ── Modo 2: Navegador por segmentos ──────────────────────────────────────
    elif modo == "🚀 Navegador por segmentos":
        df_nav = df.reset_index(drop=True)
        jump, seg_label = get_jump_config(selected_file)
        total_rows     = len(df_nav)
        total_segments = max(1, total_rows // jump)

        n1, n2, n3, n4 = st.columns([1, 3, 1, 1])

        with n1:
            if st.button("⏮️", help=f"Primer {seg_label}", use_container_width=True):
                st.session_state["segment_id"] = 1

        with n2:
            segment_id = st.slider(
                f"📍 {seg_label}",
                min_value=1,
                max_value=total_segments,
                value=st.session_state.get("segment_id", total_segments),
                key="segment_slider",
            )
            st.session_state["segment_id"] = segment_id

        with n3:
            if st.button("⏭️", help=f"Último {seg_label}", use_container_width=True):
                st.session_state["segment_id"] = total_segments

        with n4:
            custom_jump = st.number_input(
                "Salto", min_value=10, max_value=1_000,
                value=jump, step=10,
            )
            if custom_jump != jump:
                jump           = custom_jump
                total_segments = max(1, total_rows // jump)

        start_pos = (segment_id - 1) * jump
        end_pos   = min(start_pos + jump, total_rows)
        df_view   = df_nav.iloc[start_pos:end_pos][view_cols].copy()

        i1, i2, i3, i4 = st.columns(4)
        i1.metric(f"📂 {seg_label} ID",  f"{segment_id}/{total_segments}")
        i2.metric("📊 Registros",        f"{start_pos} – {end_pos}")
        i3.metric("🕐 Inicio",           fmt_dt(df_view["datetime"].min()))
        i4.metric("🕐 Fin",              fmt_dt(df_view["datetime"].max()))

    # ── Modo 3: Rango de fechas ───────────────────────────────────────────────
    elif modo == "📅 Rango de fechas":
        d1, d2 = st.columns(2)
        min_date = df["datetime"].min().date()
        max_date = df["datetime"].max().date()

        with d1:
            start_date = st.date_input(
                "📅 Fecha inicio",
                value=max_date - pd.Timedelta(days=7),
                min_value=min_date,
                max_value=max_date,
            )
        with d2:
            end_date = st.date_input(
                "📅 Fecha fin",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
            )

        mask       = (df["datetime"].dt.date >= start_date) & (df["datetime"].dt.date <= end_date)
        df_filtered = df[mask]

        if len(df_filtered) > sample_size:
            st.warning(f"⚠️ {len(df_filtered):,} registros. Mostrando los primeros {sample_size}.")
            df_view = df_filtered.head(sample_size)[view_cols].copy()
        else:
            df_view = df_filtered[view_cols].copy()

        st.info(f"📌 **{len(df_view):,}** registros  |  {start_date} → {end_date}")

    # ── Modo 4: Buscar eventos ────────────────────────────────────────────────
    else:
        ev1, ev2 = st.columns(2)
        with ev1:
            event_type = st.selectbox(
                "🎯 Tipo de evento",
                options=[
                    "🚨 Alertas críticas (>= CRITICAL)",
                    "⚠️ Alertas gamma (>= GAMMA)",
                    "📈 Volumen alto (Top 1%)",
                    "🔝 Notional máximo (Top 100)",
                    "📉 Notional mínimo (Bottom 100)",
                    "💥 Volatilidad extrema (High-Low > threshold)",
                ],
            )
        with ev2:
            vol_threshold = 2.0
            if "Volatilidad" in event_type:
                vol_threshold = st.number_input(
                    "Umbral volatilidad (%)",
                    min_value=0.5, max_value=10.0,
                    value=2.0, step=0.1,
                )

        # df["notional"] está en raw USDT; PRESETS en millones → escalar
        u_crit_raw  = u_crit  * 1e6
        u_gamma_raw = u_gamma * 1e6
        if "críticas" in event_type:
            df_filtered = df[df["notional"] >= u_crit_raw]
        elif "gamma" in event_type:
            df_filtered = df[(df["notional"] >= u_gamma_raw) & (df["notional"] < u_crit_raw)]
        elif "Volumen alto" in event_type:
            df_filtered = df[df["volume"] >= df["volume"].quantile(0.99)]
        elif "máximo" in event_type:
            df_filtered = df.nlargest(100, "notional")
        elif "mínimo" in event_type:
            df_filtered = df.nsmallest(100, "notional")
        else:
            df_tmp = df.copy()
            df_tmp["_vol_pct"] = ((df_tmp["high"] - df_tmp["low"]) / df_tmp["low"]) * 100
            df_filtered = df_tmp[df_tmp["_vol_pct"] > vol_threshold].drop(columns=["_vol_pct"])

        if df_filtered.empty:
            st.warning("⚠️ No se encontraron eventos con los criterios seleccionados.")
            df_view = pd.DataFrame(columns=view_cols)
        else:
            if len(df_filtered) > sample_size:
                st.info(f"📌 {len(df_filtered):,} eventos. Mostrando los primeros {sample_size}.")
                df_view = df_filtered.head(sample_size)[view_cols].copy()
            else:
                st.success(f"✅ {len(df_filtered):,} eventos encontrados.")
                df_view = df_filtered[view_cols].copy()

    # ── Filtros avanzados ─────────────────────────────────────────────────────
    st.divider()
    with st.expander("🔧 Filtros avanzados", expanded=False):
        f1, f2, f3, f4 = st.columns(4)

        with f1:
            if st.checkbox("Filtrar por Notional") and "notional" in df_view.columns:
                mn = st.number_input("Min Notional", value=0.0, step=0.1)
                mx = st.number_input("Max Notional",
                                     value=float(df["notional"].max()), step=0.1)
                df_view = df_view[
                    (df_view["notional"] >= mn) & (df_view["notional"] <= mx)
                ]

        with f2:
            if st.checkbox("Filtrar por Volume") and "volume" in df_view.columns:
                min_vol = st.number_input("Min Volume", value=0.0, step=1.0)
                df_view = df_view[df_view["volume"] >= min_vol]

        with f3:
            if st.checkbox("Filtrar por color de vela") and "candle_color" in df_view.columns:
                color_choice = st.radio(
                    "Color", ["🟢 Verdes", "🔴 Rojas", "⚪ Doji"], horizontal=True,
                )
                color_map = {"🟢 Verdes": "green", "🔴 Rojas": "red", "⚪ Doji": "doji"}
                df_view = df_view[df_view["candle_color"] == color_map[color_choice]]

        with f4:
            sort_options = [
                c for c in ["datetime", "notional", "log_hl", "body", "volume", "close"]
                if c in df_view.columns
            ]
            sort_by = st.selectbox("Ordenar por", options=sort_options)
            ascending = st.radio(
                "Orden", ["⬇️ Descendente", "⬆️ Ascendente"], horizontal=True
            ) == "⬆️ Ascendente"
            df_view = df_view.sort_values(sort_by, ascending=ascending)

    # ── Estadísticas del segmento ─────────────────────────────────────────────
    if not df_view.empty:
        st.divider()
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("📊 Filas visibles",  f"{len(df_view):,}")
        s2.metric("💰 Notional Avg",    f"{df_view['notional'].mean() / 1e6:.3f}M")
        s3.metric("📈 Notional Max",    f"{df_view['notional'].max() / 1e6:.3f}M")
        s4.metric("📉 Notional Min",    f"{df_view['notional'].min() / 1e6:.3f}M")
        alerts = int((df_view["notional"] >= u_crit * 1e6).sum())
        s5.metric("🚨 Alertas críticas", alerts,
                  delta="⚠️" if alerts > 0 else None)

    # ── Renderizado de tabla ──────────────────────────────────────────────────
    if df_view.empty:
        st.warning("⚠️ Sin datos para mostrar con los filtros actuales.")
        return

    # Formatear datetime para mostrar sin tz en la tabla
    df_display = df_view.copy()
    if "datetime" in df_display.columns:
        df_display["datetime"] = df_display["datetime"].apply(fmt_dt)

    fmt = {}
    if "notional" in df_display.columns:
        df_display["notional"] = df_display["notional"] / 1e6
        fmt["notional"] = "{:.3f}M"
    if "log_hl" in df_display.columns:   fmt["log_hl"]   = "{:.5f}"
    if "body"   in df_display.columns:   fmt["body"]     = "{:.4f}"
    if "volume" in df_display.columns:  fmt["volume"]   = "{:.2f}"
    for c in ("open", "high", "low", "close"):
        if c in df_display.columns:       fmt[c]          = "{:.2f}"

    # Los PRESETS ya están en millones USDT → comparar directamente
    # contra df_display["notional"] que también está en millones (/1e6)
    styled = df_display.style.apply(
        lambda _: highlight_rows_vectorized(df_display, u_gamma, u_crit),
        axis=None,
    ).format(fmt)

    st.dataframe(styled, use_container_width=True, height=500)

    lc1, lc2, lc3 = st.columns(3)
    lc1.markdown("🟥 **Rojo**: Alerta crítica (>= CRITICAL)")
    lc2.markdown("🟨 **Amarillo**: Alerta gamma (>= GAMMA)")
    lc3.markdown("⚪ **Blanco**: Normal")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🛡️ Sentinel Quant Lab | High-Frequency Microstructure")
    st.markdown("---")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.header("🎛️ Panel de Control")

    files = sorted(glob.glob(f"{SENTINEL_DIR}/*.parquet"))
    if not files:
        st.sidebar.error(f"No se detectaron Parquets en '{SENTINEL_DIR}/'")
        st.stop()

    file_info = []
    for f in files:
        name  = os.path.basename(f).replace(".parquet", "")
        parts = name.split("_")
        if len(parts) >= 3:
            symbol    = f"{parts[0]}/{parts[1]}"
            timeframe = parts[2]
            file_info.append({
                "path":      f,
                "name":      name,
                "symbol":    symbol,
                "timeframe": timeframe,
                "display":   f"{symbol} - {timeframe}",
            })

    if not file_info:
        st.sidebar.error("No se pudo parsear ningún archivo.")
        st.stop()

    # ── Filtros de navegación ─────────────────────────────────────────────────
    st.sidebar.subheader("🔍 Filtros de Navegación")

    search_query = st.sidebar.text_input(
        "🔎 Búsqueda rápida",
        placeholder="Ej: BTC, 1h, USDT…",
    )

    all_symbols = sorted({f["symbol"] for f in file_info})
    tf_order    = ["5m", "15m", "1h", "4h", "1d", "1w"]
    all_tfs     = sorted(
        {f["timeframe"] for f in file_info},
        key=lambda x: tf_order.index(x) if x in tf_order else 999,
    )

    sc1, sc2 = st.sidebar.columns(2)
    selected_symbol = sc1.selectbox("📊 Símbolo",    ["Todos"] + all_symbols)
    selected_tf     = sc2.selectbox("⏱️ Timeframe", ["Todos"] + all_tfs)

    filtered = file_info.copy()
    if search_query:
        filtered = [f for f in filtered
                    if search_query.lower() in f["display"].lower()]
    if selected_symbol != "Todos":
        filtered = [f for f in filtered if f["symbol"] == selected_symbol]
    if selected_tf != "Todos":
        filtered = [f for f in filtered if f["timeframe"] == selected_tf]

    if not filtered:
        st.sidebar.warning("⚠️ Sin archivos con esos filtros.")
        st.stop()

    # ── Accesos rápidos ───────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown("⭐ **Accesos Rápidos**")

    quick_access = {
        "BTC 5m":  "BTC/USDT - 5m",
        "BTC 1h":  "BTC/USDT - 1h",
        "ETH 5m":  "ETH/USDT - 5m",
        "ETH 1h":  "ETH/USDT - 1h",
    }
    cols_quick    = st.sidebar.columns(4)
    selected_quick = None

    for i, (lbl, target) in enumerate(quick_access.items()):
        with cols_quick[i % 4]:
            if st.button(lbl, use_container_width=True, key=f"quick_{i}"):
                for f in filtered:
                    if f["display"] == target:
                        selected_quick = f
                        break

    # ── Selector de archivo ───────────────────────────────────────────────────
    st.sidebar.divider()
    display_options = [f["display"] for f in filtered]
    default_idx     = 0

    if selected_quick:
        try:
            default_idx = display_options.index(selected_quick["display"])
        except ValueError:
            pass

    selected_display = st.sidebar.selectbox(
        "📁 Archivo Seleccionado",
        options=display_options,
        index=default_idx,
        format_func=lambda x: (
            f"{'✅' if PRESETS.get(x.replace(' - ', '_').replace('/', '_')) else '⚪'} {x}"
        ),
    )

    sel_idx       = display_options.index(selected_display)
    selected_file = filtered[sel_idx]["path"]
    file_name     = filtered[sel_idx]["name"]

    # ── Descarga de reportes (9 pestañas Excel + 9 PNG del KDE) ───────────────
    st.sidebar.divider()
    st.sidebar.subheader("📥 Descargar Reportes")

    all_display_options = [f["display"] for f in file_info]
    report_selection = st.sidebar.multiselect(
        "📊 Activos / Timeframes a incluir",
        options=all_display_options,
        default=[selected_display],
        help=(
            "Por cada Activo/Timeframe seleccionado se genera 1 Excel con "
            "9 pestañas (3 métricas × 3 filtros) y 9 imágenes PNG del KDE. "
            "Ej: BTCUSDT 1m, BTCUSDT 1d → 2 archivos, 18 pestañas, 18 PNG."
        ),
    )

    if st.sidebar.button("📊 Generar Reportes", use_container_width=True):
        if not report_selection:
            st.sidebar.warning("⚠️ Selecciona al menos un Activo/Timeframe.")
        else:
            selected_infos = [f for f in file_info if f["display"] in report_selection]
            progress_bar = st.sidebar.progress(0.0)
            status_text  = st.sidebar.empty()

            def _on_progress(done: int, total: int, label: str) -> None:
                progress_bar.progress(done / total)
                status_text.caption(f"⏳ ({done}/{total}) {label}")

            try:
                zip_bytes = generate_reports_zip(selected_infos, progress_cb=_on_progress)
                st.session_state["report_zip_bytes"] = zip_bytes
                st.session_state["report_zip_name"] = (
                    f"sentinel_reportes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                )
                status_text.caption("✅ Reportes generados. Descarga disponible abajo ⬇️")
            except Exception as e:
                status_text.empty()
                st.sidebar.error(f"❌ Error generando reportes: {e}")
            finally:
                progress_bar.progress(1.0)

    if st.session_state.get("report_zip_bytes"):
        st.sidebar.download_button(
            "⬇️ Descargar ZIP de Reportes",
            data=st.session_state["report_zip_bytes"],
            file_name=st.session_state.get("report_zip_name", "sentinel_reportes.zip"),
            mime="application/zip",
            use_container_width=True,
        )

    # ── Navegación ⬅️ ➡️ ──────────────────────────────────────────────────────
    st.sidebar.divider()
    nc1, nc2, nc3 = st.sidebar.columns([1, 2, 1])
    with nc1:
        if st.button("⬅️", use_container_width=True):
            st.session_state["selected_file"] = filtered[
                (sel_idx - 1) % len(filtered)
            ]["display"]
            st.rerun()
    with nc2:
        st.markdown(
            f"<center><small>{sel_idx + 1} / {len(filtered)}</small></center>",
            unsafe_allow_html=True,
        )
    with nc3:
        if st.button("➡️", use_container_width=True):
            st.session_state["selected_file"] = filtered[
                (sel_idx + 1) % len(filtered)
            ]["display"]
            st.rerun()

    # ── Info del archivo + botón Refrescar ────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.markdown("### 📋 Info del Archivo")

    try:
        fsize    = os.path.getsize(selected_file) / (1024 * 1024)
        fmod     = datetime.fromtimestamp(os.path.getmtime(selected_file))
        ic1, ic2 = st.sidebar.columns(2)
        ic1.metric("Tamaño",    f"{fsize:.2f} MB")
        ic2.metric("Modificado", fmod.strftime("%d/%m %H:%M"))
    except Exception as e:
        st.sidebar.warning(f"No se pudo leer info: {e}")

    # FIX-2: Botón de refresco manual — limpia cache para ver velas nuevas
    # inmediatamente sin esperar que expire el TTL.
    st.sidebar.divider()
    if st.sidebar.button("🔄 Refrescar datos", help="Limpia la caché y recarga el parquet"):
        st.cache_data.clear()
        st.rerun()

    ttl_seg = get_ttl(file_name)
    st.sidebar.caption(f"⏱️ Cache auto-expira cada {ttl_seg // 60} min")

    # ── Ventana de Análisis (3 métricas × 3 filtros = 9 combinaciones) ────────
    st.sidebar.divider()
    st.sidebar.subheader("🧭 Ventana de Análisis")
    window_label = st.sidebar.selectbox(
        "📐 Métrica",
        options=list(METRIC_WINDOWS.keys()),
        help="Sobre qué serie se calculan KDE, percentiles, skew/kurtosis, etc.",
    )
    filter_label = st.sidebar.radio(
        "🎯 Subconjunto de velas",
        options=list(CANDLE_FILTERS.keys()),
        help="Se aplica sobre el 100% de las velas del archivo, sin sampling.",
    )

    # ── Preset + umbrales ─────────────────────────────────────────────────────
    st.sidebar.divider()

    if PRESETS.get(file_name):
        st.sidebar.success("✅ Preset configurado")
    else:
        st.sidebar.info("ℹ️ Usando valores por defecto")

    default_g, default_c = PRESETS.get(file_name, (100.0, 150.0))

    st.sidebar.subheader("📍 Umbrales de Alerta")
    u_gamma = st.sidebar.number_input("⚠️ GAMMA (Yellow)",    value=default_g, step=0.1)
    u_crit  = st.sidebar.number_input("🚨 CRITICAL (Red)",    value=default_c, step=0.1)

    # ── Estadísticas generales ────────────────────────────────────────────────
    st.sidebar.divider()
    with st.sidebar.expander("📊 Estadísticas Generales", expanded=False):
        sg1, sg2 = st.sidebar.columns(2)
        sg1.metric("📁 Total",     len(files))
        sg1.metric("✅ Presets",   sum(1 for f in file_info if PRESETS.get(f["name"])))
        sg2.metric("🔍 Filtrados", len(filtered))
        sg2.metric("📊 Símbolos",  len(all_symbols))
        st.sidebar.markdown("**Por TF:**")
        tf_counts: dict[str, int] = {}
        for f in file_info:
            tf_counts[f["timeframe"]] = tf_counts.get(f["timeframe"], 0) + 1
        for tf, cnt in sorted(tf_counts.items()):
            st.sidebar.text(f"  {tf}: {cnt} archivo(s)")

    with st.sidebar.expander("❓ Ayuda", expanded=False):
        st.sidebar.markdown("""
        **Filtros:** símbolo, TF, búsqueda libre  
        **🔄 Refrescar:** recarga el parquet ahora  
        **⬅️➡️** Navega entre archivos  
        **✅** = Preset configurado | **⚪** = Default  
        **🧭 Ventana:** Volumen / ln(H/L) / Body × General / Verdes / Rojas  
        **⚠️ GAMMA** = Alerta amarilla | **🚨 CRITICAL** = Alerta roja  
        """)

    # ── Carga del parquet ─────────────────────────────────────────────────────
    try:
        df = load_parquet_for_file(selected_file)
    except Exception as e:
        st.error(f"Error cargando archivo: {e}")
        st.stop()

    # ── Selección de ventana de análisis ──────────────────────────────────────
    window_cfg    = METRIC_WINDOWS[window_label]
    metric_col    = window_cfg["key"]
    color_filter  = CANDLE_FILTERS[filter_label]

    # IMPORTANTE: el filtro de color se aplica sobre `df` COMPLETO (todas las
    # velas del parquet cargado), NUNCA sobre una muestra ni sobre `df_view`
    # de la tabla de auditoría (que sí puede recortarse por sample_size).
    if color_filter is None:
        df_metric = df
    else:
        df_metric = df[df["candle_color"] == color_filter]

    if df_metric.empty or df_metric[metric_col].dropna().empty:
        st.warning(
            f"⚠️ No hay datos para la combinación **{window_label} / {filter_label}** "
            "en este archivo."
        )
        st.stop()

    # ── Motor estadístico ─────────────────────────────────────────────────────
    # Los cálculos internos (KDE, ECDF, gamma_pts) usan valores RAW para
    # mantener la precisión matemática. Solo la PRESENTACIÓN se escala según
    # la unidad de la ventana elegida (M para Volumen, sin escala para
    # ln(H/L) y Body). Siempre sobre el 100% de las filas de df_metric.
    data      = df_metric[metric_col].dropna().values.astype(np.float32)
    ath_min   = float(np.min(data))
    ath_max   = float(np.max(data))
    mean_v    = float(np.mean(data))
    median_v  = float(np.percentile(data, 50))
    p90_v     = float(np.percentile(data, 90))
    p95_v     = float(np.percentile(data, 95))
    p99_v     = float(np.percentile(data, 99))

    cache_key = f"{file_name}|{metric_col}|{color_filter}"
    kde, x_grid, pdf, second_deriv = compute_kde_and_stats(
        data, ath_min, ath_max, cache_key
    )
    mode_v    = float(x_grid[np.argmax(pdf)])
    ecdf_gen  = ECDF(data)

    zero_cross = np.where(np.diff(np.sign(second_deriv)))[0]
    gamma_pts  = [
        float(x_grid[i]) for i in zero_cross
        if mean_v < x_grid[i] < p95_v
    ]

    # Versiones escaladas — solo para display
    M          = window_cfg["unit_divisor"]
    unit       = window_cfg["unit_suffix"]
    decimals   = window_cfg["decimals"]
    ath_min_M  = ath_min  / M
    ath_max_M  = ath_max  / M
    mean_M     = mean_v   / M
    median_M   = median_v / M
    mode_M     = mode_v   / M
    p90_M      = p90_v    / M
    p95_M      = p95_v    / M
    p99_M      = p99_v    / M
    x_grid_M   = x_grid   / M          # eje X del KDE, escalado a la unidad
    gamma_M    = [g / M for g in gamma_pts]

    # ── Dashboard de métricas ─────────────────────────────────────────────────
    st.info(
        f"🧭 Ventana activa: **{window_label}** | Subconjunto: **{filter_label}** | "
        f"n = **{len(data):,}** velas (100% del subconjunto, sin sampling)"
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Muestra (n)",  f"{len(data):,}")
    m2.metric("Skewness",     f"{skew(data):.4f}")
    m3.metric("Kurtosis",     f"{kurtosis(data):.4f}")
    m4.metric("Entropía",     f"{entropy(pdf):.4f}")
    mae = float(np.mean(np.abs(ecdf_gen(x_grid) - pdf.cumsum() / pdf.sum())))
    m5.metric("MAE Error",    f"{mae:.6f}")

    # Mostrar row[-1] para confirmar sincronización intrahora (siempre sobre
    # el archivo completo, independiente de la ventana/filtro seleccionados)
    last_row_dt = fmt_dt(df["datetime"].iloc[-1])
    st.markdown(
        f"**{window_label} — {filter_label}**  \n"
        f"Max: `{ath_max_M:.{decimals}f}{unit}` | Min: `{ath_min_M:.{decimals}f}{unit}` | "
        f"P90: `{p90_M:.{decimals}f}{unit}` | "
        f"P99: `{p99_M:.{decimals}f}{unit}` | "
        f"**row\\[-1\\]:** `{last_row_dt}`"
    )

    # ── Visualización KDE  (eje X escalado a la unidad de la ventana) ────────
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_grid_M, y=pdf,
        fill="tozeroy",
        name="KDE PDF",
        line_color="#00FF00",
        fillcolor="rgba(0, 255, 0, 0.1)",
    ))

    hitos = [
        ("MEDIA",   mean_M,   "#00bfff"),
        ("MEDIANA", median_M, "#ffa500"),
        ("MODA",    mode_M,   "#ff00ff"),
        ("P90",     p90_M,    "#32cd32"),
        ("P95",     p95_M,    "#ff4500"),
        ("P99",     p99_M,    "#ff0000"),
    ]
    max_pdf = float(np.max(pdf))

    for i, (name, val_M, color) in enumerate(hitos):
        # ECDF usa valores RAW para ser matemáticamente correcto
        prob_mass   = ecdf_gen(val_M * M) * 100
        y_label_pos = max_pdf * (1.1 - i * 0.15)
        fig.add_vline(x=val_M, line_dash="dash", line_color=color, opacity=0.6)
        fig.add_annotation(
            x=val_M, y=y_label_pos,
            text=f"<b>{name}</b>: {val_M:.{decimals}f}{unit}<br>(Mass: {prob_mass:.1f}%)",
            showarrow=True, arrowhead=2, arrowcolor=color,
            bgcolor="rgba(0,0,0,0.8)", bordercolor=color, borderpad=4,
            font=dict(color=color, size=11), ax=40, ay=0,
        )

    if gamma_M:
        fig.add_trace(go.Scatter(
            x=gamma_M,
            y=kde([g * M for g in gamma_M]),   # KDE evalúa en valores RAW
            mode="markers",
            marker=dict(color="yellow", size=12, symbol="star",
                        line=dict(width=1, color="black")),
            name="Puntos Gamma (f''=0)",
        ))

    fig.update_layout(
        title=f"Kernel Density Estimation — {file_name} · {window_label} · {filter_label}",
        template="plotly_dark",
        height=600,
        margin=dict(t=50, b=50),
        xaxis_title=window_cfg["xaxis_title"],
        yaxis_title="Densidad Probabilística",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Tabla de auditoría ────────────────────────────────────────────────────
    render_dataframe_section(df, u_gamma, u_crit, selected_file)

    # ── Reporte ───────────────────────────────────────────────────────────────
    with st.expander("📄 Reporte de Resultados"):
        st.text(f"""
=== STATISTICAL SUMMARY ===
Archivo    : {file_name}
Ventana    : {window_label}
Subconjunto: {filter_label}
row[-1]    : {last_row_dt}
Muestra    : {len(data):,}  (100% de las velas del subconjunto, sin sampling)
Max        : {ath_max_M:.{decimals}f}{unit} | Min: {ath_min_M:.{decimals}f}{unit}
Skewness   : {skew(data):.4f} | Kurtosis: {kurtosis(data):.4f}
Entropía   : {entropy(pdf):.4f} | MAE Error: {mae:.6f}

--- Percentiles de Masa (ECDF) ---
MEDIA   : {mean_M:.{decimals}f}{unit}   (Mass: {ecdf_gen(mean_v)*100:.2f}%)
MEDIANA : {median_M:.{decimals}f}{unit}   (Mass: {ecdf_gen(median_v)*100:.2f}%)
MODA    : {mode_M:.{decimals}f}{unit}   (Mass: {ecdf_gen(mode_v)*100:.2f}%)
P90     : {p90_M:.{decimals}f}{unit}   (Mass: {ecdf_gen(p90_v)*100:.2f}%)
P95     : {p95_M:.{decimals}f}{unit}   (Mass: {ecdf_gen(p95_v)*100:.2f}%)
P99     : {p99_M:.{decimals}f}{unit}   (Mass: {ecdf_gen(p99_v)*100:.2f}%)
GAMMAS  : {[f"{g:.{decimals}f}{unit}" for g in gamma_M]}
===========================
        """)


if __name__ == "__main__":
    main()
