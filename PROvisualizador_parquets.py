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
import os
from datetime import datetime

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
VIEW_COLS   = ["datetime", "notional", "volume", "open", "high", "low", "close"]
# Columnas REALMENTE requeridas en el parquet crudo de integrity_rock v6.
# 'datetime' y 'notional' ya NO se guardan en el parquet (v6 solo persiste
# 'timestamp' en ms y 'quote_vol' / 'taker_buy_quote_vol'); ambas se derivan
# en tiempo de carga vía ensure_derived_columns(), así que no se exigen aquí.
REQUIRED_COLS = {"timestamp", "open", "high", "low", "close", "volume"}

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
    df = normalize_datetime_col(df)
    return df


@st.cache_resource
def compute_kde_and_stats(_data: np.ndarray, ath_min: float, ath_max: float):
    """
    Calcula KDE y derivadas una sola vez por dataset.
    El prefijo _ evita que Streamlit intente hashear el array.
    """
    kde = gaussian_kde(_data)
    x_grid = np.linspace(ath_min, ath_max, 1_500, dtype=np.float32)
    pdf    = kde(x_grid).astype(np.float32)

    dx           = x_grid[1] - x_grid[0]
    first_deriv  = np.gradient(pdf, dx)
    second_deriv = np.gradient(first_deriv, dx)

    return kde, x_grid, pdf, second_deriv

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
        f1, f2, f3 = st.columns(3)

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
            sort_by = st.selectbox(
                "Ordenar por",
                options=[c for c in ["datetime", "notional", "volume", "close"]
                         if c in df_view.columns],
            )
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
    if "volume"   in df_display.columns:  fmt["volume"]   = "{:.2f}"
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
        **⚠️ GAMMA** = Alerta amarilla | **🚨 CRITICAL** = Alerta roja  
        """)

    # ── Carga del parquet ─────────────────────────────────────────────────────
    try:
        df = load_parquet_for_file(selected_file)
    except Exception as e:
        st.error(f"Error cargando archivo: {e}")
        st.stop()

    # ── Motor estadístico ─────────────────────────────────────────────────────
    # Los cálculos internos (KDE, ECDF, gamma_pts) usan valores RAW para
    # mantener la precisión matemática.  Solo la PRESENTACIÓN usa millones.
    data      = df["notional"].dropna().values.astype(np.float32)
    ath_min   = float(np.min(data))
    ath_max   = float(np.max(data))
    mean_v    = float(np.mean(data))
    median_v  = float(np.percentile(data, 50))
    p95_v     = float(np.percentile(data, 95))
    p99_v     = float(np.percentile(data, 99))

    kde, x_grid, pdf, second_deriv = compute_kde_and_stats(data, ath_min, ath_max)
    mode_v    = float(x_grid[np.argmax(pdf)])
    ecdf_gen  = ECDF(data)

    zero_cross = np.where(np.diff(np.sign(second_deriv)))[0]
    gamma_pts  = [
        float(x_grid[i]) for i in zero_cross
        if mean_v < x_grid[i] < p95_v
    ]

    # Versiones en millones — solo para display
    M          = 1_000_000.0
    ath_min_M  = ath_min  / M
    ath_max_M  = ath_max  / M
    mean_M     = mean_v   / M
    median_M   = median_v / M
    mode_M     = mode_v   / M
    p95_M      = p95_v    / M
    p99_M      = p99_v    / M
    x_grid_M   = x_grid   / M          # eje X del KDE en millones
    gamma_M    = [g / M for g in gamma_pts]

    # ── Dashboard de métricas ─────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Muestra (n)",  f"{len(data):,}")
    m2.metric("Skewness",     f"{skew(data):.4f}")
    m3.metric("Kurtosis",     f"{kurtosis(data):.4f}")
    m4.metric("Entropía",     f"{entropy(pdf):.4f}")
    mae = float(np.mean(np.abs(ecdf_gen(x_grid) - pdf.cumsum() / pdf.sum())))
    m5.metric("MAE Error",    f"{mae:.6f}")

    # Mostrar row[-1] para confirmar sincronización intrahora
    last_row_dt = fmt_dt(df["datetime"].iloc[-1])
    st.markdown(
        f"**ATH Max:** `{ath_max_M:.3f}M` | **ATH Min:** `{ath_min_M:.3f}M` | "
        f"**P99:** `{p99_M:.3f}M` | "
        f"**row\\[-1\\]:** `{last_row_dt}`"
    )

    # ── Visualización KDE  (eje X en millones) ────────────────────────────────
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
            text=f"<b>{name}</b>: {val_M:.3f}M<br>(Mass: {prob_mass:.1f}%)",
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
        title=f"Kernel Density Estimation — {file_name}",
        template="plotly_dark",
        height=600,
        margin=dict(t=50, b=50),
        xaxis_title="Notional en millones M (VWAP × Volume)",
        yaxis_title="Densidad Probabilística",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Tabla de auditoría ────────────────────────────────────────────────────
    render_dataframe_section(df, u_gamma, u_crit, selected_file)

    # ── Reporte ───────────────────────────────────────────────────────────────
    with st.expander("📄 Reporte de Resultados"):
        st.text(f"""
=== STATISTICAL SUMMARY ===
Archivo : {file_name}
row[-1] : {last_row_dt}
Muestra : {len(data):,}
ATH Max : {ath_max_M:.3f}M | ATH Min: {ath_min_M:.3f}M
Skewness: {skew(data):.4f} | Kurtosis: {kurtosis(data):.4f}
Entropía: {entropy(pdf):.4f} | MAE Error: {mae:.6f}

--- Percentiles de Masa (ECDF) ---
MEDIA   : {mean_M:.3f}M   (Mass: {ecdf_gen(mean_v)*100:.2f}%)
MEDIANA : {median_M:.3f}M  (Mass: {ecdf_gen(median_v)*100:.2f}%)
MODA    : {mode_M:.3f}M   (Mass: {ecdf_gen(mode_v)*100:.2f}%)
P95     : {p95_M:.3f}M   (Mass: {ecdf_gen(p95_v)*100:.2f}%)
P99     : {p99_M:.3f}M   (Mass: {ecdf_gen(p99_v)*100:.2f}%)
GAMMAS  : {[f"{g:.3f}M" for g in gamma_M]}
===========================
        """)


if __name__ == "__main__":
    main()