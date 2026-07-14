"""
PROanalizerVWAPbands — High-Frequency Microstructure Tool
=========================================================
VWAP Bands · KDE Statistical Engine · Gamma Detection · P99

Cambios en esta versión:
  - CACHE SYSTEM: caché inteligente por rango de fechas, procesamiento por
    chunks para datasets grandes, carpeta ./db_cache/ automática.
  - KDE MULTI-PEAK IMPROVEMENT: detección robusta de múltiples peaks con
    scipy.signal.find_peaks, prominencia adaptativa configurable (8% default),
    panel HTML renovado con sección dedicada por peak con colores sincronizados
    entre gráfico y panel, etiquetas cortas en el chart, reporte completo
    Peak Principal + Peaks Relativos + Valley + P99.
  - compute_segment_metrics(): parámetro anchored_from_start + soporte
    global-anchored VWAP (anclado desde el primer dato del parquet).
  - build_kde_figure() / render_kde_density_milestones(): Peak principal,
    peaks relativos con color sincronizado, Valley y P99 diferenciados.
  - Sidebar: opción "Use Global Anchored VWAP" + "Use Cache System".
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks          # MULTI-PEAK IMPROVEMENT
import glob
import os
import hashlib
import re
from datetime import datetime

# Aumentar límite global de Pandas Styler
pd.set_option("styler.render.max_elements", 1_000_000)

# CACHE SYSTEM: carpeta destino para archivos de caché
CACHE_DIR = "./db_cache"

# MULTI-PEAK IMPROVEMENT: paleta de colores para peaks relativos
# Índice 0 = Peak Principal (dorado), índice 1+ = peaks relativos
PEAK_COLORS = [
    "#ffb300",   # 0 → Peak Principal (ámbar/dorado)
    "#a78bfa",   # 1 → Violeta
    "#f97316",   # 2 → Naranja
    "#22d3ee",   # 3 → Cyan
    "#fb7185",   # 4 → Rosa
    "#84cc16",   # 5 → Lima
    "#e879f9",   # 6 → Magenta
    "#fbbf24",   # 7 → Amarillo
    "#34d399",   # 8 → Verde esmeralda
]

# MULTI-PEAK IMPROVEMENT: prominencia mínima como % del peak principal
# (ajustable; 0.08 = 8%)
PEAK_MIN_PROMINENCE_RATIO = 0.08


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PROanalizerVWAPbands",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --bg-primary:   #0e1117;
        --bg-secondary: #161b22;
        --bg-card:      #1a1f2e;
        --accent-green: #00FF00;
        --accent-cyan:  #00e5ff;
        --accent-amber: #ffb300;
        --accent-pink:  #ff4081;
        --text-primary: #e6edf3;
        --text-muted:   #8b949e;
        --border:       #30363d;
    }

    .stApp { background-color: var(--bg-primary); color: var(--text-primary); }
    .main .block-container { padding: 1.5rem 2rem; max-width: 1600px; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1520 0%, #0e1117 100%);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .block-container { padding: 1rem; }

    .pro-header {
        background: linear-gradient(135deg, #0d1520 0%, #111827 50%, #0e1117 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 1.2rem 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .pro-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent-green),
                                    var(--accent-cyan), transparent);
    }
    .pro-header h1 {
        font-family: 'Courier New', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        color: var(--accent-green);
        margin: 0;
        text-shadow: 0 0 20px rgba(0,255,0,0.4);
    }
    .pro-header p {
        color: var(--text-muted);
        font-size: 0.78rem;
        margin: 0.2rem 0 0 0;
        letter-spacing: 0.05em;
        font-family: 'Courier New', monospace;
    }

    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .metric-card .label {
        font-size: 0.7rem;
        color: var(--text-muted);
        font-family: 'Courier New', monospace;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .metric-card .value {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--accent-green);
        font-family: 'Courier New', monospace;
    }

    /* MULTI-PEAK IMPROVEMENT: panel rediseñado con soporte para sub-peaks */
    .kde-milestone-panel {
        background: #0a0f1a;
        border: 1px solid #1e3a5f;
        border-left: 4px solid #00e5ff;
        border-radius: 10px;
        padding: 1rem 1.4rem 1.2rem;
        margin: 0.6rem 0 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
    }
    .kde-milestone-title {
        color: #00e5ff;
        font-weight: 700;
        letter-spacing: 0.12em;
        font-size: 0.76rem;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #1e3a5f;
    }

    /* Fila principal genérica */
    .kde-row {
        display: flex;
        gap: 1.6rem;
        flex-wrap: wrap;
        align-items: center;
        margin-bottom: 0.35rem;
        padding: 0.35rem 0.5rem;
        border-radius: 5px;
    }
    .kde-row:hover { background: rgba(255,255,255,0.03); }

    /* Celda de dato */
    .kde-cell {
        color: #8b949e;
        font-size: 0.78rem;
        white-space: nowrap;
    }
    .kde-cell span {
        color: #e6edf3;
        font-weight: 600;
    }

    /* Badge de tipo de hito */
    .kde-badge {
        display: inline-block;
        padding: 0.1rem 0.55rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        border: 1px solid currentColor;
        margin-right: 0.3rem;
    }

    /* Bloque de peaks relativos */
    .kde-subpeaks-section {
        border-top: 1px solid #1e3a5f;
        margin-top: 0.7rem;
        padding-top: 0.7rem;
    }
    .kde-subpeak-title {
        font-weight: 700;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .kde-subpeak-row {
        display: flex;
        gap: 1.4rem;
        flex-wrap: wrap;
        align-items: center;
        margin-bottom: 0.35rem;
        padding: 0.3rem 0.6rem;
        border-left: 3px solid;
        border-radius: 0 5px 5px 0;
        background: rgba(255,255,255,0.025);
    }
    .kde-subpeak-row:hover { background: rgba(255,255,255,0.05); }

    /* Separadores de sección (valley, p99) */
    .kde-section-sep {
        border-top: 1px solid #1e3a5f;
        margin-top: 0.7rem;
        padding-top: 0.7rem;
    }

    /* CACHE SYSTEM: estilos para el panel de caché */
    .cache-panel {
        background: #0a1a0a;
        border: 1px solid #1a4a1a;
        border-left: 3px solid #00FF00;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.75rem;
    }
    .cache-panel.miss {
        background: #1a0a0a;
        border-color: #4a1a1a;
        border-left-color: #ffb300;
    }
    .cache-panel.info {
        background: #0a0a1a;
        border-color: #1a1a4a;
        border-left-color: #00e5ff;
    }

    .segment-panel {
        background: var(--bg-secondary);
        border: 1px solid #1e3a5f;
        border-left: 3px solid var(--accent-cyan);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .segment-title {
        font-family: 'Courier New', monospace;
        color: var(--accent-cyan);
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }

    .stButton > button {
        background: transparent;
        border: 1px solid var(--accent-green);
        color: var(--accent-green);
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        letter-spacing: 0.05em;
        border-radius: 6px;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: rgba(0,255,0,0.1);
        box-shadow: 0 0 12px rgba(0,255,0,0.3);
    }

    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        font-family: 'Courier New', monospace !important;
        font-size: 0.8rem !important;
        border-radius: 6px !important;
    }

    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        font-family: 'Courier New', monospace !important;
        color: var(--accent-amber) !important;
    }

    hr { border-color: var(--border) !important; }
    .stAlert { border-radius: 8px; font-family: 'Courier New', monospace; font-size: 0.8rem; }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        color: var(--text-muted);
    }
    .stTabs [aria-selected="true"] { color: var(--accent-green) !important; }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #484f58; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CACHE SYSTEM — Utilidades de nombre y gestión de caché
# ═══════════════════════════════════════════════════════════════════════════════

def _sanitize_name(s: str) -> str:
    """Elimina caracteres no-alfanuméricos del nombre base del archivo."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", s)


def build_cache_filename(
    source_path: str,
    start_str: str,
    end_str: str,
    use_global_anchor: bool,
) -> str:
    """
    CACHE SYSTEM: Construye el nombre canónico del archivo de caché.

    Formato: cache_<basename>_<start>_<end>[_global].parquet
    Ejemplo: cache_BTC_USDT_1m_2026-02-06_2026-04-05.parquet

    El hash MD5 de los parámetros se añade como sufijo corto para evitar
    colisiones cuando los timestamps tienen segundos distintos.
    """
    base   = _sanitize_name(os.path.splitext(os.path.basename(source_path))[0])
    s_date = _sanitize_name(start_str[:10]) if start_str else "start"
    e_date = _sanitize_name(end_str[:10])   if end_str   else "EOF"
    anchor = "_global" if use_global_anchor else ""

    key   = f"{source_path}|{start_str}|{end_str}|{use_global_anchor}"
    short = hashlib.md5(key.encode()).hexdigest()[:6]

    fname = f"cache_{base}_{s_date}_{e_date}{anchor}_{short}.parquet"
    return os.path.join(CACHE_DIR, fname)


def cache_exists(cache_path: str) -> bool:
    """CACHE SYSTEM: Verifica si el archivo de caché existe y no está vacío."""
    return os.path.isfile(cache_path) and os.path.getsize(cache_path) > 0


def get_cache_info(cache_path: str) -> dict:
    """CACHE SYSTEM: Retorna metadata del caché (tamaño, fecha, filas)."""
    if not cache_exists(cache_path):
        return {}
    size_mb = os.path.getsize(cache_path) / (1024 ** 2)
    mtime   = datetime.fromtimestamp(os.path.getmtime(cache_path))
    try:
        df_peek = pd.read_parquet(cache_path, columns=["VWAP"])
        rows    = len(df_peek)
    except Exception:
        rows = -1
    return {
        "size_mb": size_mb,
        "modified": mtime.strftime("%Y-%m-%d %H:%M:%S"),
        "rows": rows,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CACHE SYSTEM — Motor de construcción por chunks
# ═══════════════════════════════════════════════════════════════════════════════

def build_cache_file(
    source_path: str,
    start_str: str,
    end_str: str,
    use_global_anchor: bool,
    cache_path: str,
    chunk_rows: int = 100_000,
    progress_callback=None,
) -> pd.DataFrame:
    """
    CACHE SYSTEM: Construye el archivo de caché calculando todas las columnas
    running (VWAP, Sigma_W, Sigma_S, Ratio_Sigma_W, Ratio_Sigma_S, TP,
    Diferencial) procesando el parquet fuente por chunks para minimizar uso de RAM.
    """
    import pyarrow.parquet as pq

    os.makedirs(CACHE_DIR, exist_ok=True)

    pf       = pq.ParquetFile(source_path)
    n_groups = pf.metadata.num_row_groups

    cum_tp_vol    = 0.0
    cum_vol       = 0.0
    cum_dev_sq_w  = 0.0
    cum_diff_sq_s = 0.0
    n_count       = 0

    anchor_start = None
    if not use_global_anchor and start_str.strip():
        anchor_start = pd.to_datetime(start_str)

    start_ts = pd.to_datetime(start_str) if start_str.strip() else None
    end_ts   = pd.to_datetime(end_str)   if end_str.strip()   else None

    chunks_out    = []
    accumulating  = use_global_anchor
    total_rows_read = 0

    for group_idx in range(n_groups):
        batch = pf.read_row_group(group_idx).to_pandas()

        batch.columns = [c.strip().capitalize() for c in batch.columns]
        rename_map = {}
        for col in batch.columns:
            cl = col.lower()
            if cl in ("open", "o"):            rename_map[col] = "Open"
            elif cl in ("high", "h"):          rename_map[col] = "High"
            elif cl in ("low", "l"):           rename_map[col] = "Low"
            elif cl in ("close", "c"):         rename_map[col] = "Close"
            elif cl in ("volume", "v", "vol"): rename_map[col] = "Volume"
        batch.rename(columns=rename_map, inplace=True)

        if not isinstance(batch.index, pd.DatetimeIndex):
            for col in batch.columns:
                cl = col.lower()
                if "date" in cl or "time" in cl or "ts" in cl:
                    sample = pd.to_numeric(batch[col].iloc[0], errors="coerce")
                    unit   = "ms" if (sample and sample > 2e10) else "s"
                    batch[col] = pd.to_datetime(batch[col], unit=unit, errors="coerce")
                    batch.set_index(col, inplace=True)
                    break
        else:
            if batch.index[0].year == 1970:
                raw_int     = batch.index.astype(np.int64)
                unit        = "ms" if raw_int[0] > 2e10 else "s"
                batch.index = pd.to_datetime(raw_int, unit=unit)

        batch.sort_index(inplace=True)

        if not use_global_anchor and not accumulating and anchor_start is not None:
            if batch.index[-1] >= anchor_start:
                accumulating = True
                batch = batch.loc[batch.index >= anchor_start]

        if not accumulating:
            total_rows_read += len(batch)
            if progress_callback:
                progress_callback(total_rows_read)
            continue

        tp  = ((batch["High"].values + batch["Low"].values +
                batch["Close"].values) / 3.0)
        vol = batch["Volume"].values.astype(np.float64)
        n_chunk = len(batch)

        local_cumtpvol = np.cumsum(tp * vol)
        local_cumvol   = np.cumsum(vol)

        running_tp_vol   = cum_tp_vol + local_cumtpvol
        running_vol      = cum_vol    + local_cumvol
        running_vol_safe = np.where(running_vol == 0, 1.0, running_vol)

        vwap_chunk = running_tp_vol / running_vol_safe

        dev_sq_local    = vol * (tp - vwap_chunk) ** 2
        local_cumdevsq  = np.cumsum(dev_sq_local)
        running_devsq_w = cum_dev_sq_w + local_cumdevsq
        sigma_w_chunk   = np.sqrt(running_devsq_w / running_vol_safe)

        diff_sq_local   = (tp - vwap_chunk) ** 2
        local_cumdiffsq = np.cumsum(diff_sq_local)
        running_diffsqs = cum_diff_sq_s + local_cumdiffsq
        n_running       = np.arange(n_count + 1, n_count + n_chunk + 1, dtype=np.float64)
        sigma_s_chunk   = np.sqrt(running_diffsqs / n_running)

        cum_tp_vol    += local_cumtpvol[-1]
        cum_vol       += local_cumvol[-1]
        cum_dev_sq_w  += local_cumdevsq[-1]
        cum_diff_sq_s += local_cumdiffsq[-1]
        n_count       += n_chunk

        batch = batch.copy()
        batch["TP"]            = tp
        batch["VWAP"]          = vwap_chunk
        batch["Sigma_VWAP_W"]  = sigma_w_chunk
        batch["Sigma_VWAP_S"]  = sigma_s_chunk
        batch["Diferencial"]   = tp - vwap_chunk
        batch["Ratio_Sigma_W"] = np.where(
            sigma_w_chunk > 0, batch["Diferencial"] / sigma_w_chunk, 0.0
        )
        batch["Ratio_Sigma_S"] = np.where(
            sigma_s_chunk > 0, batch["Diferencial"] / sigma_s_chunk, 0.0
        )
        batch["Ratio_Sigma"]   = batch["Ratio_Sigma_W"]

        if start_ts is not None:
            batch = batch.loc[batch.index >= start_ts]
        if end_ts is not None:
            batch = batch.loc[batch.index <= end_ts]

        if len(batch) > 0:
            chunks_out.append(batch)

        total_rows_read += n_chunk
        if progress_callback:
            progress_callback(total_rows_read)

    if not chunks_out:
        raise ValueError("No data found in the specified range.")

    df_result = pd.concat(chunks_out, axis=0)
    df_result.sort_index(inplace=True)
    df_result.to_parquet(cache_path, index=True)
    return df_result


@st.cache_data(show_spinner=False)
def load_cache_parquet(cache_path: str) -> pd.DataFrame:
    """CACHE SYSTEM: Carga el parquet de caché precomputado (muy rápido)."""
    return pd.read_parquet(cache_path)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def scan_parquet_files(folder: str = "db_sentinel") -> list[str]:
    """Escanea recursivamente una carpeta buscando archivos .parquet."""
    patterns = [
        os.path.join(folder, "*.parquet"),
        os.path.join(folder, "**", "*.parquet"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    return sorted(set(files))


@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    """
    Carga y valida un archivo parquet.
    Normaliza nombres de columnas OHLCV y maneja timestamps epoch-ms/s.
    """
    df = pd.read_parquet(path)
    df.columns = [c.strip().capitalize() for c in df.columns]

    rename_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ("open", "o"):            rename_map[col] = "Open"
        elif cl in ("high", "h"):          rename_map[col] = "High"
        elif cl in ("low", "l"):           rename_map[col] = "Low"
        elif cl in ("close", "c"):         rename_map[col] = "Close"
        elif cl in ("volume", "v", "vol"): rename_map[col] = "Volume"
    df.rename(columns=rename_map, inplace=True)

    if not isinstance(df.index, pd.DatetimeIndex):
        time_col = None
        for col in df.columns:
            cl = col.lower()
            if "date" in cl or "time" in cl or "ts" in cl or col == "Timestamp":
                time_col = col
                break
        if time_col is not None:
            sample_val = pd.to_numeric(df[time_col].iloc[0], errors="coerce")
            unit = "ms" if (sample_val is not None and sample_val > 2e10) else "s"
            df[time_col] = pd.to_datetime(df[time_col], unit=unit, errors="coerce")
            df.set_index(time_col, inplace=True)
    else:
        if df.index[0].year == 1970:
            raw_int  = df.index.astype(np.int64)
            unit     = "ms" if raw_int[0] > 2e10 else "s"
            df.index = pd.to_datetime(raw_int, unit=unit)

    df.sort_index(inplace=True)
    return df


def get_volume_at_timestamp(df: pd.DataFrame, ts_str: str) -> str:
    """Devuelve el volumen de la vela más cercana a un timestamp dado."""
    if not ts_str.strip():
        return "—"
    try:
        ts  = pd.to_datetime(ts_str)
        idx = df.index.get_indexer([ts], method="nearest")[0]
        vol = df.iloc[idx]["Volume"]
        return f"{vol:,.0f}"
    except Exception:
        return "❓"


def validate_required_cols(df: pd.DataFrame) -> tuple[bool, str]:
    """Verifica que el DataFrame tenga las columnas OHLCV requeridas."""
    required = {"High", "Low", "Close", "Volume"}
    missing  = required - set(df.columns)
    if missing:
        return False, f"Missing columns: {missing}"
    return True, "OK"


# ═══════════════════════════════════════════════════════════════════════════════
# QUANT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_segment_metrics(
    df_seg: pd.DataFrame,
    anchored_from_start: bool = True,
    global_anchor_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Calcula VWAP running (anchored), Sigma Weighted, Sigma Simple,
    Diferencial y Ratio_Sigma para ambos métodos.

    Nota: Si el DataFrame ya contiene la columna VWAP (proveniente del caché),
    retorna el segmento tal cual sin recalcular (fast path).
    """
    # CACHE SYSTEM: fast path
    if "VWAP" in df_seg.columns and "Ratio_Sigma_W" in df_seg.columns:
        df_out = df_seg.copy()
        if "Ratio_Sigma" not in df_out.columns:
            df_out["Ratio_Sigma"] = df_out["Ratio_Sigma_W"]
        return df_out

    if global_anchor_df is not None:
        df_full  = global_anchor_df.copy()
        tp_full  = (df_full["High"].values + df_full["Low"].values +
                    df_full["Close"].values) / 3.0
        vol_full = df_full["Volume"].values.astype(np.float64)

        cum_tp_vol_full    = np.cumsum(tp_full * vol_full)
        cum_vol_full       = np.cumsum(vol_full)
        cum_vol_safe_full  = np.where(cum_vol_full == 0, 1.0, cum_vol_full)
        vwap_full          = cum_tp_vol_full / cum_vol_safe_full

        dev_sq_full        = vol_full * (tp_full - vwap_full) ** 2
        cum_dev_sq_full    = np.cumsum(dev_sq_full)
        sigma_w_full       = np.sqrt(cum_dev_sq_full / cum_vol_safe_full)

        diff_sq_s_full     = (tp_full - vwap_full) ** 2
        cum_diff_sq_s_full = np.cumsum(diff_sq_s_full)
        n_counts_full      = np.arange(1, len(df_full) + 1, dtype=np.float64)
        sigma_s_full       = np.sqrt(cum_diff_sq_s_full / n_counts_full)

        df_full["TP"]           = tp_full
        df_full["VWAP"]         = vwap_full
        df_full["Sigma_VWAP_W"] = sigma_w_full
        df_full["Sigma_VWAP_S"] = sigma_s_full

        seg_start = df_seg.index[0]
        seg_end   = df_seg.index[-1]
        df_out    = df_full.loc[seg_start:seg_end].copy()

        df_out["Diferencial"]   = df_out["TP"] - df_out["VWAP"]
        df_out["Ratio_Sigma_W"] = np.where(
            df_out["Sigma_VWAP_W"] > 0,
            df_out["Diferencial"] / df_out["Sigma_VWAP_W"], 0.0
        )
        df_out["Ratio_Sigma_S"] = np.where(
            df_out["Sigma_VWAP_S"] > 0,
            df_out["Diferencial"] / df_out["Sigma_VWAP_S"], 0.0
        )
        df_out["Ratio_Sigma"] = df_out["Ratio_Sigma_W"]
        return df_out

    df  = df_seg.copy()
    tp  = (df["High"].values + df["Low"].values + df["Close"].values) / 3.0
    vol = df["Volume"].values.astype(np.float64)

    cum_tp_vol   = np.cumsum(tp * vol)
    cum_vol      = np.cumsum(vol)
    cum_vol_safe = np.where(cum_vol == 0, 1.0, cum_vol)
    vwap         = cum_tp_vol / cum_vol_safe

    dev_sq       = vol * (tp - vwap) ** 2
    cum_dev_sq   = np.cumsum(dev_sq)
    sigma_vwap_w = np.sqrt(cum_dev_sq / cum_vol_safe)

    diff_sq_s     = (tp - vwap) ** 2
    cum_diff_sq_s = np.cumsum(diff_sq_s)
    n_counts      = np.arange(1, len(df) + 1, dtype=np.float64)
    sigma_vwap_s  = np.sqrt(cum_diff_sq_s / n_counts)

    df["TP"]            = tp
    df["VWAP"]          = vwap
    df["Diferencial"]   = tp - vwap
    df["Sigma_VWAP_W"]  = sigma_vwap_w
    df["Ratio_Sigma_W"] = np.where(
        sigma_vwap_w > 0, df["Diferencial"] / sigma_vwap_w, 0.0
    )
    df["Sigma_VWAP_S"]  = sigma_vwap_s
    df["Ratio_Sigma_S"] = np.where(
        sigma_vwap_s > 0, df["Diferencial"] / sigma_vwap_s, 0.0
    )
    df["Ratio_Sigma"] = df["Ratio_Sigma_W"]
    return df


def slice_segment(df: pd.DataFrame, start_str: str, end_str: str) -> pd.DataFrame:
    """Recorta el DataFrame por timestamps ISO (inclusivo). end vacío → último dato."""
    start = pd.to_datetime(start_str)
    if end_str.strip():
        end = pd.to_datetime(end_str)
        return df.loc[start:end]
    return df.loc[start:]


# ═══════════════════════════════════════════════════════════════════════════════
# KDE STATISTICAL ENGINE — MULTI-PEAK IMPROVEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def ecdf_at(data: np.ndarray, x: float) -> float:
    """ECDF empírica: P(X ≤ x)."""
    return float((data <= x).mean())


def compute_kde_stats(
    data: np.ndarray,
    min_prominence_ratio: float = PEAK_MIN_PROMINENCE_RATIO,
) -> dict:
    """
    Estadísticas descriptivas completas + KDE para un array 1-D.

    # MULTI-PEAK IMPROVEMENT
    Detecta múltiples peaks locales con scipy.signal.find_peaks usando:
      - Prominencia adaptativa: min_prominence_ratio * peak_principal_density.
        Configurable vía parámetro (default 8%). Filtra peaks de ruido sin
        ocultar modos secundarios reales.
      - Distancia mínima dinámica: 2% del ancho del grid (evita peaks gemelos
        causados por el ancho de banda de la KDE).
      - Garantía de inclusión: el peak principal siempre está en la lista
        aunque find_peaks no lo retorne (borde del array u otros casos edge).
      - Ordenado por densidad descendente, is_main=True solo para el máximo.

    Retorna en "peaks_info":
        [{"x": float, "y": float, "mass": float, "is_main": bool}, ...]
        ordenado de mayor a menor densidad.

    Compatibilidad legacy: "kde_peak_x/y/mass" = peak principal.
    """
    data = data[np.isfinite(data)]
    n    = len(data)
    if n < 5:
        return {}

    mean_val   = float(np.mean(data))
    median_val = float(np.median(data))
    skew_val   = float(stats.skew(data))
    kurt_val   = float(stats.kurtosis(data))

    counts, _ = np.histogram(data, bins=min(50, n // 5))
    probs      = counts / counts.sum()
    probs      = probs[probs > 0]
    entropy    = float(-np.sum(probs * np.log(probs)))

    p95_val = float(np.percentile(data, 95))
    p99_val = float(np.percentile(data, 99))
    ath_max = float(np.max(data))
    ath_min = float(np.min(data))

    kde    = gaussian_kde(data, bw_method="scott")
    x_grid = np.linspace(ath_min - 0.5, ath_max + 0.5, 1024)
    pdf    = kde(x_grid)

    ecdf_vals = np.array([(data <= x).mean() for x in x_grid])
    dx        = x_grid[1] - x_grid[0]
    cdf_kde   = np.cumsum(pdf) * dx
    mae_kde   = float(np.mean(np.abs(cdf_kde - ecdf_vals)))

    # ── Peak principal (máximo absoluto) ─────────────────────────────────────
    mode_idx    = int(np.argmax(pdf))
    mode_val    = float(x_grid[mode_idx])
    main_peak_y = float(pdf[mode_idx])

    # ── MULTI-PEAK IMPROVEMENT: detección robusta con find_peaks ─────────────
    # Prominencia mínima adaptativa: evita detectar ruido de la KDE como picos
    # reales pero captura modos secundarios con al menos 8% de la densidad máxima.
    min_prominence = min_prominence_ratio * main_peak_y

    # Distancia mínima: ~2% del grid → evita picos gemelos artificiales del KDE
    min_distance = max(5, int(len(x_grid) * 0.02))

    peak_indices, _ = find_peaks(
        pdf,
        prominence=min_prominence,
        distance=min_distance,
    )

    # MULTI-PEAK IMPROVEMENT: garantizar que el peak principal esté siempre incluido
    if len(peak_indices) == 0 or mode_idx not in peak_indices:
        peak_indices = np.unique(np.append(peak_indices, mode_idx))

    # MULTI-PEAK IMPROVEMENT: construir lista enriquecida ordenada por densidad
    peaks_info = []
    for idx in peak_indices:
        px    = float(x_grid[idx])
        py    = float(pdf[idx])
        pmass = ecdf_at(data, px) * 100
        peaks_info.append({
            "x":       px,
            "y":       py,
            "mass":    pmass,
            "is_main": (int(idx) == mode_idx),
        })
    # Ordenar de mayor a menor densidad → el principal siempre es el primero
    peaks_info.sort(key=lambda p: p["y"], reverse=True)

    # Aliases legacy para compatibilidad
    kde_peak_x    = mode_val
    kde_peak_y    = main_peak_y
    kde_peak_mass = ecdf_at(data, mode_val) * 100

    # ── Puntos Gamma: inflexiones de la 2ª derivada en zona [mean, p95] ──────
    d2           = np.gradient(np.gradient(pdf, x_grid), x_grid)
    sign_changes = np.where(np.diff(np.sign(d2)))[0]
    gamma_mask   = ((x_grid[sign_changes] >= mean_val) &
                    (x_grid[sign_changes] <= p95_val))
    gamma_x      = x_grid[sign_changes][gamma_mask]
    gamma_y      = pdf[sign_changes][gamma_mask]

    # ── Valley: mínimo en zona central [mean, p95] ───────────────────────────
    central_mask = (x_grid >= mean_val) & (x_grid <= p95_val)
    if central_mask.sum() > 0:
        valley_idx      = int(np.argmin(pdf[central_mask]))
        full_idx        = np.where(central_mask)[0][valley_idx]
        kde_valley_x    = float(x_grid[full_idx])
        kde_valley_y    = float(pdf[full_idx])
        kde_valley_mass = ecdf_at(data, kde_valley_x) * 100
    else:
        kde_valley_x    = np.nan
        kde_valley_y    = np.nan
        kde_valley_mass = np.nan

    return {
        "n":               n,
        "mean":            mean_val,
        "median":          median_val,
        "mode":            mode_val,
        "p95":             p95_val,
        "p99":             p99_val,
        "skewness":        skew_val,
        "kurtosis":        kurt_val,
        "entropy":         entropy,
        "mae_kde":         mae_kde,
        "ath_max":         ath_max,
        "ath_min":         ath_min,
        "x_grid":          x_grid,
        "pdf":             pdf,
        "gamma_x":         gamma_x,
        "gamma_y":         gamma_y,
        "data":            data,
        # MULTI-PEAK IMPROVEMENT: lista completa ordenada
        "peaks_info":      peaks_info,
        # Legacy aliases
        "kde_peak_x":      kde_peak_x,
        "kde_peak_y":      kde_peak_y,
        "kde_peak_mass":   kde_peak_mass,
        # Valley
        "kde_valley_x":    kde_valley_x,
        "kde_valley_y":    kde_valley_y,
        "kde_valley_mass": kde_valley_mass,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTLY RENDERING
# ═══════════════════════════════════════════════════════════════════════════════

PLOTLY_DARK = dict(
    plot_bgcolor  = "#0e1117",
    paper_bgcolor = "#0e1117",
    font          = dict(color="#e6edf3", family="Courier New, monospace", size=11),
    gridcolor     = "#1e2738",
    zerolinecolor = "#1e2738",
)

HITO_COLORS = {
    "Mean":   "#00FF00",
    "Median": "#00e5ff",
    "Mode":   "#ffb300",
    "P95":    "#ff4081",
    "P99":    "#cc00ff",
}


def build_kde_figure(stats_dict: dict, title: str) -> go.Figure:
    """
    Construye la figura KDE con hitos Mean/Median/Mode/P95/P99,
    puntos Gamma, Peak principal y Peaks relativos.

    # MULTI-PEAK IMPROVEMENT
    - Peaks relativos marcados con marcadores circulares de colores
      sincronizados con PEAK_COLORS (misma paleta que el panel HTML).
    - Etiquetas cortas "P2", "P3", … para no saturar el gráfico.
    - Valley con marcador triangular apuntando hacia abajo.
    - Leyenda descriptiva con valor x de cada peak.
    """
    s    = stats_dict
    x    = s["x_grid"]
    pdf  = s["pdf"]
    data = s["data"]

    fig = go.Figure()

    # Curva KDE rellena
    fig.add_trace(go.Scatter(
        x=x, y=pdf,
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(0,255,0,0.08)",
        line=dict(color="#00FF00", width=2),
        name="KDE PDF",
        hovertemplate="x=%{x:.4f}<br>f(x)=%{y:.6f}<extra></extra>",
    ))

    # Hitos verticales: Mean, Median, Mode, P95, P99
    hitos      = [
        ("Mean",   s["mean"]),
        ("Median", s["median"]),
        ("Mode",   s["mode"]),
        ("P95",    s["p95"]),
        ("P99",    s["p99"]),
    ]
    y_max      = float(pdf.max())
    base_fracs = [1.05, 0.87, 0.68, 0.50, 0.32]
    ax_offsets = [58, -58, 62, -62, 58]

    for i, (name, val) in enumerate(hitos):
        color   = HITO_COLORS[name]
        mass    = ecdf_at(data, val) * 100
        label_y = y_max * base_fracs[i]

        fig.add_vline(
            x=val,
            line=dict(color=color, width=1.5, dash="dash"),
            opacity=0.8,
        )
        fig.add_annotation(
            x=val, y=label_y,
            text=f"<b>{name}</b><br>val: {val:+.4f}<br>prob: {mass:.1f}%",
            showarrow=True, arrowhead=2,
            arrowcolor=color, arrowwidth=1.2,
            ax=ax_offsets[i], ay=-20,
            font=dict(color=color, size=10, family="Courier New, monospace"),
            bordercolor=color, borderwidth=1,
            bgcolor="rgba(14,17,23,0.88)", borderpad=3,
        )

    # ── MULTI-PEAK IMPROVEMENT: separar peaks en principal y relativos ────────
    peaks_info = s.get("peaks_info", [])
    main_peaks = [p for p in peaks_info if p["is_main"]]
    sub_peaks  = [p for p in peaks_info if not p["is_main"]]

    # Peak Principal — diamante dorado con etiqueta "▲ P1"
    if main_peaks:
        pk  = main_peaks[0]
        col = PEAK_COLORS[0]
        fig.add_trace(go.Scatter(
            x=[pk["x"]], y=[pk["y"]],
            mode="markers+text",
            marker=dict(symbol="diamond", size=14, color=col,
                        line=dict(color="#fff", width=1.2)),
            text=["▲ P1"],
            textposition="top center",
            textfont=dict(color=col, size=10, family="Courier New, monospace"),
            name=f"P1 · Peak Principal (x={pk['x']:+.4f})",
            hovertemplate=(
                f"<b>Peak Principal</b><br>"
                f"x = {pk['x']:+.4f}<br>"
                f"f(x) = {pk['y']:.6f}<br>"
                f"masa = {pk['mass']:.1f}%<extra></extra>"
            ),
            showlegend=True,
        ))

    # MULTI-PEAK IMPROVEMENT: Peaks relativos — círculos con paleta PEAK_COLORS[1+]
    for j, pk in enumerate(sub_peaks):
        # Índice 1-based del color (0 reservado para el principal)
        col_idx  = (j % (len(PEAK_COLORS) - 1)) + 1
        color_sp = PEAK_COLORS[col_idx]
        label    = f"P{j + 2}"   # P2, P3, P4, …

        fig.add_trace(go.Scatter(
            x=[pk["x"]], y=[pk["y"]],
            mode="markers+text",
            marker=dict(symbol="circle", size=11, color=color_sp,
                        line=dict(color="#fff", width=0.9)),
            text=[f"▲ {label}"],
            textposition="top center",
            textfont=dict(color=color_sp, size=9, family="Courier New, monospace"),
            name=f"{label} · Peak Rel. (x={pk['x']:+.4f})",
            hovertemplate=(
                f"<b>Peak Relativo {j + 1}</b><br>"
                f"x = {pk['x']:+.4f}<br>"
                f"f(x) = {pk['y']:.6f}<br>"
                f"masa = {pk['mass']:.1f}%<br>"
                f"rel. al principal = {(pk['y'] / main_peaks[0]['y'] * 100) if main_peaks else 0:.1f}%"
                f"<extra></extra>"
            ),
            showlegend=True,
        ))

    # Valley — triángulo hacia abajo en cyan
    vl_x = s["kde_valley_x"]
    vl_y = s["kde_valley_y"]
    if not np.isnan(vl_x):
        fig.add_trace(go.Scatter(
            x=[vl_x], y=[vl_y],
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=13, color="#00e5ff",
                        line=dict(color="#fff", width=1)),
            text=["▼ VALLEY"],
            textposition="bottom center",
            textfont=dict(color="#00e5ff", size=10, family="Courier New, monospace"),
            name=f"Valley (x={vl_x:+.4f})",
            hovertemplate=(
                f"<b>Valley</b><br>"
                f"x = {vl_x:+.4f}<br>"
                f"f(x) = {vl_y:.6f}<extra></extra>"
            ),
            showlegend=True,
        ))

    # Puntos Gamma
    if len(s["gamma_x"]) > 0:
        fig.add_trace(go.Scatter(
            x=s["gamma_x"], y=s["gamma_y"],
            mode="markers",
            marker=dict(symbol="star", size=12, color="#ff4081",
                        line=dict(color="#ffffff", width=0.8)),
            name="Gamma Points (f″=0)",
            hovertemplate="γ x=%{x:.4f}<br>f(x)=%{y:.6f}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(color="#00FF00", size=13, family="Courier New, monospace"),
        ),
        xaxis=dict(
            title="Ratio_Sigma  (desviación normalizada en σ)",
            gridcolor=PLOTLY_DARK["gridcolor"],
            zerolinecolor=PLOTLY_DARK["zerolinecolor"],
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title="Densidad de Probabilidad",
            gridcolor=PLOTLY_DARK["gridcolor"],
            zerolinecolor=PLOTLY_DARK["zerolinecolor"],
            tickfont=dict(size=10),
        ),
        plot_bgcolor  = PLOTLY_DARK["plot_bgcolor"],
        paper_bgcolor = PLOTLY_DARK["paper_bgcolor"],
        font          = PLOTLY_DARK["font"],
        legend=dict(
            bgcolor="rgba(14,17,23,0.8)",
            bordercolor="#30363d",
            borderwidth=1,
        ),
        height=490,
        margin=dict(l=60, r=30, t=55, b=55),
    )
    return fig


def render_kde_density_milestones(s: dict) -> None:
    """
    # MULTI-PEAK IMPROVEMENT — Panel de hitos de densidad completamente rediseñado.

    Estructura visual:
    ┌─────────────────────────────────────────────────────┐
    │  🔬 Density Milestone Report                        │
    ├─────────────────────────────────────────────────────┤
    │  🏔  P1 · Peak Principal   x=…  f(x)=…  masa=…%   │
    ├─────────────────────────────────────────────────────┤
    │  🔎  Peaks Relativos detectados (N)                 │
    │      P2 · Peak Rel. 1      x=…  f(x)=…  masa=…%   │
    │      P3 · Peak Rel. 2      x=…  f(x)=…  masa=…%   │
    │      … (o mensaje "distribución unimodal")          │
    ├─────────────────────────────────────────────────────┤
    │  🕳  Valley (zona central)  x=…  f(x)=…  masa=…%  │
    ├─────────────────────────────────────────────────────┤
    │  📌  P99                    x=…  f(x)=…  masa=…%  │
    └─────────────────────────────────────────────────────┘

    Colores de cada peak sincronizados con PEAK_COLORS (misma paleta que el
    gráfico Plotly) para que el usuario pueda cruzar información visualmente.
    """
    def _fmt(v: float, d: int = 4) -> str:
        return f"{v:+.{d}f}" if not np.isnan(v) else "N/A"

    peaks_info = s.get("peaks_info", [])
    main_peaks = [p for p in peaks_info if p["is_main"]]
    sub_peaks  = [p for p in peaks_info if not p["is_main"]]

    # ── Datos del peak principal ──────────────────────────────────────────────
    if main_peaks:
        pk_x = main_peaks[0]["x"]
        pk_y = main_peaks[0]["y"]
        pk_m = main_peaks[0]["mass"]
    else:
        pk_x = s.get("kde_peak_x", np.nan)
        pk_y = s.get("kde_peak_y", np.nan)
        pk_m = s.get("kde_peak_mass", np.nan)

    # ── Valley ────────────────────────────────────────────────────────────────
    vl_x = s.get("kde_valley_x",    np.nan)
    vl_y = s.get("kde_valley_y",    np.nan)
    vl_m = s.get("kde_valley_mass", np.nan)

    # ── P99 ───────────────────────────────────────────────────────────────────
    p99         = s.get("p99", np.nan)
    p99_m       = ecdf_at(s["data"], p99) * 100 if not np.isnan(p99) else np.nan
    p99_density = np.nan
    if not np.isnan(p99):
        kde_obj     = gaussian_kde(s["data"], bw_method="scott")
        p99_density = float(kde_obj([p99])[0])

    # ── MULTI-PEAK IMPROVEMENT: HTML del Peak Principal ───────────────────────
    col_main = PEAK_COLORS[0]
    html_main = f"""
    <div class="kde-row">
        <div class="kde-cell">
            <span class="kde-badge" style="color:{col_main};border-color:{col_main};">P1</span>
            <b style="color:{col_main};">Peak Principal</b>
            &nbsp;·&nbsp; densidad máxima
        </div>
        <div class="kde-cell">x = <span>{_fmt(pk_x)}</span></div>
        <div class="kde-cell">f(x) = <span>{pk_y:.6f}</span></div>
        <div class="kde-cell">masa = <span>{pk_m:.1f}%</span></div>
    </div>"""

    # ── MULTI-PEAK IMPROVEMENT: HTML de peaks relativos ───────────────────────
    if sub_peaks:
        sub_rows = ""
        for j, pk in enumerate(sub_peaks):
            col_idx  = (j % (len(PEAK_COLORS) - 1)) + 1
            col_sp   = PEAK_COLORS[col_idx]
            label    = f"P{j + 2}"
            pct_main = (pk["y"] / pk_y * 100) if pk_y > 0 else 0.0

            sub_rows += f"""
            <div class="kde-subpeak-row" style="border-left-color:{col_sp};">
                <div class="kde-cell" style="min-width:120px;">
                    <span class="kde-badge" style="color:{col_sp};border-color:{col_sp};">
                        {label}
                    </span>
                    <b style="color:{col_sp};">Peak Rel. {j + 1}</b>
                </div>
                <div class="kde-cell">x = <span>{_fmt(pk['x'])}</span></div>
                <div class="kde-cell">f(x) = <span>{pk['y']:.6f}</span></div>
                <div class="kde-cell">masa = <span>{pk['mass']:.1f}%</span></div>
                <div class="kde-cell" style="font-size:0.7rem;color:#6b7280;">
                    {pct_main:.1f}% del principal
                </div>
            </div>"""

        html_subpeaks = f"""
        <div class="kde-subpeaks-section">
            <div class="kde-subpeak-title" style="color:#a78bfa;">
                🔎 Peaks Relativos detectados &nbsp;
                <span style="background:#1e1030;padding:0.1rem 0.5rem;
                             border-radius:4px;font-size:0.7rem;">
                    {len(sub_peaks)}
                </span>
            </div>
            {sub_rows}
        </div>"""
    else:
        html_subpeaks = """
        <div class="kde-subpeaks-section">
            <div class="kde-subpeak-title" style="color:#6b7280;">
                🔎 Peaks Relativos
            </div>
            <div class="kde-cell" style="padding-left:0.4rem;color:#6b7280;">
                Distribución unimodal — sin peaks relativos significativos
                <span style="font-size:0.68rem;">(prominencia &lt; 8% del principal)</span>
            </div>
        </div>"""

    # ── MULTI-PEAK IMPROVEMENT: HTML del Valley ───────────────────────────────
    if not np.isnan(vl_x):
        html_valley = f"""
        <div class="kde-section-sep">
            <div class="kde-row">
                <div class="kde-cell">
                    <span class="kde-badge" style="color:#00e5ff;border-color:#00e5ff;">V</span>
                    <b style="color:#00e5ff;">Valley</b>
                    &nbsp;·&nbsp; zona central [mean, P95]
                </div>
                <div class="kde-cell">x = <span>{_fmt(vl_x)}</span></div>
                <div class="kde-cell">f(x) = <span>{vl_y:.6f}</span></div>
                <div class="kde-cell">masa = <span>{vl_m:.1f}%</span></div>
            </div>
        </div>"""
    else:
        html_valley = """
        <div class="kde-section-sep">
            <div class="kde-cell" style="color:#6b7280;">
                Sin valley detectado en zona central
            </div>
        </div>"""

    # ── MULTI-PEAK IMPROVEMENT: HTML del P99 ─────────────────────────────────
    if not np.isnan(p99):
        html_p99 = f"""
        <div class="kde-section-sep">
            <div class="kde-row">
                <div class="kde-cell">
                    <span class="kde-badge" style="color:#cc00ff;border-color:#cc00ff;">
                        P99
                    </span>
                    <b style="color:#cc00ff;">Percentil 99</b>
                </div>
                <div class="kde-cell">x = <span>{_fmt(p99)}</span></div>
                <div class="kde-cell">f(x) = <span>{p99_density:.6f}</span></div>
                <div class="kde-cell">masa = <span>{p99_m:.1f}%</span></div>
            </div>
        </div>"""
    else:
        html_p99 = ""

    # ── Ensamblado final del panel ────────────────────────────────────────────
    st.markdown(f"""
    <div class="kde-milestone-panel">
        <div class="kde-milestone-title">🔬 Density Milestone Report</div>
        {html_main}
        {html_subpeaks}
        {html_valley}
        {html_p99}
    </div>
    """, unsafe_allow_html=True)


def build_vwap_chart(df_seg: pd.DataFrame, seg_label: str) -> go.Figure:
    """
    Gráfico de velas (OHLC) + VWAP + bandas σ-W (fila 1) +
    Ratio_Sigma_W (fila 2).
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.70, 0.30],
        vertical_spacing=0.04,
        subplot_titles=["VWAP + σ Bands (Weighted)", "Ratio Sigma W"],
    )

    open_col = "Open" if "Open" in df_seg.columns else "Close"
    fig.add_trace(go.Candlestick(
        x=df_seg.index,
        open=df_seg[open_col],
        high=df_seg["High"],
        low=df_seg["Low"],
        close=df_seg["Close"],
        name="OHLC",
        increasing_line_color="#00FF00",
        decreasing_line_color="#ff4081",
        increasing_fillcolor="rgba(0,255,0,0.15)",
        decreasing_fillcolor="rgba(255,64,129,0.15)",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_seg.index, y=df_seg["VWAP"],
        mode="lines",
        line=dict(color="#00e5ff", width=2),
        name="VWAP",
    ), row=1, col=1)

    band_cfg = [
        (1, "rgba(255,179,0,0.12)",  "#ffb300", "±1σ (W)"),
        (2, "rgba(255,64,129,0.08)", "#ff4081", "±2σ (W)"),
    ]
    for mult, fill_rgba, line_color, band_name in band_cfg:
        upper  = df_seg["VWAP"] + mult * df_seg["Sigma_VWAP_W"]
        lower  = df_seg["VWAP"] - mult * df_seg["Sigma_VWAP_W"]
        x_band = list(df_seg.index) + list(df_seg.index[::-1])
        y_band = list(upper) + list(lower[::-1])
        fig.add_trace(go.Scatter(
            x=x_band, y=y_band,
            fill="toself",
            fillcolor=fill_rgba,
            line=dict(color=line_color, width=0.8),
            name=band_name,
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_seg.index,
        y=df_seg["Ratio_Sigma_W"],
        mode="lines",
        line=dict(color="#a78bfa", width=1.5),
        name="Ratio_Sigma_W",
        fill="tozeroy",
        fillcolor="rgba(167,139,250,0.07)",
    ), row=2, col=1)

    fig.add_hline(y=0, row=2, col=1,
                  line=dict(color="#30363d", width=1, dash="dot"))

    fig.update_layout(
        title=dict(
            text=f"<b>VWAP Bands + Ratio_Sigma_W — {seg_label}</b>",
            font=dict(color="#00FF00", size=12, family="Courier New, monospace"),
        ),
        plot_bgcolor  = "#0e1117",
        paper_bgcolor = "#0e1117",
        font          = dict(color="#e6edf3", family="Courier New, monospace", size=11),
        legend=dict(
            bgcolor="rgba(14,17,23,0.8)",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(size=10),
        ),
        xaxis_rangeslider_visible=False,
        height=580,
        margin=dict(l=60, r=30, t=60, b=50),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#1e2738", zerolinecolor="#1e2738", showgrid=True)
    fig.update_yaxes(gridcolor="#1e2738", zerolinecolor="#1e2738", showgrid=True)
    fig.update_yaxes(title_text="Precio", row=1, col=1)
    fig.update_yaxes(title_text="Ratio σ", row=2, col=1)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# STATS DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def render_stats_cards(s: dict) -> None:
    """Renderiza tarjetas métricas en una fila horizontal."""
    metrics = [
        ("n",       f"{s['n']:,}",          "Samples"),
        ("Skew",    f"{s['skewness']:+.4f}", "Skewness"),
        ("Kurt",    f"{s['kurtosis']:+.4f}", "Excess Kurt."),
        ("Entropy", f"{s['entropy']:.4f}",   "Shannon H"),
        ("MAE KDE", f"{s['mae_kde']:.5f}",   "KDE vs ECDF"),
        ("ATH Max", f"{s['ath_max']:+.4f}",  "Ratio Max"),
        ("ATH Min", f"{s['ath_min']:+.4f}",  "Ratio Min"),
    ]
    cols = st.columns(len(metrics))
    for col, (label, val, sub) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{val}</div>
                <div class="label">{sub}</div>
            </div>""", unsafe_allow_html=True)


def build_print_summary(segments_stats: list, global_stats: dict) -> str:
    """Construye un resumen texto para copy-paste incluyendo W y S."""
    lines = [
        "=" * 80,
        "   PROanalizerVWAPbands — DUAL RESULTS SUMMARY (W vs S)",
        "=" * 80,
        f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    for i, item in enumerate(segments_stats):
        label, sw, ss = item if len(item) == 3 else (*item, item[1])

        n_peaks_w = len(sw.get("peaks_info", []))
        n_peaks_s = len(ss.get("peaks_info", []))

        lines += [
            f"─── SEGMENT {i+1}: {label} ───",
            f"{'METRIC':<22} | {'WEIGHTED (Anchored)':<22} | {'SIMPLE (Binance)':<22}",
            "-" * 80,
            f"{'n':<22} | {sw['n']:>22,} | {ss['n']:>22,}",
            f"{'Mean':<22} | {sw['mean']:>+22.6f} | {ss['mean']:>+22.6f}",
            f"{'Median (P50)':<22} | {sw['median']:>+22.6f} | {ss['median']:>+22.6f}",
            f"{'Mode':<22} | {sw['mode']:>+22.6f} | {ss['mode']:>+22.6f}",
            f"{'P95':<22} | {sw['p95']:>+22.6f} | {ss['p95']:>+22.6f}",
            f"{'P99':<22} | {sw['p99']:>+22.6f} | {ss['p99']:>+22.6f}",
            f"{'Skewness':<22} | {sw['skewness']:>+22.6f} | {ss['skewness']:>+22.6f}",
            f"{'Kurtosis':<22} | {sw['kurtosis']:>+22.6f} | {ss['kurtosis']:>+22.6f}",
            f"{'Entropy':<22} | {sw['entropy']:>22.6f} | {ss['entropy']:>22.6f}",
            f"{'ATH Max':<22} | {sw['ath_max']:>+22.6f} | {ss['ath_max']:>+22.6f}",
            f"{'ATH Min':<22} | {sw['ath_min']:>+22.6f} | {ss['ath_min']:>+22.6f}",
            f"{'Gamma Points':<22} | {len(sw['gamma_x']):>22} | {len(ss['gamma_x']):>22}",
            f"{'KDE Peaks (total)':<22} | {n_peaks_w:>22} | {n_peaks_s:>22}",
            f"{'KDE Peak x':<22} | {sw['kde_peak_x']:>+22.6f} | {ss['kde_peak_x']:>+22.6f}",
            f"{'KDE Peak mass%':<22} | {sw['kde_peak_mass']:>22.2f} | {ss['kde_peak_mass']:>22.2f}",
            "",
        ]

    if global_stats:
        sg        = global_stats
        n_peaks_g = len(sg.get("peaks_info", []))
        lines += [
            "─── GLOBAL (Weighted Combined) ───",
            f"  n             = {sg['n']:,}",
            f"  Mean          = {sg['mean']:+.6f}",
            f"  Median        = {sg['median']:+.6f}",
            f"  P95           = {sg['p95']:+.6f}",
            f"  P99           = {sg['p99']:+.6f}",
            f"  Skewness      = {sg['skewness']:+.6f}",
            f"  Kurtosis      = {sg['kurtosis']:+.6f}",
            f"  KDE Peaks     = {n_peaks_g}",
            f"  KDE Peak x    = {sg['kde_peak_x']:+.6f}",
            "",
        ]

    lines.append("=" * 80)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

if "segments" not in st.session_state:
    st.session_state.segments = [{"start": "", "end": ""}]


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:0.5rem 0 1rem;">
        <span style="font-family:'Courier New',monospace; font-size:1.1rem;
                     color:#00FF00; letter-spacing:0.15em; font-weight:700;">
            ◈ PROanalyzer
        </span><br>
        <span style="font-family:'Courier New',monospace; font-size:0.65rem;
                     color:#8b949e; letter-spacing:0.12em;">
            VWAP BANDS · MICROSTRUCTURE
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**📂 Data Source**")

    parquet_files = scan_parquet_files("db_sentinel")
    if parquet_files:
        selected_file = st.selectbox(
            "Select Parquet File",
            options=parquet_files,
            format_func=lambda p: os.path.basename(p),
            label_visibility="collapsed",
        )
    else:
        st.warning("No `.parquet` files found in `db_sentinel/`")
        st.markdown(
            "<small style='color:#8b949e;font-family:Courier New'>"
            "→ Place files in `db_sentinel/*.parquet`</small>",
            unsafe_allow_html=True,
        )
        selected_file = None

    manual_path = st.text_input(
        "Or enter path manually",
        placeholder="path/to/data.parquet",
        label_visibility="visible",
    )
    if manual_path.strip():
        selected_file = manual_path.strip()

    st.markdown("---")
    st.markdown("**⚙️ Analysis Options**")
    show_vwap_chart = st.checkbox("Show VWAP Chart per segment", value=True)
    show_global_kde = st.checkbox("Show Global KDE", value=True)
    kde_bw_method   = st.selectbox("KDE Bandwidth", ["scott", "silverman"], index=0)

    use_global_anchor = st.checkbox(
        "Use Global Anchored VWAP",
        value=False,
        help=(
            "Cuando está activo, el VWAP y las Sigmas se calculan desde el "
            "PRIMER dato del archivo parquet (en lugar de anclar al inicio "
            "de cada segmento)."
        ),
    )

    # MULTI-PEAK IMPROVEMENT: control de prominencia mínima en el sidebar
    st.markdown("---")
    st.markdown("**🔬 KDE Multi-Peak Settings**")
    peak_prominence_pct = st.slider(
        "Min. Peak Prominence (%)",
        min_value=3,
        max_value=25,
        value=int(PEAK_MIN_PROMINENCE_RATIO * 100),
        step=1,
        help=(
            "Prominencia mínima como % del peak principal para considerar un "
            "peak como significativo. Menor valor → más peaks detectados. "
            "Mayor valor → solo peaks muy prominentes."
        ),
    )
    peak_prominence_ratio = peak_prominence_pct / 100.0

    # CACHE SYSTEM: opciones de caché en el sidebar
    st.markdown("---")
    st.markdown("**💾 Cache System**")

    use_cache = st.checkbox(
        "Use Cache System",
        value=True,
        help=(
            "Guarda en ./db_cache/ un parquet precomputado por rango. "
            "La segunda carga es instantánea. Procesamiento por chunks "
            "para datasets grandes (optimizado para 8 GB RAM)."
        ),
    )

    force_recreate_cache = st.checkbox(
        "Force Recreate Cache",
        value=False,
        help="Regenera el caché aunque ya exista (útil si cambian los datos fuente).",
    )

    cache_chunk_rows = st.number_input(
        "Chunk size (rows)",
        min_value=10_000,
        max_value=500_000,
        value=100_000,
        step=10_000,
        help="Filas por chunk al construir el caché. Reduce si hay problemas de RAM.",
    )

    st.markdown("---")
    st.markdown("**🗃 Raw Data Display**")
    max_raw_rows = st.number_input(
        "Max rows to display in Raw Data",
        min_value=100,
        max_value=50_000,
        value=5_000,
        step=500,
        help="Styler con formato se aplica solo si el slice tiene ≤ 2000 filas.",
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-family:Courier New;font-size:0.65rem;color:#8b949e;text-align:center;'>"
        "RAM-optimised · NumPy vectorized · Chunk Cache<br>© PROanalyzer 2025"
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="pro-header">
    <h1>◈ PROanalizerVWAPbands</h1>
    <p>HIGH-FREQUENCY MICROSTRUCTURE TOOL · VWAP BANDS · KDE ENGINE · GAMMA DETECTION · P99 · MULTI-PEAK</p>
</div>
""", unsafe_allow_html=True)

df_main = None
if selected_file and os.path.exists(selected_file):
    try:
        with st.spinner("Loading parquet…"):
            df_main = load_parquet(selected_file)
        ok, msg = validate_required_cols(df_main)
        if not ok:
            st.error(f"⛔ {msg}")
            df_main = None
        else:
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"""<div class="metric-card">
                <div class="label">File</div>
                <div class="value" style="font-size:0.85rem">{os.path.basename(selected_file)}</div>
            </div>""", unsafe_allow_html=True)
            c2.markdown(f"""<div class="metric-card">
                <div class="label">Rows</div>
                <div class="value">{len(df_main):,}</div>
            </div>""", unsafe_allow_html=True)
            c3.markdown(f"""<div class="metric-card">
                <div class="label">Range</div>
                <div class="value" style="font-size:0.7rem">
                    {df_main.index[0].strftime('%Y-%m-%d %H:%M')}<br>
                    {df_main.index[-1].strftime('%Y-%m-%d %H:%M')}
                </div>
            </div>""", unsafe_allow_html=True)
            st.markdown("")
    except Exception as e:
        st.error(f"⛔ Failed to load file: {e}")
elif selected_file:
    st.warning(f"File not found: `{selected_file}`")


# ── Definición de segmentos ───────────────────────────────────────────────────
st.markdown("### 📐 Segment Ranges")

col_add, col_reset = st.columns([1, 5])
with col_add:
    if st.button("➕ Add Segment Range"):
        st.session_state.segments.append({"start": "", "end": ""})
with col_reset:
    if st.button("🗑 Reset Segments"):
        st.session_state.segments = [{"start": "", "end": ""}]

for i, seg in enumerate(st.session_state.segments):
    color_i = ["#00e5ff", "#00FF00", "#ffb300", "#ff4081", "#a78bfa"][i % 5]
    st.markdown(f"""
    <div class="segment-panel">
        <div class="segment-title" style="color:{color_i}">▸ SEGMENT {i+1}</div>
    </div>
    """, unsafe_allow_html=True)

    sc1, sc2, sc3, sc4, sc5 = st.columns([3, 2, 3, 2, 1])
    with sc1:
        start_val = st.text_input(
            f"Start (ISO) — Seg {i+1}",
            value=seg["start"],
            placeholder="2024-01-01 09:30:00",
            key=f"start_{i}",
        )
        st.session_state.segments[i]["start"] = start_val

    with sc2:
        vol_start = get_volume_at_timestamp(df_main, start_val) if df_main is not None else "—"
        st.markdown(f"""
        <div style="padding-top:1.6rem;">
            <div class="metric-card">
                <div class="label">Vol @ Start</div>
                <div class="value" style="font-size:0.9rem;color:{color_i}">{vol_start}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    with sc3:
        end_val = st.text_input(
            f"End (ISO) — Seg {i+1}  [empty = file end]",
            value=seg["end"],
            placeholder="2024-01-01 16:00:00  (optional)",
            key=f"end_{i}",
        )
        st.session_state.segments[i]["end"] = end_val

    with sc4:
        vol_end = (
            get_volume_at_timestamp(df_main, end_val)
            if (df_main is not None and end_val.strip()) else "—"
        )
        st.markdown(f"""
        <div style="padding-top:1.6rem;">
            <div class="metric-card">
                <div class="label">Vol @ End</div>
                <div class="value" style="font-size:0.9rem;color:{color_i}">{vol_end}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    with sc5:
        st.markdown("<div style='padding-top:1.6rem;'>", unsafe_allow_html=True)
        if len(st.session_state.segments) > 1:
            if st.button("✕", key=f"del_{i}", help="Remove segment"):
                st.session_state.segments.pop(i)
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # CACHE SYSTEM: mostrar estado del caché para este segmento
    if use_cache and selected_file and start_val.strip():
        cache_path = build_cache_filename(
            selected_file, start_val, end_val, use_global_anchor
        )
        if cache_exists(cache_path) and not force_recreate_cache:
            info = get_cache_info(cache_path)
            st.markdown(f"""
            <div class="cache-panel">
                ✅ <b>Cache HIT</b> &nbsp;|&nbsp;
                {os.path.basename(cache_path)}<br>
                <span style="color:#8b949e;">
                    {info.get('rows', '?'):,} rows · {info.get('size_mb', 0):.2f} MB ·
                    last built: {info.get('modified', '?')}
                </span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="cache-panel miss">
                ⚠️ <b>Cache MISS</b> &nbsp;|&nbsp;
                Se construirá al ejecutar el análisis<br>
                <span style="color:#8b949e;">
                    {os.path.basename(cache_path)}
                </span>
            </div>""", unsafe_allow_html=True)


# ── Botón Run ─────────────────────────────────────────────────────────────────
st.markdown("---")
run_col, _ = st.columns([1, 5])
with run_col:
    run_analysis = st.button("▶ RUN ANALYSIS", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if run_analysis:
    if df_main is None:
        st.error("⛔ No data loaded. Please select a valid parquet file.")
        st.stop()

    valid_segs = [s for s in st.session_state.segments if s["start"].strip()]
    if not valid_segs:
        st.warning("⚠️ Define at least one segment with a start timestamp.")
        st.stop()

    global_anchor_df = df_main if use_global_anchor else None

    if use_global_anchor:
        st.info(
            "🌐 **Global Anchored VWAP activo**: el VWAP y Sigmas se calculan "
            "desde el primer dato del archivo parquet."
        )

    all_ratio_sigma = []
    segments_stats  = []

    st.markdown("---")
    st.markdown("## 📊 Segment Analysis")

    for i, seg in enumerate(valid_segs):
        color_i = ["#00e5ff", "#00FF00", "#ffb300", "#ff4081", "#a78bfa"][i % 5]
        label   = f"Seg {i+1}: {seg['start']} → {seg['end'] or 'EOF'}"

        df_computed = None

        if use_cache and selected_file:
            cache_path = build_cache_filename(
                selected_file, seg["start"], seg["end"], use_global_anchor
            )

            if cache_exists(cache_path) and not force_recreate_cache:
                with st.spinner(f"⚡ Loading segment {i+1} from cache…"):
                    try:
                        df_computed = load_cache_parquet(cache_path)
                        st.markdown(f"""
                        <div class="cache-panel">
                            ⚡ <b>Loaded from cache</b> &nbsp;|&nbsp;
                            {len(df_computed):,} rows in milliseconds
                        </div>""", unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Cache load failed ({e}), falling back to live compute.")
                        df_computed = None

            if df_computed is None:
                progress_bar    = st.progress(0, text=f"Building cache for segment {i+1}…")
                total_rows_src  = len(df_main)

                def _update_progress(rows_done: int):
                    pct = min(int(rows_done / max(total_rows_src, 1) * 100), 99)
                    progress_bar.progress(pct, text=f"Processing chunks… {rows_done:,} rows")

                try:
                    with st.spinner(f"Building cache (chunk_rows={cache_chunk_rows:,})…"):
                        df_computed = build_cache_file(
                            source_path=selected_file,
                            start_str=seg["start"],
                            end_str=seg["end"],
                            use_global_anchor=use_global_anchor,
                            cache_path=cache_path,
                            chunk_rows=int(cache_chunk_rows),
                            progress_callback=_update_progress,
                        )
                    progress_bar.progress(100, text="✅ Cache built successfully!")
                    st.markdown(f"""
                    <div class="cache-panel">
                        💾 <b>Cache created</b> &nbsp;|&nbsp;
                        {len(df_computed):,} rows saved to
                        <code>{os.path.basename(cache_path)}</code>
                    </div>""", unsafe_allow_html=True)
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"⛔ Cache build failed: {e}. Falling back to standard compute.")
                    df_computed = None

        # Fallback: cálculo estándar sin caché
        if df_computed is None:
            try:
                df_seg = slice_segment(df_main, seg["start"], seg["end"])
            except Exception as e:
                st.error(f"⛔ Segment {i+1} slice error: {e}")
                continue

            if len(df_seg) < 5:
                st.warning(f"⚠️ Segment {i+1} has fewer than 5 rows — skipped.")
                continue

            with st.spinner(f"Computing segment {i+1}…"):
                df_computed = compute_segment_metrics(
                    df_seg,
                    anchored_from_start=True,
                    global_anchor_df=global_anchor_df,
                )

        if df_computed is None or len(df_computed) < 5:
            st.warning(f"⚠️ Segment {i+1} has insufficient data.")
            continue

        all_ratio_sigma.append(df_computed["Ratio_Sigma_W"].values)

        # MULTI-PEAK IMPROVEMENT: pasar peak_prominence_ratio del sidebar
        seg_stats_w = compute_kde_stats(
            df_computed["Ratio_Sigma_W"].dropna().values,
            min_prominence_ratio=peak_prominence_ratio,
        )
        seg_stats_s = compute_kde_stats(
            df_computed["Ratio_Sigma_S"].dropna().values,
            min_prominence_ratio=peak_prominence_ratio,
        )

        if not seg_stats_w or not seg_stats_s:
            st.warning(f"⚠️ Insufficient data in segment {i+1} for KDE analysis.")
            continue

        segments_stats.append((label, seg_stats_w, seg_stats_s))

        n_peaks_w = len(seg_stats_w.get("peaks_info", []))
        n_peaks_s = len(seg_stats_s.get("peaks_info", []))

        st.markdown(f"""
        <div style="border-left:3px solid {color_i}; padding:0.5rem 1rem; margin-bottom:0.8rem;
                    background:#0d1520; border-radius:0 8px 8px 0;">
            <span style="font-family:'Courier New',monospace; color:{color_i};
                         font-weight:700; letter-spacing:0.1em;">
                ▸ {label.upper()}
            </span>
            <span style="font-family:'Courier New',monospace; color:#8b949e;
                         font-size:0.75rem; margin-left:1rem;">
                {len(df_computed):,} candles
            </span>
            <span style="font-family:'Courier New',monospace; color:#a78bfa;
                         font-size:0.72rem; margin-left:1rem;">
                KDE peaks W:{n_peaks_w} · S:{n_peaks_s}
            </span>
        </div>
        """, unsafe_allow_html=True)

        render_stats_cards(seg_stats_w)
        st.markdown("")

        tab_kde_w, tab_kde_s, tab_vwap, tab_data_w, tab_data_s = st.tabs([
            "📈 KDE Weighted",
            "📊 KDE Simple",
            "🕯 VWAP Chart",
            "🗃 Raw Data (W)",
            "📂 Raw Data (S)",
        ])

        with tab_kde_w:
            fig_kde_w = build_kde_figure(
                seg_stats_w, f"Ratio_Sigma_W (Weighted) — {label}"
            )
            st.plotly_chart(fig_kde_w, use_container_width=True, key=f"kde_w_{i}")
            render_kde_density_milestones(seg_stats_w)

        with tab_kde_s:
            fig_kde_s = build_kde_figure(
                seg_stats_s, f"Ratio_Sigma_S (Simple) — {label}"
            )
            st.plotly_chart(fig_kde_s, use_container_width=True, key=f"kde_s_{i}")
            render_kde_density_milestones(seg_stats_s)

        with tab_vwap:
            if show_vwap_chart:
                fig_vwap = build_vwap_chart(df_computed, label)
                st.plotly_chart(fig_vwap, use_container_width=True, key=f"vwap_{i}")
            else:
                st.info("VWAP chart disabled in sidebar.")

        display_cols_w = ["TP", "VWAP", "Sigma_VWAP_W", "Diferencial", "Ratio_Sigma_W"]
        display_cols_s = ["TP", "VWAP", "Sigma_VWAP_S", "Diferencial", "Ratio_Sigma_S"]

        with tab_data_w:
            df_slice_w = df_computed[[c for c in display_cols_w if c in df_computed.columns]]
            n_total_w  = len(df_slice_w)
            if n_total_w > max_raw_rows:
                st.warning(
                    f"⚠️ Mostrando primeras {max_raw_rows:,} de {n_total_w:,} filas."
                )
                df_slice_w = df_slice_w.head(max_raw_rows)
            if len(df_slice_w) <= 2_000:
                st.dataframe(
                    df_slice_w.style.format("{:.6f}"),
                    use_container_width=True, height=320,
                )
            else:
                st.dataframe(df_slice_w.round(6), use_container_width=True, height=320)

        with tab_data_s:
            df_slice_s = df_computed[[c for c in display_cols_s if c in df_computed.columns]]
            n_total_s  = len(df_slice_s)
            if n_total_s > max_raw_rows:
                st.warning(
                    f"⚠️ Mostrando primeras {max_raw_rows:,} de {n_total_s:,} filas."
                )
                df_slice_s = df_slice_s.head(max_raw_rows)
            if len(df_slice_s) <= 2_000:
                st.dataframe(
                    df_slice_s.style.format("{:.6f}"),
                    use_container_width=True, height=320,
                )
            else:
                st.dataframe(df_slice_s.round(6), use_container_width=True, height=320)

        st.markdown("---")

    # ── KDE Global (todos los segmentos combinados) ───────────────────────────
    if show_global_kde and len(all_ratio_sigma) > 1:
        global_data  = np.concatenate(all_ratio_sigma)
        # MULTI-PEAK IMPROVEMENT: prominencia configurable también en global KDE
        global_stats = compute_kde_stats(
            global_data,
            min_prominence_ratio=peak_prominence_ratio,
        )

        if global_stats:
            n_peaks_g = len(global_stats.get("peaks_info", []))
            st.markdown(
                f"## 🌐 Global KDE — All Segments Combined (Weighted)"
                f"<span style='font-family:Courier New;color:#a78bfa;"
                f"font-size:0.8rem;margin-left:1rem;'>"
                f"peaks detected: {n_peaks_g}</span>",
                unsafe_allow_html=True,
            )
            render_stats_cards(global_stats)
            st.markdown("")
            fig_global = build_kde_figure(
                global_stats, "Global Ratio_Sigma KDE (Weighted)"
            )
            st.plotly_chart(fig_global, use_container_width=True, key="kde_global")
            render_kde_density_milestones(global_stats)
            st.markdown("---")
        else:
            global_stats = {}
    else:
        global_stats = {}

    # ── Resumen exportable ────────────────────────────────────────────────────
    if segments_stats:
        with st.expander("🖨 Print Results — Copy/Paste Summary", expanded=False):
            summary_text = build_print_summary(segments_stats, global_stats)
            st.code(summary_text, language="text")
            st.download_button(
                label="⬇ Download Summary (.txt)",
                data=summary_text,
                file_name=f"PROanalyzer_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

    # CACHE SYSTEM: panel de gestión de caché al final del análisis
    if use_cache:
        with st.expander("🗂 Cache Management", expanded=False):
            os.makedirs(CACHE_DIR, exist_ok=True)
            cache_files = glob.glob(os.path.join(CACHE_DIR, "cache_*.parquet"))
            if cache_files:
                st.markdown(
                    f"<div class='cache-panel info'>"
                    f"📁 <b>{len(cache_files)}</b> cache file(s) in "
                    f"<code>{CACHE_DIR}/</code></div>",
                    unsafe_allow_html=True,
                )
                rows = []
                for cf in sorted(cache_files):
                    sz = os.path.getsize(cf) / (1024 ** 2)
                    mt = datetime.fromtimestamp(os.path.getmtime(cf)).strftime("%Y-%m-%d %H:%M")
                    rows.append({
                        "File": os.path.basename(cf),
                        "Size (MB)": f"{sz:.2f}",
                        "Modified": mt,
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, height=200)

                if st.button("🗑 Delete ALL cache files", key="del_all_cache"):
                    for cf in cache_files:
                        try:
                            os.remove(cf)
                        except Exception:
                            pass
                    st.success("All cache files deleted. Reload to refresh.")
                    st.rerun()
            else:
                st.info("No cache files found yet.")


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

with st.expander("🧪 Generate Demo Parquet (for testing)", expanded=False):
    st.markdown(
        "<small style='font-family:Courier New;color:#8b949e;'>"
        "Crea un parquet OHLCV sintético en `db_sentinel/` para pruebas inmediatas."
        "</small>",
        unsafe_allow_html=True,
    )
    demo_rows = st.slider("Number of candles", 200, 2000, 500, 50)
    if st.button("Generate demo_data.parquet"):
        os.makedirs("db_sentinel", exist_ok=True)
        np.random.seed(42)
        n       = demo_rows
        idx     = pd.date_range("2024-01-02 09:30", periods=n, freq="1min")
        base    = 100.0
        ret     = np.random.normal(0, 0.002, n).cumsum()
        close_p = base * np.exp(ret)
        noise   = np.abs(np.random.normal(0, 0.5, n))
        demo_df = pd.DataFrame({
            "Open":   close_p * (1 + np.random.normal(0, 0.001, n)),
            "High":   close_p + noise,
            "Low":    close_p - noise,
            "Close":  close_p,
            "Volume": np.random.lognormal(10, 1, n).astype(int),
        }, index=idx)
        out_path = "db_sentinel/demo_data.parquet"
        demo_df.to_parquet(out_path)
        st.success(
            f"✅ Guardado en `{out_path}` — recarga la página para verlo en el selector."
        )