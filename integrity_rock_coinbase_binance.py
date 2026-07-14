"""
integrity_rock.py  (v5 — Multi-Exchange Spot Edition)
======================================================
Quant-grade data pipeline para Binance USDT Perpetual, Binance Spot y Coinbase Spot.

Assets     : BTC
Timeframes : 5m, 1h, 4h, 1d
Start date : 2023-01-01 00:00:00 UTC

Cambios v5 vs v4
-----------------
[ADICIÓN]    MULTI-EXCHANGE CONFIG
             Diccionario SYNC_SOURCES define los 3 exchanges de forma declarativa.
             Agregar una fuente nueva = un dict entry, cero cambios en el motor.

[MODIFICACIÓN] JERARQUÍA DE DIRECTORIOS POR EXCHANGE
             data/[exchange_id]/[ASSET]_[TF].parquet    ← OHLCV raw
             db_sentinel/[exchange_id]/[ASSET]_[TF].parquet  ← Sentinel

[MODIFICACIÓN] raw_path / sentinel_path
             Reciben exchange_id como parámetro adicional para el sub-directorio.

[MODIFICACIÓN] main()
             Itera sobre SYNC_SOURCES: instancia un exchange ccxt por fuente,
             procesa todos los pares, cierra la conexión. Memory-safe.

[MANTENIDO]  Todo el motor de integridad de v4:
             live_open rodante, cursor delta +1ms, triple filtro vela cerrada,
             gap detection/fill quirúrgico, escritura atómica verificada,
             dual-write Sentinel, float32, gc.collect() post-concat.

Política de diff visual:
    # >>> [ACCIÓN] INICIO
    # <<< [ACCIÓN] FIN
"""

from __future__ import annotations

import gc
import logging
import math
import os
import random
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ==============================================================================
# >>> [ADICIÓN] MULTI-EXCHANGE CONFIG INICIO
# ==============================================================================

ASSETS     = ["BTC", "ETH"]
TIMEFRAMES = ["5m", "1h", "4h", "1d"]
START_DATE = "2023-01-01T00:00:00Z"

# >>> MODIFICACIÓN INICIO
# Coinbase API v3 solo acepta granularidades específicas — "4h" no existe.
# tf_whitelist: si está definida, solo se procesan esos TFs para ese exchange.
# batch_limit:  Coinbase hard-cap = 300 velas por request (Binance = 1000).
SYNC_SOURCES: list[dict] = [
    {
        "exchange_id":        "binanceusdm",
        "symbol_tpl":         "{asset}/USDT:USDT",
        "file_prefix":        "{asset}_USDT",
        "label":              "Binance USDM Perp",
        # concurrent_workers=1 → secuencial (Binance ya es muy rápido)
        "concurrent_workers": 1,
    },
    {
        "exchange_id":        "binance",
        "symbol_tpl":         "{asset}/USDT",
        "file_prefix":        "{asset}_USDT",
        "label":              "Binance Spot",
        "concurrent_workers": 1,
    },
    {
        "exchange_id":        "coinbase",
        "symbol_tpl":         "{asset}/USD",
        "file_prefix":        "{asset}_USD",
        "label":              "Coinbase Spot",
        # 4h no existe en la API v3 de Coinbase
        "tf_whitelist":       ["5m", "1h", "1d"],
        # Coinbase hard-cap: 300 velas por request
        "batch_limit":        300,
        # 2 threads: descarga BTC y ETH en paralelo para el mismo TF.
        # ccxt enableRateLimit=True ya maneja el throttling interno —
        # no necesitamos sleep manual adicional.
        # No usar >2 porque cada thread necesita su propia instancia ccxt.
        "concurrent_workers": 2,
    },
]
# <<< MODIFICACIÓN FIN

# <<< [ADICIÓN] MULTI-EXCHANGE CONFIG FIN


# ==============================================================================
# DIRECTORIOS BASE (raíz — los sub-directorios se crean por exchange_id)
# ==============================================================================
DATA_DIR     = Path("data")
SENTINEL_DIR = Path("db_sentinel")

# ==============================================================================
# PARÁMETROS OPERATIVOS (sin cambios respecto a v4)
# ==============================================================================
BATCH_SIZE     = 1_000
MAX_RETRIES    = 6
BACKOFF_BASE   = 2.0
BACKOFF_JITTER = 0.5

TF_MS: dict[str, int] = {
    "1m":  60_000,
    "5m":  300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h":  3_600_000,
    "4h":  14_400_000,
    "1d":  86_400_000,
}

# ==============================================================================
# SCHEMAS PYARROW (sin cambios respecto a v4)
# ==============================================================================
RAW_SCHEMA = pa.schema([
    pa.field("timestamp", pa.int64()),
    pa.field("open",      pa.float32()),
    pa.field("high",      pa.float32()),
    pa.field("low",       pa.float32()),
    pa.field("close",     pa.float32()),
    pa.field("volume",    pa.float32()),
])

SENTINEL_SCHEMA = pa.schema([
    pa.field("timestamp", pa.int64()),
    pa.field("datetime",  pa.timestamp("ms", tz="UTC")),
    pa.field("open",      pa.float32()),
    pa.field("high",      pa.float32()),
    pa.field("low",       pa.float32()),
    pa.field("close",     pa.float32()),
    pa.field("volume",    pa.float32()),
    pa.field("notional",  pa.float32()),
])

_OHLCV_COLS    = ["timestamp", "open", "high", "low", "close", "volume"]
_SENTINEL_COLS = ["timestamp", "datetime", "open", "high", "low",
                  "close", "volume", "notional"]

# ==============================================================================
# LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("integrity_rock_v5")

# ==============================================================================
# DATA STRUCTURES (sin cambios respecto a v4)
# ==============================================================================

@dataclass
class GapRange:
    """Bloque contiguo de velas faltantes entre dos timestamps conocidos."""
    gap_start_ms: int
    gap_end_ms:   int
    missing:      int

    @property
    def fetch_since(self) -> int:
        return self.gap_start_ms + 1

    def __str__(self) -> str:
        return (
            f"{ms_to_iso(self.gap_start_ms)} → {ms_to_iso(self.gap_end_ms)}"
            f"  ({self.missing} faltantes)"
        )


@dataclass
class SyncStats:
    """Métricas por par recolectadas durante la ejecución."""
    exchange_id:     str
    asset:           str
    tf:              str
    candles_before:  int   = 0
    candles_after:   int   = 0
    gaps_found:      int   = 0
    gaps_filled:     int   = 0
    gaps_remaining:  int   = 0
    raw_mb:          float = 0.0
    sentinel_mb:     float = 0.0
    errors:          list[str] = field(default_factory=list)


# ==============================================================================
# UTILITY HELPERS (sin cambios respecto a v4)
# ==============================================================================

def tf_to_ms(tf: str) -> int:
    if tf not in TF_MS:
        raise ValueError(f"Timeframe desconocido: {tf!r}")
    return TF_MS[tf]


def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1_000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def anchor_ms() -> int:
    """Epoch-ms del START_DATE (2023-01-01 00:00:00 UTC)."""
    return int(
        datetime.fromisoformat(START_DATE.replace("Z", "+00:00")).timestamp() * 1_000
    )


# >>> [MODIFICACIÓN] raw_path / sentinel_path INICIO
# v4: raw_path(asset, tf)  →  data/BTC_USDT_5m.parquet
# v5: raw_path(exchange_id, file_prefix, tf)  →  data/binance/BTC_USDT_5m.parquet
def raw_path(exchange_id: str, file_prefix: str, tf: str) -> Path:
    """Ruta canónica del parquet OHLCV raw (fuente de verdad por exchange)."""
    return DATA_DIR / exchange_id / f"{file_prefix}_{tf}.parquet"


def sentinel_path(exchange_id: str, file_prefix: str, tf: str) -> Path:
    """Ruta canónica del parquet Sentinel (por exchange, para el visualizador)."""
    return SENTINEL_DIR / exchange_id / f"{file_prefix}_{tf}.parquet"
# <<< [MODIFICACIÓN] raw_path / sentinel_path FIN


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = df["timestamp"].astype("int64")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype("float32")
    return df


def empty_ohlcv_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_OHLCV_COLS).astype({
        "timestamp": "int64",
        "open":      "float32",
        "high":      "float32",
        "low":       "float32",
        "close":     "float32",
        "volume":    "float32",
    })


def get_live_open(exchange: ccxt.Exchange, tf: str) -> int:
    """
    Open-time (epoch-ms) de la vela que se está formando AHORA.
    Usa exchange.milliseconds() — tiempo del servidor, NO reloj local.
    Se recalcula en cada batch del loop: nunca se cachea.
    """
    step = tf_to_ms(tf)
    return (exchange.milliseconds() // step) * step


# ==============================================================================
# NOTIONAL & SENTINEL DERIVATION (sin cambios respecto a v4)
# ==============================================================================

def compute_notional(df: pd.DataFrame) -> pd.Series:
    """notional = (high + low + close) / 3 * volume  (VWAP Typical Price)"""
    tp = (
        df["high"].astype("float64")
        + df["low"].astype("float64")
        + df["close"].astype("float64")
    ) / 3.0
    return (tp * df["volume"].astype("float64")).astype("float32")


def build_sentinel_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    sentinel = raw_df.copy()
    sentinel["notional"] = compute_notional(sentinel)
    sentinel["datetime"] = pd.to_datetime(
        sentinel["timestamp"], unit="ms", utc=True
    )
    sentinel = sentinel[_SENTINEL_COLS].copy()
    sentinel["timestamp"] = sentinel["timestamp"].astype("int64")
    sentinel["notional"]  = sentinel["notional"].astype("float32")
    for col in ("open", "high", "low", "close", "volume"):
        sentinel[col] = sentinel[col].astype("float32")
    return sentinel


# ==============================================================================
# >>> [MODIFICACIÓN] build_exchange → build_exchange_from_source INICIO
# v4: build_exchange() — hardcoded a binanceusdm
# v5: build_exchange_from_source(source) — instancia cualquier ccxt exchange
#     declarado en SYNC_SOURCES, con opciones mínimas y rate-limit habilitado.
# ==============================================================================

def build_exchange_from_source(source: dict) -> ccxt.Exchange:
    """
    Instancia el exchange ccxt correcto para cada entrada de SYNC_SOURCES.

    binanceusdm recibe la opción defaultType=future para que fetch_ohlcv
    opere sobre los futuros perpetuos y no sobre spot.
    binance y coinbase usan la configuración mínima con rate-limit.
    """
    eid = source["exchange_id"]

    if eid == "binanceusdm":
        exchange = ccxt.binanceusdm({
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,
            },
        })
    elif eid == "binance":
        exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"adjustForTimeDifference": True},
        })
    elif eid == "coinbase":
        exchange = ccxt.coinbase({
            "enableRateLimit": True,
        })
    else:
        # Fallback genérico para fuentes futuras en SYNC_SOURCES
        exchange_class = getattr(ccxt, eid)
        exchange = exchange_class({"enableRateLimit": True})

    exchange.load_markets()
    return exchange

# <<< [MODIFICACIÓN] build_exchange_from_source FIN


# ==============================================================================
# EXCHANGE / API LAYER (sin cambios respecto a v4)
# ==============================================================================

def fetch_ohlcv_with_retry(
    exchange: ccxt.Exchange,
    symbol:   str,
    tf:       str,
    since_ms: int,
    limit:    int = BATCH_SIZE,
) -> list[list]:
    """Fetch de un batch con exponential back-off + jitter."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return exchange.fetch_ohlcv(
                symbol, timeframe=tf, since=since_ms, limit=limit
            )
        except ccxt.RateLimitExceeded as exc:
            sleep_s = BACKOFF_BASE ** attempt + random.uniform(0, BACKOFF_JITTER)
            log.warning("Rate limit [%s]  intento %d/%d  %.1fs …",
                        exc, attempt, MAX_RETRIES, sleep_s)
            time.sleep(sleep_s)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as exc:
            sleep_s = BACKOFF_BASE ** attempt + random.uniform(0, BACKOFF_JITTER)
            log.warning("Network error [%s]  intento %d/%d  %.1fs …",
                        exc, attempt, MAX_RETRIES, sleep_s)
            time.sleep(sleep_s)
        except ccxt.BaseError as exc:
            log.error("Error CCXT no-retryable: %s", exc)
            raise

    raise RuntimeError(
        f"fetch_ohlcv falló para {symbol} {tf} tras {MAX_RETRIES} intentos."
    )


# >>> MODIFICACIÓN INICIO
# _download_range acepta batch_limit por source (Coinbase=300, resto=1000).
# ccxt.BaseError en el primer fetch se propaga como UnsupportedTimeframe
# para que update_asset_tf_for_source lo capture limpiamente sin stack trace.

class UnsupportedTimeframe(Exception):
    """El exchange no soporta este timeframe — skip silencioso."""


def _download_range(
    exchange:      ccxt.Exchange,
    symbol:        str,
    tf:            str,
    since_ms:      int,
    label:         str           = "",
    show_bar:      bool          = True,
    last_known_ts: Optional[int] = None,
    batch_limit:   int           = BATCH_SIZE,   # ← nuevo parámetro
) -> pd.DataFrame:
    """
    Descarga paginada de todas las velas cerradas desde since_ms.
    live_open se recalcula en CADA iteración (fix central v3+).

    batch_limit : máximo de velas por request — respeta el cap del exchange.
    Si el primer fetch lanza ccxt.BaseError (ej: granularity inválida en
    Coinbase), convierte la excepción a UnsupportedTimeframe para que el
    llamador pueda hacer skip sin imprimir un traceback completo.
    """
    step = tf_to_ms(tf)

    initial_live  = get_live_open(exchange, tf)
    total_ms      = max(0, initial_live - since_ms)
    total_batches = max(1, math.ceil(total_ms / (step * batch_limit)))

    all_bars: list[list] = []
    cursor = since_ms
    first_fetch = True   # flag para detectar error de TF en el primer intento

    pbar = (
        tqdm(total=total_batches, desc=label, unit="batch", leave=False)
        if show_bar else None
    )

    while True:
        live_open = get_live_open(exchange, tf)

        if cursor >= live_open:
            break

        try:
            bars = fetch_ohlcv_with_retry(exchange, symbol, tf, cursor, batch_limit)
        except ccxt.BaseError as exc:
            # Solo en el primer fetch un BaseError indica TF incompatible.
            # En fetches posteriores podría ser un error transitorio real.
            if first_fetch:
                if pbar is not None:
                    pbar.close()
                raise UnsupportedTimeframe(
                    f"[{exchange.id}] {symbol} {tf}: {exc}"
                ) from exc
            raise   # reraise si no es el primer fetch

        first_fetch = False

        if not bars:
            break

        closed: list[list] = []
        for b in bars:
            if b[0] >= live_open:
                continue
            if last_known_ts is not None and b[0] <= last_known_ts:
                continue
            closed.append(b)

        if not closed:
            break

        all_bars.extend(closed)
        cursor = closed[-1][0] + step

        if pbar is not None:
            pbar.update(1)

        if len(bars) < batch_limit:
            break

    if pbar is not None:
        pbar.close()

    if not all_bars:
        return empty_ohlcv_df()

    df = pd.DataFrame(all_bars, columns=_OHLCV_COLS)
    return enforce_schema(df)
# <<< MODIFICACIÓN FIN


# ==============================================================================
# FILE I/O (sin cambios respecto a v4)
# ==============================================================================

def load_existing_raw(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        log.info("Sin archivo previo: %s  →  descarga completa", path)
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        log.warning("Corrupto / ilegible: %s  [%s]  →  recreando", path, exc)
        return None
    if not set(_OHLCV_COLS).issubset(df.columns):
        log.warning("Schema inválido en %s  →  recreando", path)
        return None
    if df.empty:
        log.warning("Archivo vacío: %s  →  recreando", path)
        return None
    return enforce_schema(df)


def _write_atomic(
    table:  pa.Table,
    path:   Path,
    schema: pa.Schema,
    n_rows: int,
) -> None:
    """
    Escritura en tres fases (corruption-safe):
    .tmp → verify rowcount → os.replace() atómico.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_str = tempfile.mkstemp(
        dir=path.parent, prefix=".tmp_", suffix=".parquet"
    )
    os.close(fd)
    tmp_path = Path(tmp_str)

    try:
        pq.write_table(table, tmp_path, compression="snappy")
        check = pd.read_parquet(tmp_path)
        if check.empty or len(check) != n_rows:
            raise IOError(
                f"Verificación fallida en {path.name}: "
                f"esperadas {n_rows} filas, leídas {len(check)}."
            )
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def write_raw(df: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(df, schema=RAW_SCHEMA, preserve_index=False)
    _write_atomic(table, path, RAW_SCHEMA, len(df))


def write_sentinel(df: pd.DataFrame, path: Path) -> None:
    sentinel_df = build_sentinel_df(df)
    table = pa.Table.from_pandas(
        sentinel_df, schema=SENTINEL_SCHEMA, preserve_index=False
    )
    _write_atomic(table, path, SENTINEL_SCHEMA, len(sentinel_df))
    del sentinel_df
    gc.collect()


# ==============================================================================
# GAP DETECTION & SURGICAL FILL (sin cambios respecto a v4)
# ==============================================================================

def detect_gaps(df: pd.DataFrame, tf: str) -> list[GapRange]:
    step = tf_to_ms(tf)
    ts   = df["timestamp"].sort_values().reset_index(drop=True)
    gaps: list[GapRange] = []
    diffs = ts.diff()
    for idx in diffs[diffs > step].index:
        g_start = int(ts[idx - 1])
        g_end   = int(ts[idx])
        missing = int((g_end - g_start) // step) - 1
        gaps.append(GapRange(g_start, g_end, missing))
    return gaps


# >>> MODIFICACIÓN INICIO
# fill_gaps pasa batch_limit a _download_range y etiqueta MARKET_HOLE
# cuando el exchange confirma que no tiene datos para ese periodo.
# Un gap irrecuperable = exchange devuelve 0 velas tras un intento limpio.
# No hay bucle de reintentos sobre datos históricos inexistentes.
def fill_gaps(
    exchange:    ccxt.Exchange,
    symbol:      str,
    tf:          str,
    gaps:        list[GapRange],
    live_cutoff: int,
    batch_limit: int = BATCH_SIZE,   # ← nuevo parámetro
) -> pd.DataFrame:
    step      = tf_to_ms(tf)
    recovered: list[pd.DataFrame] = []

    for gap in gaps:
        log.info("  ↳ Rellenando  %s  (%d vela(s))", str(gap), gap.missing)
        try:
            patch = _download_range(
                exchange, symbol, tf,
                since_ms=gap.fetch_since,
                label=f"    gap {ms_to_iso(gap.gap_start_ms)[:16]}",
                show_bar=True,
                last_known_ts=gap.gap_start_ms,
                batch_limit=batch_limit,
            )
        except UnsupportedTimeframe as exc:
            # TF incompatible — no debería llegar aquí si tf_whitelist es correcta
            log.warning("  ⚠ MARKET_HOLE (TF incompatible): %s", exc)
            continue

        if patch.empty:
            # Exchange no devolvió nada — downtime histórico confirmado
            log.warning(
                "  ⚠ MARKET_HOLE  %s → %s: exchange sin datos para este periodo "
                "(downtime histórico confirmado). Marcado y continuando.",
                ms_to_iso(gap.gap_start_ms), ms_to_iso(gap.gap_end_ms),
            )
            continue

        patch = patch[
            (patch["timestamp"] < gap.gap_end_ms + step) &
            (patch["timestamp"] < live_cutoff)
        ]
        if not patch.empty:
            recovered.append(patch)

    if not recovered:
        return empty_ohlcv_df()

    combined = pd.concat(recovered, ignore_index=True)
    del recovered
    gc.collect()
    return enforce_schema(combined)
# <<< MODIFICACIÓN FIN


# ==============================================================================
# MEMORY-CONSCIOUS MERGE (sin cambios respecto a v4)
# ==============================================================================

def merge_and_clean(
    existing:    Optional[pd.DataFrame],
    incoming:    pd.DataFrame,
    live_cutoff: int,
) -> pd.DataFrame:
    frames = [f for f in (existing, incoming) if f is not None and not f.empty]
    if not frames:
        return empty_ohlcv_df()

    merged = pd.concat(frames, ignore_index=True)
    del frames, existing, incoming
    gc.collect()

    merged = (
        merged
        .drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    merged = merged[merged["timestamp"] < live_cutoff].reset_index(drop=True)
    return enforce_schema(merged)


# ==============================================================================
# >>> [MODIFICACIÓN] update_asset_tf → update_asset_tf_for_source INICIO
# v4: update_asset_tf(exchange, asset, tf)
#     — hardcoded a binanceusdm, rutas planas data/ y db_sentinel/
# v5: update_asset_tf_for_source(exchange, source, asset, tf)
#     — exchange ya instanciado externamente (reusable entre pares del mismo source)
#     — rutas con sub-directorio por exchange_id
#     — symbol y file_prefix se resuelven desde el template en source
# ==============================================================================

# >>> MODIFICACIÓN INICIO
# update_asset_tf_for_source v5.1:
#   · Resuelve batch_limit desde source (Coinbase=300, resto=1000)
#   · Captura UnsupportedTimeframe → WARNING + skip limpio, sin traceback
#   · del merged + gc.collect() explícito tras cada escritura
def update_asset_tf_for_source(
    exchange: ccxt.Exchange,
    source:   dict,
    asset:    str,
    tf:       str,
) -> SyncStats:
    """
    Pipeline completo para un par (exchange_source, asset, timeframe).

    Paso 0  Validar tf contra tf_whitelist del source (skip limpio si no aplica)
    Paso 1  Resolver symbol, rutas y batch_limit desde source + asset + tf
    Paso 2  Cargar / validar raw Parquet existente en data/[exchange_id]/
    Paso 3  Capturar live_cutoff inicial (tiempo servidor)
    Paso 4  Cursor: delta = last_ts + 1 ms  |  full = anchor_ms()
    Paso 5  Descarga con live_open RODANTE y batch_limit correcto
            → UnsupportedTimeframe: WARNING + return stats vacío
    Paso 6  Merge (existing + new), dedup, sort, strip vela viva
    Paso 7  Gap detection → fill quirúrgico → MARKET_HOLE si 0 velas
    Paso 8  Escritura atómica raw  →  del raw_df + gc.collect()
    Paso 9  Escritura atómica Sentinel → del merged + gc.collect()
    """
    eid         = source["exchange_id"]
    symbol      = source["symbol_tpl"].format(asset=asset)
    file_prefix = source["file_prefix"].format(asset=asset)
    batch_limit = source.get("batch_limit", BATCH_SIZE)

    r_path = raw_path(eid, file_prefix, tf)
    s_path = sentinel_path(eid, file_prefix, tf)
    stats  = SyncStats(exchange_id=eid, asset=asset, tf=tf)

    # ── Paso 0: validar tf_whitelist ──────────────────────────────────────────
    tf_whitelist = source.get("tf_whitelist")
    if tf_whitelist is not None and tf not in tf_whitelist:
        log.warning(
            "SKIP  [%s] %s %s — TF no soportado por este exchange "
            "(whitelist: %s)",
            eid, asset, tf, tf_whitelist,
        )
        stats.errors.append(f"TF_NOT_SUPPORTED:{tf}")
        return stats

    log.info("━━━  [%s]  %s  %s  (batch=%d) ━━━", eid, asset, tf, batch_limit)

    # ── Paso 2: cargar histórico raw ──────────────────────────────────────────
    existing_df = load_existing_raw(r_path)
    stats.candles_before = len(existing_df) if existing_df is not None else 0

    # ── Paso 3: live_cutoff inicial ───────────────────────────────────────────
    live_cutoff = get_live_open(exchange, tf)
    log.info("Vela viva boundary  : %s", ms_to_iso(live_cutoff))

    # ── Paso 4: cursor de descarga ────────────────────────────────────────────
    if existing_df is not None:
        last_ts       = int(existing_df["timestamp"].max())
        since         = last_ts + 1
        last_known_ts = last_ts
        log.info("Último guardado : %s  →  delta desde %s",
                 ms_to_iso(last_ts), ms_to_iso(since))
        label = f"↻ [{eid[:8]}] {asset} {tf}"
    else:
        since         = anchor_ms()
        last_known_ts = None
        log.info("Sin histórico  →  descarga completa desde %s", ms_to_iso(since))
        label = f"↓ [{eid[:8]}] {asset} {tf}"

    # ── Paso 5: descarga ──────────────────────────────────────────────────────
    try:
        new_df = _download_range(
            exchange, symbol, tf,
            since_ms=since,
            label=label,
            last_known_ts=last_known_ts,
            batch_limit=batch_limit,
        )
    except UnsupportedTimeframe as exc:
        log.warning(
            "SKIP  [%s] %s %s — TF rechazado por la API del exchange: %s",
            eid, asset, tf, exc,
        )
        stats.errors.append(f"TF_API_REJECTED:{exc}")
        existing_df = None
        gc.collect()
        return stats

    live_cutoff = get_live_open(exchange, tf)
    log.info("Descargadas %d vela(s)  |  boundary final: %s",
             len(new_df), ms_to_iso(live_cutoff))

    # ── Paso 6: merge ─────────────────────────────────────────────────────────
    merged = merge_and_clean(existing_df, new_df, live_cutoff)
    existing_df = None
    new_df      = None
    gc.collect()

    if merged.empty:
        log.warning("Sin datos para [%s] %s %s — omitiendo escritura.", eid, asset, tf)
        return stats

    last_saved_ts = int(merged["timestamp"].iloc[-1])
    log.info(
        "row[-1] (última vela cerrada): %s  [lag: %d vela(s)]",
        ms_to_iso(last_saved_ts),
        (live_cutoff - last_saved_ts) // tf_to_ms(tf),
    )

    # ── Paso 7: gap detection + fill quirúrgico ───────────────────────────────
    gaps = detect_gaps(merged, tf)
    stats.gaps_found = len(gaps)

    if gaps:
        log.warning("GAP CHECK  [%s] %s %s  →  %d gap(s)  — relleno quirúrgico …",
                    eid, asset, tf, len(gaps))
        patch_df = fill_gaps(
            exchange, symbol, tf, gaps, live_cutoff,
            batch_limit=batch_limit,
        )

        if not patch_df.empty:
            merged = merge_and_clean(merged, patch_df, live_cutoff)
            patch_df = None
            gc.collect()

        remaining = detect_gaps(merged, tf)
        stats.gaps_filled    = stats.gaps_found - len(remaining)
        stats.gaps_remaining = len(remaining)

        if remaining:
            log.warning("  %d gap(s) MARKET_HOLE confirmados (downtime histórico):",
                        len(remaining))
            for r in remaining:
                log.warning("    MARKET_HOLE  [%s] %s  %s", eid, asset, str(r))
        else:
            log.info("  Todos los gaps rellenados.")
    else:
        log.info("Gap check OK — serie continua.")

    stats.candles_after = len(merged)

    # ── Paso 8: escribir raw + limpiar memoria ────────────────────────────────
    write_raw(merged, r_path)
    stats.raw_mb = r_path.stat().st_size / 1_048_576
    log.info("Raw guardado      : %s  →  %d velas  (%.2f MB)",
             r_path, stats.candles_after, stats.raw_mb)

    # ── Paso 9: escribir Sentinel + limpiar merged ────────────────────────────
    write_sentinel(merged, s_path)
    stats.sentinel_mb = s_path.stat().st_size / 1_048_576
    log.info("Sentinel guardado : %s  →  notional + datetime  (%.2f MB)",
             s_path, stats.sentinel_mb)

    del merged
    gc.collect()

    return stats
# <<< MODIFICACIÓN FIN

# <<< [MODIFICACIÓN] update_asset_tf_for_source FIN


# ==============================================================================
# SUMMARY TABLE
# ==============================================================================

# >>> MODIFICACIÓN INICIO
def _print_summary(all_stats: list[SyncStats]) -> None:
    sep = "─" * 90
    log.info("")
    log.info(sep)
    log.info(
        "  %-14s  %-5s  %-4s  %8s  %8s  %10s  %7s  %7s  %s",
        "EXCHANGE", "ACTIVO", "TF", "ANTES", "DESPUÉS", "GAPS F/R",
        "RAW MB", "SEN MB", "ESTADO",
    )
    log.info(sep)
    for s in all_stats:
        if not s.errors:
            estado = "OK ✓"
        elif any(e.startswith("TF_NOT_SUPPORTED") or e.startswith("TF_API_REJECTED")
                 for e in s.errors):
            estado = "SKIP (TF)"
        else:
            estado = f"ERR: {s.errors[0][:20]}"

        gap_info = f"{s.gaps_filled}/{s.gaps_remaining}"
        log.info(
            "  %-14s  %-5s  %-4s  %8d  %8d  %10s  %5.2f MB  %5.2f MB  %s",
            s.exchange_id[:14], s.asset, s.tf,
            s.candles_before, s.candles_after,
            gap_info, s.raw_mb, s.sentinel_mb, estado,
        )
    log.info(sep)

    total_raw  = sum(s.raw_mb for s in all_stats)
    total_sen  = sum(s.sentinel_mb for s in all_stats)
    total_hole = sum(s.gaps_remaining for s in all_stats)
    total_skip = sum(1 for s in all_stats
                     if any(e.startswith("TF_") for e in s.errors))
    log.info(
        "  %-21s  %19s  %10s  %5.2f MB  %5.2f MB  skip=%d",
        "TOTAL", "", f"market_holes:{total_hole}",
        total_raw, total_sen, total_skip,
    )
    log.info(sep)
    log.info("")
# <<< MODIFICACIÓN FIN


# ==============================================================================
# >>> MODIFICACIÓN INICIO
# main() v5.2 — ThreadPoolExecutor para sources con concurrent_workers > 1.
#
# Estrategia de concurrencia:
#   workers=1  → loop secuencial (Binance — ya es rápido, no necesita paralelo)
#   workers=2  → BTC y ETH se descargan en paralelo POR TF (Coinbase)
#
# Por qué ThreadPoolExecutor y no multiprocessing:
#   · El GIL de Python se libera durante I/O de red → threads son efectivos
#   · Memoria compartida → sin duplicar los DataFrames en RAM
#   · multiprocessing duplicaría ~200MB de RAM por proceso con datos ya cargados
#
# Por qué NO asyncio:
#   · ccxt sync no es awaitable — requeriría migrar a ccxt.async_support
#   · Cambio mayor que no aporta más velocidad que 2 threads para 2 assets
#
# Seguridad de threads:
#   · Cada thread recibe su PROPIA instancia ccxt (build_exchange_from_source)
#   · Los Parquet se escriben con _write_atomic (.tmp → os.replace) → sin races
#   · all_stats se acumula con lock implícito (append es GIL-safe en CPython)

def _run_pair(
    source: dict,
    asset:  str,
    tf:     str,
) -> SyncStats:
    """
    Worker de thread: instancia su propio exchange ccxt, procesa el par,
    cierra la conexión y retorna SyncStats.
    Cada thread tiene estado completamente independiente.
    """
    try:
        exchange = build_exchange_from_source(source)
        stats    = update_asset_tf_for_source(exchange, source, asset, tf)
    except Exception as exc:
        log.error("FALLO  [%s] %s %s: %s", source["exchange_id"], asset, tf, exc,
                  exc_info=True)
        stats = SyncStats(
            exchange_id=source["exchange_id"], asset=asset, tf=tf,
            errors=[str(exc)],
        )
    finally:
        gc.collect()
    return stats


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SENTINEL_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=== integrity_rock.py  v5  Multi-Exchange Spot Edition  ===")
    log.info("Assets     : %s", ASSETS)
    log.info("Timeframes : %s", TIMEFRAMES)
    log.info("Sources    : %s", [s["label"] for s in SYNC_SOURCES])
    log.info("Raw root   : %s", DATA_DIR.resolve())
    log.info("Sentinel   : %s", SENTINEL_DIR.resolve())
    log.info("")

    # Contar solo pares efectivos (respetando tf_whitelist)
    total_pairs = sum(
        len(ASSETS) * len(s.get("tf_whitelist", TIMEFRAMES))
        for s in SYNC_SOURCES
    )
    all_stats: list[SyncStats] = []
    outer = tqdm(total=total_pairs, desc="Progreso total", unit="par")

    for source in SYNC_SOURCES:
        eid      = source["exchange_id"]
        label    = source["label"]
        workers  = source.get("concurrent_workers", 1)
        tfs      = source.get("tf_whitelist", TIMEFRAMES)

        log.info("▶  Iniciando source: %s  (%s)  [workers=%d]", label, eid, workers)

        # Verificar conectividad con una instancia de prueba
        try:
            probe = build_exchange_from_source(source)
            log.info("Tiempo servidor [%s]: %s", eid, ms_to_iso(probe.milliseconds()))
            del probe
        except Exception as exc:
            log.error("No se pudo conectar a %s: %s", eid, exc)
            for asset in ASSETS:
                for tf in tfs:
                    all_stats.append(SyncStats(
                        exchange_id=eid, asset=asset, tf=tf,
                        errors=[f"connect: {str(exc)[:60]}"],
                    ))
                    outer.update(1)
            continue

        if workers == 1:
            # ── Modo secuencial (Binance) ────────────────────────────────────
            # Una sola instancia ccxt reutilizada entre pares del mismo source.
            exchange = build_exchange_from_source(source)
            for asset in ASSETS:
                for tf in tfs:
                    try:
                        stats = update_asset_tf_for_source(exchange, source, asset, tf)
                    except Exception as exc:
                        log.error("FALLO  [%s] %s %s: %s", eid, asset, tf, exc,
                                  exc_info=True)
                        stats = SyncStats(
                            exchange_id=eid, asset=asset, tf=tf,
                            errors=[str(exc)],
                        )
                    finally:
                        outer.update(1)
                    all_stats.append(stats)

            del exchange

        else:
            # ── Modo concurrente (Coinbase) ──────────────────────────────────
            # Iteramos TF por TF (secuencial) pero dentro de cada TF lanzamos
            # un thread por asset (BTC + ETH en paralelo).
            # Cada thread construye su propia instancia ccxt en _run_pair().
            #
            # ¿Por qué TF-by-TF y no todo en paralelo?
            #   · Coinbase tiene rate limit global por IP, no por símbolo.
            #   · 2 threads × 1 TF = carga controlada y predecible.
            #   · Lanzar 6 threads (2 assets × 3 TFs) simultáneos dispararía
            #     RateLimitExceeded en todos al mismo tiempo.
            for tf in tfs:
                log.info("  [%s] TF %s — lanzando %d threads (assets: %s)",
                         eid, tf, min(workers, len(ASSETS)), ASSETS)

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(_run_pair, source, asset, tf): (asset, tf)
                        for asset in ASSETS
                    }
                    for future in as_completed(futures):
                        asset, tf_done = futures[future]
                        try:
                            stats = future.result()
                        except Exception as exc:
                            stats = SyncStats(
                                exchange_id=eid, asset=asset, tf=tf_done,
                                errors=[str(exc)],
                            )
                        all_stats.append(stats)
                        outer.update(1)

                gc.collect()

        gc.collect()
        log.info("◀  Source %s completado. Memoria liberada.\n", label)

    outer.close()
    _print_summary(all_stats)
    log.info("=== integrity_rock.py  v5  finalizado  ===")

# <<< MODIFICACIÓN FIN


if __name__ == "__main__":
    main()