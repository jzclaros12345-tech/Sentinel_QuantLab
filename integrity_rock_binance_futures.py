"""
integrity_rock.py  (v7 — Checkpoint + Parallel Edition)
=======================================================
Quant-grade data pipeline para Binance USDT Perpetual Futures.

Assets     : BTC, ETH
Timeframes : 1m, 5m, 1h
Start date : 2023-01-01 00:00:00 UTC

Cambios v7 vs v6
-----------------
[NEW] CHECKPOINT INCREMENTAL cada CHECKPOINT_EVERY batches (default 200)
      Durante descargas largas (BTC 1m = ~1746 batches) el archivo parquet
      se actualiza en disco progresivamente. Si se corta la ejecución, el
      progreso queda guardado y la próxima corrida hace delta desde el último
      timestamp guardado — no vuelve a descargar desde cero.
[NEW] PARALELISMO con ThreadPoolExecutor (MAX_WORKERS = 3 por defecto)
      1m + 5m + 1h se descargan simultáneamente. Cada worker tiene su propio
      requests.Session para evitar conflictos. Ajustar MAX_WORKERS según
      límites de rate (hasta 6 es seguro con Binance USDT-M).
[KEEP] API REST directa a Binance (sin ccxt)
[KEEP] Logs estratégicos de escritura
[KEEP] Gap detection + fill quirúrgico
[KEEP] Escritura atómica corruption-safe
[KEEP] Un solo path data/
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

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

ASSETS      = ["BTC", "ETH"]
TIMEFRAMES  = ["1m", "5m", "1h", "4h", "1d"]
START_DATE  = "2023-01-01T00:00:00Z"

DATA_DIR = Path("data")

# Binance USDT-M Futures base URL
FAPI_BASE = "https://fapi.binance.com"

BATCH_SIZE       = 1_000   # Binance hard cap 1500; 1000 reduce request weight
MAX_RETRIES      = 6
BACKOFF_BASE     = 2.0
BACKOFF_JITTER   = 0.5
CHECKPOINT_EVERY = 200    # Guardar al disco cada N batches durante descarga larga
MAX_WORKERS      = 3      # Pares en paralelo (1m+5m+1h simultáneo); max 6 seguro

TF_MS: dict[str, int] = {
    "1m":  60_000,
    "5m":  300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h":  3_600_000,
    "4h":  14_400_000,
    "1d":  86_400_000,
}

# ── Schemas PyArrow ───────────────────────────────────────────────────────────
RAW_SCHEMA = pa.schema([
    pa.field("timestamp",           pa.int64()),
    pa.field("open",                pa.float32()),
    pa.field("high",                pa.float32()),
    pa.field("low",                 pa.float32()),
    pa.field("close",               pa.float32()),
    pa.field("volume",              pa.float32()),
    pa.field("quote_vol",           pa.float32()),
    pa.field("taker_buy_quote_vol", pa.float32()),
    # n_trades viene en la respuesta cruda de Binance (índice 8 del kline,
    # ver docstring de fetch_klines_with_retry) pero _bars_to_df nunca lo
    # tomaba — se agrega acá para que quede persistido desde ahora.
    pa.field("trades",              pa.int64()),
])

_OHLCV_COLS = ["timestamp", "open", "high", "low", "close",
               "volume", "quote_vol", "taker_buy_quote_vol", "trades"]

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("integrity_rock")

# ──────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GapRange:
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
    asset:           str
    tf:              str
    candles_before:  int   = 0
    candles_after:   int   = 0
    gaps_found:      int   = 0
    gaps_filled:     int   = 0
    gaps_remaining:  int   = 0
    raw_mb:          float = 0.0
    errors:          list[str] = field(default_factory=list)

# ──────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def tf_to_ms(tf: str) -> int:
    if tf not in TF_MS:
        raise ValueError(f"Timeframe desconocido: {tf!r} — agrégalo a TF_MS.")
    return TF_MS[tf]


def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1_000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def anchor_ms() -> int:
    return int(
        datetime.fromisoformat(START_DATE.replace("Z", "+00:00")).timestamp() * 1_000
    )


def raw_path(asset: str, tf: str) -> Path:
    return DATA_DIR / f"{asset}_USDT_{tf}.parquet"


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = df["timestamp"].astype("int64")
    for col in ("open", "high", "low", "close", "volume",
                "quote_vol", "taker_buy_quote_vol"):
        df[col] = df[col].astype("float32")
    df["trades"] = df["trades"].astype("int64")
    return df


def empty_ohlcv_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_OHLCV_COLS).astype({
        "timestamp":           "int64",
        "open":                "float32",
        "high":                "float32",
        "low":                 "float32",
        "close":               "float32",
        "volume":              "float32",
        "quote_vol":           "float32",
        "taker_buy_quote_vol": "float32",
        "trades":              "int64",
    })

# ──────────────────────────────────────────────────────────────────────────────
# BINANCE REST API — sin ccxt
# ──────────────────────────────────────────────────────────────────────────────

_session = requests.Session()
_session.headers.update({"Accept": "application/json"})


def get_server_time_ms() -> int:
    """
    Tiempo del servidor Binance Futures vía /fapi/v1/time.
    Evita desincronía del reloj local.
    """
    resp = _session.get(f"{FAPI_BASE}/fapi/v1/time", timeout=10)
    resp.raise_for_status()
    return int(resp.json()["serverTime"])


def get_live_open(tf: str) -> int:
    """
    Open-time (epoch-ms) de la vela viva actual en el servidor de Binance.
    Se recalcula en cada llamada (nunca cacheado entre batches).
    """
    step = tf_to_ms(tf)
    server_ms = get_server_time_ms()
    return (server_ms // step) * step


def _binance_symbol(asset: str) -> str:
    """BTC → BTCUSDT (formato Binance Futures)"""
    return f"{asset}USDT"


def fetch_klines_with_retry(
    symbol:   str,   # ej. "BTCUSDT"
    tf:       str,   # ej. "5m"
    since_ms: int,
    limit:    int = BATCH_SIZE,
) -> list[list]:
    """
    GET /fapi/v1/klines con exponential back-off + jitter.

    Binance devuelve lista de listas:
      [open_time, open, high, low, close, volume,
       close_time, quote_asset_vol, n_trades,
       taker_buy_base_vol, taker_buy_quote_vol, ignore]

    Retryable : 429, 418 (rate limit), 5xx, timeout, ConnectionError
    Fatal      : 4xx que no sea 429/418
    """
    url = f"{FAPI_BASE}/fapi/v1/klines"
    params = {
        "symbol":    symbol,
        "interval":  tf,
        "startTime": since_ms,
        "limit":     limit,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _session.get(url, params=params, timeout=30)

            if resp.status_code in (429, 418):
                # Rate limit — respetar Retry-After si viene en header
                retry_after = float(resp.headers.get("Retry-After", 0))
                sleep_s = max(
                    retry_after,
                    BACKOFF_BASE ** attempt + random.uniform(0, BACKOFF_JITTER),
                )
                log.warning(
                    "Rate limit HTTP %d  intento %d/%d  durmiendo %.1fs …",
                    resp.status_code, attempt, MAX_RETRIES, sleep_s,
                )
                time.sleep(sleep_s)
                continue

            if resp.status_code >= 500:
                sleep_s = BACKOFF_BASE ** attempt + random.uniform(0, BACKOFF_JITTER)
                log.warning(
                    "Error servidor %d  intento %d/%d  durmiendo %.1fs …",
                    resp.status_code, attempt, MAX_RETRIES, sleep_s,
                )
                time.sleep(sleep_s)
                continue

            resp.raise_for_status()   # 4xx que no sea rate-limit → fatal
            return resp.json()

        except (requests.ConnectionError, requests.Timeout) as exc:
            sleep_s = BACKOFF_BASE ** attempt + random.uniform(0, BACKOFF_JITTER)
            log.warning(
                "Network error [%s]  intento %d/%d  durmiendo %.1fs …",
                exc, attempt, MAX_RETRIES, sleep_s,
            )
            time.sleep(sleep_s)

    raise RuntimeError(
        f"fetch_klines falló para {symbol} {tf} tras {MAX_RETRIES} intentos."
    )

# ──────────────────────────────────────────────────────────────────────────────
# DOWNLOAD LOOP
# ──────────────────────────────────────────────────────────────────────────────

def _bars_to_df(bars: list[list]) -> pd.DataFrame:
    """Convierte lista cruda de klines Binance a DataFrame limpio."""
    malformed = [b for b in bars if len(b) < 11]
    if malformed:
        log.warning("  %d barra(s) malformada(s) descartadas (len < 11)", len(malformed))
    # b[8] = n_trades — viene siempre en la respuesta de Binance, antes se
    # descartaba acá sin razón. Orden tiene que calzar con _OHLCV_COLS.
    rows = [
        [b[0], b[1], b[2], b[3], b[4], b[5], b[7], b[10], b[8]]
        for b in bars if len(b) >= 11
    ]
    if not rows:
        return empty_ohlcv_df()
    df = pd.DataFrame(rows, columns=_OHLCV_COLS)
    return enforce_schema(df)


def _checkpoint_append(df: pd.DataFrame, path: Path, fresh_rebuild: bool = False) -> None:
    """
    Agrega filas nuevas al parquet existente (o lo crea si no existe).
    Usa escritura atómica: .tmp → verify → os.replace.
    Solo se llama durante descarga larga cada CHECKPOINT_EVERY batches.

    fresh_rebuild: True únicamente en el PRIMER checkpoint de una descarga
    que ya arrancó decidida a reconstruir todo desde el anchor (existing_df
    era None en update_asset_tf — por archivo ausente, corrupto, vacío o
    con schema desactualizado). En ese caso cualquier archivo que siga
    físicamente en disco es basura vieja que YA decidimos tirar más arriba
    en el pipeline, así que se sobreescribe directo, sin intentar mergear
    (evita el bug de "trades" viejo faltante → NaN → cast roto).

    Fuera de ese caso (resumes/delta normales) un error acá NO se traga en
    silencio: antes este except genérico sobreescribía con solo el batch
    en memoria ante CUALQUIER falla de lectura/merge, lo que en una corrida
    de resume real hubiera podido borrar meses de histórico acumulado sin
    avisar. Mejor abortar fuerte y dejar el archivo previo intacto en disco.
    """
    if df.empty:
        return

    if path.exists() and fresh_rebuild:
        log.info("💾  Checkpoint fresco (rebuild): descartando archivo previo en disco en vez de mergear — %s", path.name)
        combined = enforce_schema(df.copy())
    elif path.exists():
        try:
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = (
                combined
                .drop_duplicates(subset="timestamp")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            combined = enforce_schema(combined)
        except Exception as exc:
            log.error(
                "❌  Checkpoint: %s no se pudo mergear con el existente (%s). "
                "NO se sobreescribe a ciegas — el archivo previo en disco queda intacto. "
                "Revisá manualmente antes de reintentar.",
                path.name, exc,
            )
            raise
    else:
        combined = enforce_schema(df.copy())

    table = pa.Table.from_pandas(combined, schema=RAW_SCHEMA, preserve_index=False)
    fd, tmp_str = tempfile.mkstemp(dir=path.parent, prefix=".ckpt_", suffix=".parquet")
    os.close(fd)
    tmp_path = Path(tmp_str)
    try:
        pq.write_table(table, tmp_path, compression="snappy")
        check = pd.read_parquet(tmp_path)
        if len(check) != len(combined):
            raise IOError(f"Checkpoint verify: esperadas {len(combined)}, leídas {len(check)}")
        os.replace(tmp_path, path)
        log.info(
            "💾  Checkpoint  →  %d filas guardadas  |  %.2f MB  |  %s",
            len(combined),
            path.stat().st_size / 1_048_576,
            path.name,
        )
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        log.warning("⚠️  Checkpoint falló (%s) — continuando sin guardar", exc)


def _download_range(
    asset:         str,
    tf:            str,
    since_ms:      int,
    label:         str           = "",
    show_bar:      bool          = True,
    last_known_ts: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    fresh_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Descarga paginada de todas las velas cerradas desde since_ms.

    - live_open se recalcula FRESCO en cada batch.
    - Cada CHECKPOINT_EVERY batches escribe al disco para no perder progreso
      en descargas largas (ej. BTC 1m desde 2023 = ~1746 batches).
    - checkpoint_path: si se provee, activa checkpointing incremental.
    - fresh_rebuild: True cuando esta descarga arrancó sin histórico previo
      válido (existing_df era None). Solo afecta al PRIMER checkpoint de la
      sesión — le dice a _checkpoint_append que sobreescriba en vez de
      mergear, porque cualquier archivo que siga en disco a esta altura es
      basura vieja que ya decidimos descartar (no un histórico a preservar).
    """
    symbol = _binance_symbol(asset)
    step   = tf_to_ms(tf)

    initial_live  = get_live_open(tf)
    total_ms      = max(0, initial_live - since_ms)
    total_batches = max(1, math.ceil(total_ms / (step * BATCH_SIZE)))

    batch_buf:  list[list] = []   # buffer del batch actual
    all_bars:   list[list] = []   # acumulador total
    cursor = since_ms
    batches_since_ckpt = 0
    first_checkpoint = True

    pbar = (
        tqdm(total=total_batches, desc=label, unit="batch", leave=False)
        if show_bar else None
    )

    while True:
        live_open = get_live_open(tf)

        if cursor >= live_open:
            break

        bars = fetch_klines_with_retry(symbol, tf, cursor, BATCH_SIZE)

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
            cursor += step * BATCH_SIZE
            continue

        all_bars.extend(closed)
        cursor = closed[-1][0] + step
        batches_since_ckpt += 1

        if pbar is not None:
            pbar.update(1)

        # ── Checkpoint incremental ────────────────────────────────────────────
        if checkpoint_path is not None and batches_since_ckpt >= CHECKPOINT_EVERY:
            _checkpoint_append(
                _bars_to_df(all_bars), checkpoint_path,
                fresh_rebuild=(fresh_rebuild and first_checkpoint),
            )
            first_checkpoint = False
            all_bars.clear()   # el progreso está en disco — liberar RAM
            gc.collect()
            batches_since_ckpt = 0

        if len(bars) < BATCH_SIZE:
            break

    if pbar is not None:
        pbar.close()

    if not all_bars:
        return empty_ohlcv_df()

    return _bars_to_df(all_bars)

# ──────────────────────────────────────────────────────────────────────────────
# FILE I/O — con logs estratégicos
# ──────────────────────────────────────────────────────────────────────────────

def load_existing_raw(path: Path) -> Optional[pd.DataFrame]:
    """
    Carga y valida Parquet existente.
    Retorna None → full rebuild si: no existe / corrupto / schema malo / vacío.
    """
    if not path.exists():
        log.info("📂  Archivo no encontrado: %s  →  descarga completa desde cero", path.name)
        return None

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        log.warning("⚠️  Corrupto / ilegible: %s  [%s]  →  recreando", path.name, exc)
        return None

    if not set(_OHLCV_COLS).issubset(df.columns):
        log.warning("⚠️  Schema inválido en %s  →  recreando", path.name)
        return None

    if df.empty:
        log.warning("⚠️  Archivo vacío: %s  →  recreando", path.name)
        return None

    log.info(
        "✅  Histórico cargado: %s  →  %d filas  |  %s  →  %s",
        path.name,
        len(df),
        ms_to_iso(int(df["timestamp"].min())),
        ms_to_iso(int(df["timestamp"].max())),
    )
    log.info("    Columnas: %s", list(df.columns))
    return enforce_schema(df)


def _write_atomic(
    table:    pa.Table,
    path:     Path,
    schema:   pa.Schema,
    n_rows:   int,
    label:    str = "",
) -> None:
    """
    Escritura en tres fases (corruption-safe) + logs estratégicos.

    Fase 1 — Serializar a .tmp en el mismo directorio
    Fase 2 — Leer de vuelta y verificar row count
              Si falla: borrar .tmp, raise (original intacto)
    Fase 3 — os.replace() commit atómico
    Fase 4 — LOG de confirmación con filas + tamaño + ruta
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_str = tempfile.mkstemp(
        dir=path.parent, prefix=".tmp_", suffix=".parquet"
    )
    os.close(fd)
    tmp_path = Path(tmp_str)

    log.info(
        "💾  [%s] Pre-write  →  %d filas a guardar en %s",
        label, n_rows, path,
    )

    try:
        # Fase 1 — escribir
        pq.write_table(table, tmp_path, compression="snappy")

        # Fase 2 — verificar
        check = pd.read_parquet(tmp_path)
        rows_read = len(check)

        if check.empty or rows_read != n_rows:
            log.error(
                "❌  [%s] Verificación FALLIDA: esperadas %d filas, leídas %d  →  abortando (archivo original intacto)",
                label, n_rows, rows_read,
            )
            raise IOError(
                f"Verificación fallida en {path.name}: "
                f"esperadas {n_rows} filas, leídas {rows_read}."
            )

        # Fase 3 — commit atómico
        os.replace(tmp_path, path)

        # Fase 4 — log de éxito
        size_mb = path.stat().st_size / 1_048_576
        log.info(
            "✅  [%s] Guardado OK  →  %d filas  |  %.2f MB  |  %s",
            label, rows_read, size_mb, path.resolve(),
        )

    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def write_raw(df: pd.DataFrame, path: Path) -> None:
    """Escribe el parquet OHLCV raw (fuente de verdad)."""
    ts_min = ms_to_iso(int(df["timestamp"].min()))
    ts_max = ms_to_iso(int(df["timestamp"].max()))
    log.info(
        "📝  Raw  →  rango: %s  →  %s  (%d filas)",
        ts_min, ts_max, len(df),
    )
    log.info("    Columnas: %s", list(df.columns))
    table = pa.Table.from_pandas(df, schema=RAW_SCHEMA, preserve_index=False)
    _write_atomic(table, path, RAW_SCHEMA, len(df), label=path.name)




# ──────────────────────────────────────────────────────────────────────────────
# GAP DETECTION & SURGICAL FILL
# ──────────────────────────────────────────────────────────────────────────────

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


def fill_gaps(
    asset:        str,
    tf:           str,
    gaps:         list[GapRange],
    live_cutoff:  int,
) -> pd.DataFrame:
    step      = tf_to_ms(tf)
    recovered: list[pd.DataFrame] = []

    for gap in gaps:
        log.info("  ↳ Rellenando  %s  (%d vela(s))", str(gap), gap.missing)
        patch = _download_range(
            asset,
            tf,
            since_ms=gap.fetch_since,
            label=f"    gap {ms_to_iso(gap.gap_start_ms)[:16]}",
            show_bar=True,
            last_known_ts=gap.gap_start_ms,
            checkpoint_path=None,  # gaps son cortos, sin checkpoint
        )
        if not patch.empty:
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

# ──────────────────────────────────────────────────────────────────────────────
# MEMORY-CONSCIOUS MERGE
# ──────────────────────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────────────────────
# CORE UPDATE CYCLE
# ──────────────────────────────────────────────────────────────────────────────

def update_asset_tf(asset: str, tf: str) -> SyncStats:
    """
    Pipeline completo para un par (asset, timeframe).

    Paso 1  Cargar / validar raw Parquet existente en data/
    Paso 2  Capturar live_cutoff inicial (servidor REST, per-TF)
    Paso 3  Cursor: delta = last_ts + 1 ms  |  full = anchor_ms()
    Paso 4  Descarga con live_open RODANTE en cada batch
    Paso 5  Merge (existing + new), dedup, sort, strip vela viva
    Paso 6  Gap detection → fill quirúrgico → segundo detect_gaps
    Paso 7  Escritura atómica (data/) con verificación y log
    Paso 8  Retornar SyncStats
    """
    r_path = raw_path(asset, tf)
    stats  = SyncStats(asset=asset, tf=tf)

    log.info("━━━  %s  %s  ━━━", asset, tf)

    # ── Paso 1: cargar histórico raw ──────────────────────────────────────────
    existing_df = load_existing_raw(r_path)
    stats.candles_before = len(existing_df) if existing_df is not None else 0

    # ── Paso 2: live_cutoff inicial ───────────────────────────────────────────
    live_cutoff = get_live_open(tf)
    log.info("Vela viva boundary  : %s", ms_to_iso(live_cutoff))

    # ── Paso 3: cursor de descarga ────────────────────────────────────────────
    if existing_df is not None:
        last_ts       = int(existing_df["timestamp"].max())
        since         = last_ts + 1
        last_known_ts = last_ts
        log.info(
            "Último guardado : %s  →  delta desde %s",
            ms_to_iso(last_ts), ms_to_iso(since),
        )
        label = f"↻ {asset} {tf}"
    else:
        since         = anchor_ms()
        last_known_ts = None
        log.info("Sin histórico  →  descarga completa desde %s", ms_to_iso(since))
        label = f"↓ {asset} {tf}"

    # ── Paso 4: descarga con live_open RODANTE + checkpoint incremental ────────
    new_df = _download_range(
        asset, tf,
        since_ms=since,
        label=label,
        last_known_ts=last_known_ts,
        checkpoint_path=r_path,   # activa checkpoint cada CHECKPOINT_EVERY batches
        fresh_rebuild=(existing_df is None),
    )

    # Refrescar live_cutoff después de la descarga
    live_cutoff = get_live_open(tf)
    log.info(
        "Descargadas %d vela(s)  |  boundary final: %s",
        len(new_df), ms_to_iso(live_cutoff),
    )

    # ── Paso 5: merge ─────────────────────────────────────────────────────────
    # new_df contiene solo el remanente desde el último checkpoint.
    # Si hubo checkpoints, el parquet en disco ya tiene la mayor parte de los datos.
    # Recargamos desde disco para hacer el merge completo.
    if r_path.exists():
        disk_df = load_existing_raw(r_path)
    else:
        disk_df = existing_df
    existing_df = None
    merged = merge_and_clean(disk_df, new_df, live_cutoff)
    disk_df = None
    new_df  = None
    gc.collect()

    if merged.empty:
        log.warning("⚠️  Sin datos para %s %s — omitiendo escritura.", asset, tf)
        return stats

    last_saved_ts = int(merged["timestamp"].iloc[-1])
    log.info(
        "row[-1] (última vela cerrada): %s  [lag: %d vela(s)]",
        ms_to_iso(last_saved_ts),
        (live_cutoff - last_saved_ts) // tf_to_ms(tf),
    )

    # ── Paso 6: gap detection + fill quirúrgico ───────────────────────────────
    gaps = detect_gaps(merged, tf)
    stats.gaps_found = len(gaps)

    if gaps:
        log.warning(
            "GAP CHECK  %s %s  →  %d gap(s)  —  relleno quirúrgico …",
            asset, tf, len(gaps),
        )
        patch_df = fill_gaps(asset, tf, gaps, live_cutoff)

        if not patch_df.empty:
            merged = merge_and_clean(merged, patch_df, live_cutoff)
            patch_df = None
            gc.collect()

        remaining = detect_gaps(merged, tf)
        stats.gaps_filled    = stats.gaps_found - len(remaining)
        stats.gaps_remaining = len(remaining)

        if remaining:
            log.warning(
                "  %d gap(s) NO recuperados (outage del exchange o datos históricos faltantes):",
                len(remaining),
            )
            for r in remaining:
                log.warning("    IRRECUPERABLE  %s  %s", asset, str(r))
        else:
            log.info("  Todos los gaps rellenados.")
    else:
        log.info("Gap check OK — serie continua.")

    stats.candles_after = len(merged)

    # ── Paso 7: escribir (data/) ──────────────────────────────────────────────
    write_raw(merged, r_path)
    stats.raw_mb = r_path.stat().st_size / 1_048_576

    del merged
    gc.collect()

    return stats

# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(all_stats: list[SyncStats]) -> None:
    sep = "─" * 80
    log.info("")
    log.info(sep)
    log.info(
        "  %-5s  %-4s  %8s  %8s  %10s  %7s  %s",
        "ACTIVO", "TF", "ANTES", "DESPUÉS", "GAPS F/R", "MB", "ESTADO",
    )
    log.info(sep)
    for s in all_stats:
        estado   = "✅ OK" if not s.errors else f"❌ ERR: {s.errors[0][:24]}"
        gap_info = f"{s.gaps_filled}/{s.gaps_remaining}"
        log.info(
            "  %-5s  %-4s  %8d  %8d  %10s  %5.2f MB  %s",
            s.asset, s.tf,
            s.candles_before, s.candles_after,
            gap_info, s.raw_mb, estado,
        )
    log.info(sep)
    log.info("")

# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=== integrity_rock.py  v6  iniciado  ===")
    log.info("Activos      : %s", ASSETS)
    log.info("Timeframes   : %s", TIMEFRAMES)
    log.info("Data dir     : %s", DATA_DIR.resolve())
    log.info("API base     : %s", FAPI_BASE)

    # Verificar conectividad con el servidor de Binance
    server_ms = get_server_time_ms()
    log.info("Tiempo del servidor Binance: %s", ms_to_iso(server_ms))

    pairs = [(asset, tf) for asset in ASSETS for tf in TIMEFRAMES]
    total_pairs = len(pairs)
    all_stats: list[SyncStats] = []

    log.info("Pares a procesar : %d  |  workers : %d", total_pairs, MAX_WORKERS)

    outer = tqdm(total=total_pairs, desc="Progreso total", unit="par")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(update_asset_tf, asset, tf): (asset, tf)
                   for asset, tf in pairs}
        for fut in as_completed(futures):
            asset, tf = futures[fut]
            try:
                stats = fut.result()
            except Exception as exc:
                log.error("FALLO  %s %s: %s", asset, tf, exc, exc_info=True)
                stats = SyncStats(asset=asset, tf=tf, errors=[str(exc)])
            finally:
                outer.update(1)
            all_stats.append(stats)

    outer.close()
    _print_summary(all_stats)
    log.info("=== integrity_rock.py  v6  finalizado  ===")


if __name__ == "__main__":
    main()