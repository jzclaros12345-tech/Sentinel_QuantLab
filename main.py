#!/usr/bin/env python3
"""Sentinel Live Ticker.

Dashboard TUI en RAM para velas vivas de Binance Futures. Consulta REST cada
dos segundos, limita la concurrencia y calcula el nocional con Typical Price.
Los umbrales históricos están precalculados en ``MANUAL_STATS`` y expresados
en millones de USDT.
"""

import asyncio
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import ccxt
from textual.app import App, ComposeResult
from textual.containers import Grid
from textual import events
from textual.widgets import Footer, Header, Static

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GLOBAL
# ═══════════════════════════════════════════════════════════════════════════════

SOUND_PATHS = [
    "/usr/share/sounds/LinuxMint/stereo",
    "/usr/share/sounds/freedesktop/stereo",
]
SOUND_CRIT       = "system-ready.ogg"
SOUND_WARN       = "button-toggle-on.ogg"
ALERT_VOLUME     = 60_000
SOUND_DEBOUNCE_S = 4.0


# Máximo de requests simultáneos al exchange (evita 429)
MAX_CONCURRENT_FETCH = 3
MAX_BACKOFF_S = 60

LOGGER = logging.getLogger("sentinel")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    handler = RotatingFileHandler(
        Path(__file__).with_name("sentinel.log"), maxBytes=1_000_000, backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    ))
    LOGGER.addHandler(handler)

PAIRS = [
    ("BTC/USDT:USDT", "1m"), ("BTC/USDT:USDT", "1h"),
    ("BTC/USDT:USDT", "4h"), ("BTC/USDT:USDT", "1d"),
    ("ETH/USDT:USDT", "1m"), ("ETH/USDT:USDT", "1h"),
    ("ETH/USDT:USDT", "4h"), ("ETH/USDT:USDT", "1d"),
]

# Se rellena en main() desde MANUAL_STATS antes de arrancar la TUI
STATS_CONFIG: dict[str, dict] = {}

_ZERO = {"media": 0.0, "mediana": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}

# Valores históricos de notional en millones de USDT, consolidados el 2026-08-17.
MANUAL_STATS: dict[str, dict] = {
    "BTC/USDT:USDT_1m": {"media": 10.207, "mediana": 4.877, "p90": 22.079, "p95": 35.345, "p99": 87.226},
    "BTC/USDT:USDT_1h": {"media": 612.382, "mediana": 397.818, "p90": 1309.125, "p95": 1860.946, "p99": 3354.743},
    "BTC/USDT:USDT_4h": {"media": 2449.213, "mediana": 1787.127, "p90": 5068.413, "p95": 6626.766, "p99": 10344.709},
    "BTC/USDT:USDT_1d": {"media": 14695.429, "mediana": 13292.940, "p90": 25191.381, "p95": 30358.655, "p99": 44915.347},
    "ETH/USDT:USDT_1m": {"media": 6.810, "mediana": 3.121, "p90": 14.841, "p95": 23.835, "p99": 59.554},
    "ETH/USDT:USDT_1h": {"media": 408.575, "mediana": 257.288, "p90": 888.069, "p95": 1247.956, "p99": 2297.289},
    "ETH/USDT:USDT_4h": {"media": 1634.134, "mediana": 1158.426, "p90": 3417.237, "p95": 4607.204, "p99": 7342.577},
    "ETH/USDT:USDT_1d": {"media": 9804.948, "mediana": 8096.360, "p90": 18981.933, "p95": 23749.249, "p99": 33444.057},
}

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def make_widget_id(symbol: str, tf: str) -> str:
    """
    Genera un ID válido para Textual (minúsculas, sin caracteres especiales).

    BUG-1 + BUG-2 FIX:
      'BTC/USDT:USDT'.split('/') → ['BTC', 'USDT:USDT']   (lista, no string)
      ['BTC', 'USDT'].lower() → AttributeError   ← el bug original
      FIX: tomar [0] primero, luego .lower()
    """
    base = symbol.split("/")[0].lower()   # 'btc'
    return f"{base}-{tf}"                  # p. ej. 'btc-4h'


def notional_vwap(h: float, l: float, c: float, v: float) -> float:
    """
    Notional VWAP-style: (H + L + C) / 3 * Volume

    BUG-3 FIX: la versión anterior usaba (H+L)/2 (mid-price), inconsistente
    con integrity_rock y el visualizador Sentinel que usan Typical Price.
    """
    return ((h + l + c) / 3.0) * v / 1_000_000.0   # resultado en millones M


# ═══════════════════════════════════════════════════════════════════════════════
#  AUDIO
# ═══════════════════════════════════════════════════════════════════════════════

class AudioManager:
    def __init__(self) -> None:
        self._last_played = {"crit": 0.0, "warn": 0.0}
        self._sound_dir   = self._find_sound_dir()

    def _find_sound_dir(self) -> Optional[str]:
        for p in SOUND_PATHS:
            if os.path.isdir(p):
                return p
        return None

    def alert(self, is_crit: bool, is_warn: bool) -> None:
        if not self._sound_dir:
            return
        now = time.monotonic()
        if is_crit and (now - self._last_played["crit"]) > SOUND_DEBOUNCE_S:
            self._play(SOUND_CRIT)
            self._last_played["crit"] = now
        elif is_warn and (now - self._last_played["warn"]) > SOUND_DEBOUNCE_S:
            self._play(SOUND_WARN)
            self._last_played["warn"] = now

    def _play(self, filename: str) -> None:
        path = os.path.join(self._sound_dir, filename)
        subprocess.Popen(
            ["paplay", f"--volume={ALERT_VOLUME}", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  WIDGET DE PANEL — FORMATO ORIGINAL MANTENIDO
# ═══════════════════════════════════════════════════════════════════════════════

class VolumePanel(Static):
    def __init__(self, symbol: str, tf: str, stats: dict, widget_id: str) -> None:
        super().__init__(id=widget_id)
        self.symbol            = symbol
        self.tf                = tf
        self.stats             = stats
        self._last_notional:   Optional[float] = None
        self._last_render_args: Optional[tuple] = None
        self._error_message: Optional[str] = None
        self.muted: bool = False

    def _get_threshold_names(self) -> tuple[str, str]:
        """Devuelve los nombres de los percentiles warning y critical."""
        return ("p95", "p99") if self.tf == "1m" else ("p90", "p95")

    def _get_thresholds(self) -> tuple[float, float]:
        warn_name, crit_name = self._get_threshold_names()
        return self.stats[warn_name], self.stats[crit_name]

    def on_click(self, event: events.Click) -> None:
        """Activa o desactiva el sonido para este panel, sin ocultar alertas."""
        self.muted = not self.muted
        if self._last_render_args:
            self.update(self.render_panel(*self._last_render_args))
        self.notify(
            f"{self.symbol} {self.tf}: sonido {'silenciado' if self.muted else 'activado'}",
            severity="information",
        )

    # ── Renderizado (formato original intacto) ────────────────────────────────
    def render_panel(
        self,
        o: float, h: float, l: float, c: float, v: float,
        notional: float, cumplimiento: float,
        status: str, utc_now: str,
    ) -> str:
        stats  = self.stats
        symbol = self.symbol
        tf     = self.tf

        warn_name, crit_name = self._get_threshold_names()

        def color_for(stat_name: str) -> str:
            if stat_name == crit_name:
                return "#ff0000"
            if stat_name == warn_name:
                return "#ffa500"
            return "#00FF00"

        mute_icon = "🔇" if self.muted else "🔔"
        error_line = (
            f" [b #FF0000]ERROR: {self._error_message}[/]\n"
            if self._error_message else ""
        )
        return (
            f"[#00FF00]========================================[/]\n"
            f"  [b #00FF00]SENTINEL | {symbol} | {tf} {mute_icon}[/]\n"
            f"[#00FF00]========================================[/]\n\n"
            f"STAT | μ: {stats['media']:.2f}M | P50: {stats['mediana']:.2f}M\n"
            f"STAT | [{color_for('p90')}]P90: {stats['p90']:.2f}M[/] | "
            f"[{color_for('p95')}]P95: {stats['p95']:.2f}M[/] | "
            f"[{color_for('p99')}]P99: {stats['p99']:.2f}M[/]\n"
            f"[#00FF00]----------------------------------------[/]\n"
            f"LIVE | O:{o:>8.2f} H:{h:>8.2f}\n"
            f"LIVE | L:{l:>8.2f} C:{c:>8.2f}\n"
            f"[b #FF00FF]LIVE | VOL:{v:>10.4f} units[/]\n"
            f"[#00FF00]----------------------------------------[/]\n"
            f" [b #FFA500]NOCIONAL: [{notional:>8.2f}M USD][/] ({cumplimiento:>5.1f}%)\n"
            f" ESTADO  : [ {status:<10} ]  TS: {utc_now}\n"
            f"{error_line}"
            f"[#00FF00]----------------------------------------[/]\n"
            f"[#00FF00]MONITOREO LIVE ACTIVO...[/]\n"
            f"[#00FF00]----------------------------------------[/]"
        )

    # ── Actualización de datos ────────────────────────────────────────────────
    def update_market_data(self, ohlcv_row: list) -> tuple[bool, bool]:
        """
        Recibe una fila OHLCV [ts, o, h, l, c, v] y actualiza el panel.
        Retorna (is_crit, is_warn).
        """
        if not ohlcv_row or len(ohlcv_row) < 6:
            return False, False

        ts, o, h, l, c, v = ohlcv_row[:6]

        # BUG-3 FIX: usar Typical Price (H+L+C)/3 en lugar de mid-price (H+L)/2
        notional = notional_vwap(h, l, c, v)

        # Al recuperarse de un error se limpia el aviso aunque el valor no cambie.
        recovered = self._error_message is not None
        self._error_message = None

        # Skip si no cambió (evita re-render innecesario).
        if notional == self._last_notional:
            if recovered and self._last_render_args:
                self.update(self.render_panel(*self._last_render_args))
            return False, False
        self._last_notional = notional

        warn_th, crit_th = self._get_thresholds()

        is_crit = notional >= crit_th
        is_warn = (notional >= warn_th) and not is_crit

        # Actualizar clase CSS para el borde del panel
        self.remove_class("critical", "warning", "normal", "error")
        if is_crit:
            self.add_class("critical")
        elif is_warn:
            self.add_class("warning")
        else:
            self.add_class("normal")

        cumplimiento = (notional / crit_th * 100) if crit_th > 0 else 0.0
        status       = "CRITICAL" if is_crit else ("WARNING" if is_warn else "OK")
        utc_now      = datetime.fromtimestamp(
            ts / 1_000, tz=timezone.utc
        ).strftime("%d/%m %H:%M:%S")

        self._last_render_args = (o, h, l, c, v, notional, cumplimiento, status, utc_now)
        self.update(self.render_panel(*self._last_render_args))
        return is_crit, is_warn

    def show_error(self, message: str) -> None:
        """Muestra el fallo sin borrar la última vela válida del panel."""
        self._error_message = message[:90]
        self.remove_class("critical", "warning", "normal")
        self.add_class("error")
        if self._last_render_args:
            args = (*self._last_render_args[:7], "ERROR", self._last_render_args[8])
            self.update(self.render_panel(*args))
        else:
            self.update(
                f"[b #ff0000]SENTINEL | {self.symbol} | {self.tf}[/]\n"
                f"ERROR: {self._error_message}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  APP CORE
# ═══════════════════════════════════════════════════════════════════════════════

class SentinelApp(App):
    TITLE = "SentinelApp"
    # [CAMBIO 3/3] — grid-size pasa de 3 2 (6 paneles) a 4 2 (8 paneles)
    CSS = """
    Screen { background: #000000; }
    #main-grid {
        layout: grid;
        grid-size: 4 2;
        grid-gutter: 1;
        height: 100%;
    }
    VolumePanel {
        background: #080808;
        border: solid #1a2a1a;
        padding: 0 1;
    }
    VolumePanel.critical {
        background: #1a0000;
        border: double #ff0000;
    }
    VolumePanel.warning {
        background: #1a1400;
        border: solid #ffa500;
    }
    VolumePanel.error {
        background: #1a0000;
        border: double #ff0000;
    }
    """

    # >>> MODIFICACIÓN INICIO — Live interval (era REFRESH_INTERVAL = 60)
    LIVE_INTERVAL = 2   # segundos entre cada fetch de vela viva
    # <<< MODIFICACIÓN FIN

    def compose(self) -> ComposeResult:
        yield Header()
        with Grid(id="main-grid"):
            for symbol, tf in PAIRS:
                wid   = make_widget_id(symbol, tf)
                stats = STATS_CONFIG.get(f"{symbol}_{tf}", _ZERO)
                yield VolumePanel(symbol, tf, stats, wid)
        yield Footer()

    def on_mount(self) -> None:
        self._exchange = ccxt.binanceusdm({
            "enableRateLimit": True,
            "options": {"adjustForTimeDifference": True},
        })
        self._audio = AudioManager()
        self._sem   = asyncio.Semaphore(MAX_CONCURRENT_FETCH)
        self._failures: dict[tuple[str, str], int] = {}
        self._retry_after: dict[tuple[str, str], float] = {}
        # >>> MODIFICACIÓN INICIO — subtitle fija, sin countdown
        self.sub_title = (
            f"LIVE  │  refresco cada {self.LIVE_INTERVAL}s  │  Pares: {len(PAIRS)}"
        )
        # <<< MODIFICACIÓN FIN
        self.run_worker(self._update_loop(), exclusive=True)

    # ── Loop principal ────────────────────────────────────────────────────────
    async def _update_loop(self) -> None:
        while True:
            try:
                now = time.monotonic()
                due_pairs = [
                    pair for pair in PAIRS if now >= self._retry_after.get(pair, 0.0)
                ]
                results = await asyncio.gather(
                    *(self._fetch_one(symbol, tf) for symbol, tf in due_pairs)
                )

                any_crit = any_warn = False

                for symbol_r, tf_r, data, error in results:
                    wid = make_widget_id(symbol_r, tf_r)

                    try:
                        panel = self.query_one(f"#{wid}", VolumePanel)
                        pair = (symbol_r, tf_r)
                        if error:
                            failures = self._failures.get(pair, 0) + 1
                            self._failures[pair] = failures
                            delay = min(2 ** min(failures, 6), MAX_BACKOFF_S)
                            self._retry_after[pair] = time.monotonic() + delay
                            message = f"{error} (reintento en {delay}s)"
                            panel.show_error(message)
                            LOGGER.warning("Fetch %s %s: %s", symbol_r, tf_r, message)
                            self.notify(
                                f"Fetch error {symbol_r} {tf_r}: {error}",
                                severity="warning",
                            )
                            continue

                        self._failures.pop(pair, None)
                        self._retry_after.pop(pair, None)
                        c, w  = panel.update_market_data(data)
                        if not panel.muted and c:
                            any_crit = True
                        elif not panel.muted and w:
                            any_warn = True
                    except Exception as e:
                        LOGGER.exception("Panel update %s", wid)
                        self.notify(f"Panel update {wid}: {e}", severity="warning")

                self._audio.alert(any_crit, any_warn)

            except Exception as e:
                LOGGER.exception("Update loop error")
                self.notify(f"Update loop error: {e}", severity="error")

            # >>> MODIFICACIÓN INICIO — sleep corto para monitor live
            await asyncio.sleep(self.LIVE_INTERVAL)
            # <<< MODIFICACIÓN FIN

    # ── Fetch individual con semáforo ─────────────────────────────────────────
    async def _fetch_one(
        self, symbol: str, tf: str
    ) -> tuple[str, str, list, Optional[str]]:
        """
        OPT-2: semáforo limita concurrencia y presión sobre la API.
        """
        async with self._sem:
            loop = asyncio.get_running_loop()
            try:
                # >>> MODIFICACIÓN INICIO — vela VIVA (limit=1, índice [-1])
                ohlcv = await loop.run_in_executor(
                    None,
                    lambda: self._exchange.fetch_ohlcv(
                        symbol, timeframe=tf, limit=1
                    ),
                )
                row = ohlcv[-1] if ohlcv else []
                # <<< MODIFICACIÓN FIN
                return symbol, tf, row, None
            except ccxt.ExchangeNotAvailable as e:
                return symbol, tf, [], f"BLOQUEADO: {e}"
            except ccxt.NetworkError as e:
                return symbol, tf, [], f"RED: {e}"
            except Exception as e:
                return symbol, tf, [], f"ERROR: {type(e).__name__}: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
#  INICIALIZACIÓN — cálculo de umbrales históricos
# ═══════════════════════════════════════════════════════════════════════════════

# >>> ELIMINADO — compute_stats (reemplazado por MANUAL_STATS en v2.3)
# <<< ELIMINADO


def main() -> None:
    print("📈 SENTINEL LIVE: Cargando umbrales manuales…")

    for symbol, tf in PAIRS:
        key   = f"{symbol}_{tf}"
        stats = MANUAL_STATS.get(key, _ZERO)
        STATS_CONFIG[key] = stats

        if stats is _ZERO:
            print(f"  ⚠  {symbol:<20} {tf:<4}  sin preset — usando zeros")
        else:
            print(
                f"  ✓  {symbol:<20} {tf:<4}  "
                f"μ={stats['media']:.1f}M  "
                f"P90={stats['p90']:.1f}M  "
                f"P95={stats['p95']:.1f}M  "
                f"P99={stats['p99']:.1f}M"
            )

    print("\n  Umbrales cargados. Arrancando TUI…\n")
    SentinelApp().run()


if __name__ == "__main__":
    main()
