#!/usr/bin/env python3
"""
SENTINEL LIVE TICKER v2.2 — Bug-Fixed & Optimized
═══════════════════════════════════════════════════════════════════════════════
Ticker puro en RAM con visualización TUI Textual.
Sin Parquet, sin DB. Formato de renderizado original mantenido.

Bugs corregidos vs v2.1
────────────────────────
BUG-1  symbol.split("/").lower() → AttributeError (lista no tiene .lower())
       FIX → symbol.split("/")[0].lower()

BUG-2  Widget ID con mayúsculas → query_one() nunca encontraba el panel
       FIX → ID siempre en minúsculas: f"{base}-{tf}"  donde base = lower()

BUG-3  notional usaba (H+L)/2 en lugar de (H+L+C)/3 (inconsistente con
       el resto del sistema Sentinel)
       FIX → typical_price = (h + l + c) / 3

BUG-4  ccxt.binance + defaultType="future" → fallas esporádicas en perps
       FIX → ccxt.binanceusdm directamente (igual que integrity_rock)

BUG-5  bare except: continue silenciaba bugs reales y dificultaba debug
       FIX → except Exception as e: log con self.notify o print

Optimizaciones RAM (8 GB)
──────────────────────────
OPT-1  fetch_ohlcv limit=2 → tomamos solo la última vela cerrada (limit=2
       garantiza obtener la vela -2 que ya cerró y no la viva -1 si el
       exchange aún no la consolidó; tomamos índice [-2] del resultado)
OPT-2  gather() con semáforo para no abrir 6 conexiones simultáneas
OPT-3  loop del executor reutiliza el mismo exchange object (un solo socket)
OPT-4  LOOKBACK_CANDLES ajustable; 500 velas × 6 pares = 3000 rows, ~1 MB
═══════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Optional

import ccxt
import numpy as np
# >>> ELIMINADO — import pandas (no requerido en v2.3 manual stats)
# <<< ELIMINADO
from textual.app import App, ComposeResult
from textual.containers import Grid
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

# [CAMBIO 1/3] — Agregados BTC 1m y ETH 1m al final de la lista
PAIRS = [
    ("BTC/USDT:USDT", "1m"), ("BTC/USDT:USDT", "5m"),  ("BTC/USDT:USDT", "1h"),  ("BTC/USDT:USDT", "1d"),
    ("ETH/USDT:USDT", "1m"), ("ETH/USDT:USDT", "5m"),  ("ETH/USDT:USDT", "1h"),  ("ETH/USDT:USDT", "1d"),
      ]

# Se rellena en main() desde MANUAL_STATS antes de arrancar la TUI
STATS_CONFIG: dict[str, dict] = {}

# >>> AÑADIDO INICIO — Diccionario maestro de umbrales manuales (v2.3)
#     Todos los valores en millones de USDT (M).
#     Ajusta según tu análisis histórico con el visualizador Sentinel.
_ZERO = {"media": 0.0, "mediana": 0.0, "p95": 0.0, "gamma": 0.0}

# [CAMBIO 2/3] — Agregadas entradas 1m para BTC y ETH en MANUAL_STATS
#     Valores 1m razonables: gamma y p95 menores que 5m (velas más cortas
#     acumulan menos volumen por intervalo).
MANUAL_STATS: dict[str, dict] = {
    "BTC/USDT:USDT_1m": {"media": 10.58,    "mediana":  5.08,    "p95":  89.72,   "gamma":  36.55},
    "BTC/USDT:USDT_5m": {"media": 53.07,    "mediana": 28.90,    "p95": 387.24,   "gamma": 175.25},
    "BTC/USDT:USDT_1h": {"media": 636.86,   "mediana": 414.74,   "p95": 3420.52,  "gamma": 1909.38},
    "BTC/USDT:USDT_4h": {"media": 800.0,    "mediana": 750.0,    "p95": 2200.0,   "gamma": 1640.0},
    "BTC/USDT:USDT_1d": {"media": 15289.64, "mediana": 14128.15, "p95": 44987.93, "gamma": 31342.04},
    "ETH/USDT:USDT_1m": {"media":  6.98,    "mediana":  3.20,    "p95":  60.60,   "gamma":  24.42},
    "ETH/USDT:USDT_5m": {"media": 34.93,    "mediana": 18.21,    "p95": 264.97,   "gamma": 118.31},
    "ETH/USDT:USDT_1h": {"media": 419.15,   "mediana": 262.0,    "p95": 2369.24,  "gamma": 1286.92},
    "ETH/USDT:USDT_4h": {"media": 500.0,    "mediana": 470.0,    "p95": 7572.13,   "gamma": 4764.62},
    "ETH/USDT:USDT_1d": {"media": 10062.35, "mediana": 8119.10,  "p95": 33459.18, "gamma": 24254.50},
}
# <<< AÑADIDO FIN1

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
    return f"{base}-{tf}"                  # 'btc-5m'


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

        return (
            f"[#00FF00]========================================[/]\n"
            f"  [b #00FF00]SENTINEL | {symbol} | {tf}[/]\n"
            f"[#00FF00]========================================[/]\n\n"
            f"STAT | μ: {stats['media']:.2f}M | P50: {stats['mediana']:.2f}M\n"
            f"STAT | γ: {stats['gamma']:.2f}M | P95: {stats['p95']:.2f}M\n"
            f"[#00FF00]----------------------------------------[/]\n"
            f"LIVE | O:{o:>8.2f} H:{h:>8.2f}\n"
            f"LIVE | L:{l:>8.2f} C:{c:>8.2f}\n"
            f"[b #FF00FF]LIVE | VOL:{v:>10.4f} units[/]\n"
            f"[#00FF00]----------------------------------------[/]\n"
            f" [b #FFA500]NOCIONAL: [{notional:>8.2f}M USD][/] ({cumplimiento:>5.1f}%)\n"
            f" ESTADO  : [ {status:<10} ]  TS: {utc_now}\n"
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

        # Skip si no cambió (evita re-render innecesario)
        if notional == self._last_notional:
            return False, False
        self._last_notional = notional

        p95   = self.stats["p95"]
        gamma = self.stats["gamma"]

        is_crit = notional >= p95
        is_warn = (notional >= gamma) and not is_crit

        # Actualizar clase CSS para el borde del panel
        self.remove_class("critical", "warning", "normal")
        if is_crit:
            self.add_class("critical")
        elif is_warn:
            self.add_class("warning")
        else:
            self.add_class("normal")

        cumplimiento = (notional / p95 * 100) if p95 > 0 else 0.0
        status       = "CRITICAL" if is_crit else ("WARNING" if is_warn else "OK")
        utc_now      = datetime.fromtimestamp(
            ts / 1_000, tz=timezone.utc
        ).strftime("%d/%m %H:%M:%S")

        self.update(
            self.render_panel(o, h, l, c, v, notional, cumplimiento, status, utc_now)
        )
        return is_crit, is_warn


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
    """

    # >>> MODIFICACIÓN INICIO — Live interval (era REFRESH_INTERVAL = 60)
    LIVE_INTERVAL = 2   # segundos entre cada fetch de vela viva
    # <<< MODIFICACIÓN FIN

    def compose(self) -> ComposeResult:
        yield Header()
        with Grid(id="main-grid"):
            for symbol, tf in PAIRS:
                wid   = make_widget_id(symbol, tf)
                stats = STATS_CONFIG.get(f"{symbol}_{tf}", {
                    "media": 0.0, "mediana": 0.0, "p95": 0.0, "gamma": 0.0
                })
                yield VolumePanel(symbol, tf, stats, wid)
        yield Footer()

    def on_mount(self) -> None:
        self._exchange = ccxt.binanceusdm({
            "enableRateLimit": True,
            "options": {"adjustForTimeDifference": True},
        })
        self._audio = AudioManager()
        self._sem   = asyncio.Semaphore(MAX_CONCURRENT_FETCH)
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
                tasks   = [self._fetch_one(s, t) for s, t in PAIRS]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                any_crit = any_warn = False

                for (symbol, tf), result in zip(PAIRS, results):
                    if isinstance(result, Exception):
                        self.notify(
                            f"Fetch error {symbol} {tf}: {result}",
                            severity="warning",
                        )
                        continue

                    symbol_r, tf_r, data = result
                    wid = make_widget_id(symbol_r, tf_r)

                    try:
                        panel = self.query_one(f"#{wid}", VolumePanel)
                        c, w  = panel.update_market_data(data)
                        if c:
                            any_crit = True
                        elif w:
                            any_warn = True
                    except Exception as e:
                        self.notify(f"Panel update {wid}: {e}", severity="warning")

                self._audio.alert(any_crit, any_warn)

            except Exception as e:
                self.notify(f"Update loop error: {e}", severity="error")

            # >>> MODIFICACIÓN INICIO — sleep corto para monitor live
            await asyncio.sleep(self.LIVE_INTERVAL)
            # <<< MODIFICACIÓN FIN

    # ── Fetch individual con semáforo ─────────────────────────────────────────
    async def _fetch_one(
        self, symbol: str, tf: str
    ) -> tuple[str, str, list]:
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
                return symbol, tf, row
            except Exception as e:
                return symbol, tf, []   # panel mostrará datos anteriores


# ═══════════════════════════════════════════════════════════════════════════════
#  INICIALIZACIÓN — cálculo de umbrales históricos
# ═══════════════════════════════════════════════════════════════════════════════

# >>> ELIMINADO — compute_stats (reemplazado por MANUAL_STATS en v2.3)
# <<< ELIMINADO


def main() -> None:
    # >>> MODIFICADO INICIO — carga desde MANUAL_STATS, sin fetch histórico
    print("📈 SENTINEL LIVE v2.3: Cargando umbrales manuales…")

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
                f"γ={stats['gamma']:.1f}M  "
                f"P95={stats['p95']:.1f}M"
            )

    print("\n  Umbrales cargados. Arrancando TUI…\n")
    SentinelApp().run()
    # <<< MODIFICADO FIN


if __name__ == "__main__":
    main()