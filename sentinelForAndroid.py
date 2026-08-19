#!/usr/bin/env python3
"""Sentinel Live Ticker (versión Android/Termux).
 
Dashboard TUI en RAM para velas vivas de Binance Futures. Consulta REST cada
dos segundos, limita la concurrencia y calcula el nocional con Typical Price.
Los umbrales históricos están precalculados en ``MANUAL_STATS`` y expresados
en millones de USDT.
 
Alertas nativas de Android vía Termux:API: notificación push (ntfy, opcional),
notificación local (termux-notification) y vibración (termux-vibrate). También
avisa de forma discreta si se cae la conectividad general (no un solo símbolo).

Historial de alertas: cada vez que una vela entra en CRITICAL o WARNING se
dispara una notificación Android individual (con id único, no se pisan entre
sí) que queda ACUMULADA en la barra de notificaciones — igual que una lista de
mensajes de WhatsApp — para revisar qué pasó mientras no mirabas la pantalla
o no escuchaste la alarma sonora/vibración.

Paneles compactos: se sacaron las líneas de "====" / "----" y texto de
relleno del render original para aprovechar mejor una pantalla de celular
chica. Cada panel muestra el emoji de estado de alarma bien visible en el
encabezado (🔔 ACTIVA / 🔇 MUTEADA).

BUGFIX conocido (Termux/Textual 7.5.0): el widget AndroidVolumeControl pintaba
vía self.update() llamado desde on_mount(), pero Textual puede intentar el
primer render ANTES de que on_mount termine — en ese instante el widget no
tenía contenido asignado y tiraba AttributeError: 'NoneType' object has no
attribute 'render_strips'. Se corrigió sobreescribiendo render() en vez de
depender de update()/on_mount. Si volvés a generar este archivo desde una
copia vieja (p. ej. pidiéndole mejoras a otra IA), verificá que esta clase
siga usando render()/refresh() y no update()/on_mount().
 
────────────────────────────────────────────────────────────────────────────
 INSTALACIÓN EN TERMUX (Android) — probado en Termux + Termux:API
────────────────────────────────────────────────────────────────────────────
 
1) Paquetes del sistema (vía pkg, NO pip):
 
    pkg update -y
    pkg install python termux-api -y
 
   Estos dos son OBLIGATORIOS aparte de tener la app Termux:API instalada:
   - `python`         → intérprete para correr este script.
   - `termux-api`      → paquete de comandos (termux-notification,
                         termux-vibrate, termux-volume) que hablan con la
                         app Termux:API. Sin este paquete, la app instalada
                         sola no sirve de nada — son dos cosas distintas.
 
2) Dependencias de compilación (vía pkg, ANTES del pip install):
 
   Sin esto, `pip install` falla al intentar compilar dependencias
   transitivas de ccxt desde cero (no hay wheels precompiladas para
   Termux/Android de estas dos):
 
    pkg install python-cryptography -y   # evita compilar 'cryptography' con Rust
    pkg install cmake -y                 # necesario para compilar 'pycares'
 
3) Librerías Python (recién ahora, vía pip):
 
    pip install ccxt==4.5.36 textual==7.5.0 requests==2.32.5
 
4) Verificar que Termux:API responde antes de correr el script:
 
    termux-notification --title test --content prueba   # debe aparecer una notif.
    termux-vibrate -d 200                                # debe vibrar
 
5) Variables de entorno opcionales:
 
    export SENTINEL_NTFY_TOPIC="tu-topico-secreto"   # activa push vía ntfy.sh
    export SENTINEL_ALERT_DEBOUNCE_S="60"            # segundos entre alertas repetidas
 
6) Evitar que Android mate el proceso en background, y ejecutar:
 
    termux-wake-lock
    python sentinelForAndroid.py
 
────────────────────────────────────────────────────────────────────────────
 ORDEN CORRECTO — resumen para copiar/pegar en un dispositivo nuevo
────────────────────────────────────────────────────────────────────────────
 
    pkg update -y
    pkg install python termux-api cmake python-cryptography -y
    pip install ccxt==4.5.36 textual==7.5.0 requests==2.32.5
    termux-wake-lock
    python sentinelForAndroid.py
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import ccxt
import requests
from textual.app import App, ComposeResult
from textual.containers import Grid, Horizontal, VerticalScroll
from textual import events
from textual.widgets import Button, Footer, Header, Static

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GLOBAL
# ═══════════════════════════════════════════════════════════════════════════════

# Configura SENTINEL_NTFY_TOPIC en Termux para activar el canal push.
# No hay tópico por defecto: así nunca se publica un secreto accidentalmente.
NTFY_TOPIC = os.environ.get("SENTINEL_NTFY_TOPIC", "").strip()
NTFY_SERVER = os.environ.get("SENTINEL_NTFY_SERVER", "https://ntfy.sh").rstrip("/")
ALERT_DEBOUNCE_S = float(os.environ.get("SENTINEL_ALERT_DEBOUNCE_S", "60"))


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
#  ALERTAS ANDROID / TERMUX
# ═══════════════════════════════════════════════════════════════════════════════

class TermuxAlertManager:
    """Envía alertas sin depender de audio de escritorio ni bloquear la TUI."""

    def __init__(self) -> None:
        self._last_sent = {"crit": 0.0, "warn": 0.0}
        # Contador para IDs únicos de notificación de HISTORIAL. A diferencia
        # de _notify_termux (id fijo "sentinel-price", se pisa a sí misma),
        # cada entrada de historial necesita un id distinto para que Android
        # las apile como notificaciones separadas — igual que una lista de
        # mensajes de WhatsApp, no una sola notificación que se actualiza.
        self._history_counter = 0

    @property
    def push_configured(self) -> bool:
        return bool(NTFY_TOPIC)

    def _notify_push(self, title: str, message: str, priority: str) -> None:
        if not NTFY_TOPIC:
            return
        try:
            url = f"{NTFY_SERVER}/{quote(NTFY_TOPIC, safe='')}"
            response = requests.post(
                url,
                data=message.encode("utf-8"),
                headers={"Title": title, "Priority": priority},
                timeout=5,
            )
            response.raise_for_status()
        except Exception as exc:
            LOGGER.warning("ntfy notification failed: %s", exc)

    def _notify_termux(self, title: str, message: str, notification_id: str,
                       priority: str = "default") -> None:
        try:
            result = subprocess.run(
                [
                    "termux-notification", "--id", notification_id,
                    "--title", title, "--content", message,
                    "--priority", priority,
                ],
                timeout=5,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode:
                LOGGER.warning("termux-notification exited with status %s", result.returncode)
        except Exception as exc:
            LOGGER.warning("termux-notification failed: %s", exc)

    def _vibrate(self, is_crit: bool) -> None:
        try:
            duration_ms = "1000" if is_crit else "400"
            result = subprocess.run(
                ["termux-vibrate", "-d", duration_ms],
                timeout=5,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode:
                LOGGER.warning("termux-vibrate exited with status %s", result.returncode)
        except Exception as exc:
            LOGGER.warning("termux-vibrate failed: %s", exc)

    async def notify_history(self, symbol: str, tf: str, status: str, utc_now: str) -> None:
        """Notificación de HISTORIAL: una por cada vela que cruza a
        CRITICAL/WARNING, sin debounce y con id único, así se van apilando
        en la barra de notificaciones de Android como una lista de mensajes
        (igual que WhatsApp) en vez de reemplazarse entre sí. Útil para
        revisar qué pasó mientras no mirabas la pantalla o no escuchaste
        la alarma sonora/vibración.
        """
        self._history_counter += 1
        notif_id = f"sentinel-hist-{self._history_counter}"
        pair_label = symbol.split("/")[0]
        title = f"{pair_label} {tf} · {status}"
        content = f"{utc_now} UTC"
        await asyncio.to_thread(
            self._notify_termux, title, content, notif_id,
            "high" if status == "CRITICAL" else "default",
        )

    async def alert(self, is_crit: bool, is_warn: bool, details: list[str]) -> None:
        """Notifica una alerta de precio con debounce por severidad."""
        level = "crit" if is_crit else ("warn" if is_warn else None)
        if level is None:
            return
        now = time.monotonic()
        if now - self._last_sent[level] < ALERT_DEBOUNCE_S:
            return
        self._last_sent[level] = now

        critical = level == "crit"
        label = "CRITICAL" if critical else "WARNING"
        message = "Umbral cruzado: " + (", ".join(details) or "sin detalle")
        await asyncio.gather(
            asyncio.to_thread(
                self._notify_push, f"Sentinel - {label}", message,
                "urgent" if critical else "high",
            ),
            asyncio.to_thread(
                self._notify_termux, f"Sentinel - {label}", message,
                "sentinel-price", "high" if critical else "default",
            ),
            asyncio.to_thread(self._vibrate, critical),
        )

    async def notify_connectivity(self, recovered: bool) -> None:
        """Avisa una sola vez al caer o recuperarse la conectividad general."""
        title = "Sentinel - Red"
        message = (
            "Conexión recuperada: las velas vuelven a actualizarse."
            if recovered else
            "Sin conexión: los datos mostrados pueden estar desactualizados."
        )
        await asyncio.gather(
            asyncio.to_thread(self._notify_push, title, message, "low"),
            asyncio.to_thread(
                self._notify_termux, title, message, "sentinel-connectivity", "low",
            ),
        )


class AndroidVolumeControl(Static):
    """Barra táctil para ajustar el stream de notificaciones de Android."""

    def __init__(self, value: int = 8, maximum: int = 15) -> None:
        # BUGFIX (Termux/Textual 7.5.0): antes se pintaba vía self.update()
        # llamado desde on_mount(), pero Textual puede intentar el primer
        # render ANTES de que on_mount termine de correr. En ese instante
        # el widget no tenía contenido asignado (visual = None) y el intento
        # de dibujarlo tiraba AttributeError: 'NoneType' object has no
        # attribute 'render_strips'. FIX: sobrescribir render() (que Textual
        # siempre puede invocar con seguridad) en vez de depender de update().
        super().__init__(id="android-volume")
        self.value = value
        self.maximum = max(1, maximum)
        self._dragging = False

    def set_range(self, value: int, maximum: int) -> None:
        self.maximum = max(1, maximum)
        self.value = max(0, min(value, self.maximum))
        self.refresh()

    def render(self) -> str:
        bar_width = max(10, self.size.width - 29)
        filled = round(bar_width * self.value / self.maximum) if self.maximum else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        return (
            f"[b #00FF00]VOL. ALERTAS[/]  [{bar}]  {self.value}/{self.maximum}\n"
            "[dim]Botones +/- o arrastra y suelta[/]"
        )

    def step(self, delta: int) -> None:
        """Sube/baja el volumen de a 1 con los botones +/-, aplicando el
        cambio de inmediato (a diferencia del drag, que aplica recién al
        soltar). Más confiable que el drag táctil, que puede sentirse
        trabado según cómo Termux traduzca eventos de touch a mouse.
        """
        new_value = max(0, min(self.value + delta, self.maximum))
        if new_value != self.value:
            self.value = new_value
            self.refresh()
            self.app.apply_notification_volume(self.value)

    def _set_from_x(self, x: int) -> None:
        width = max(1, self.size.width - 1)
        value = round(self.maximum * max(0, min(x, width)) / width)
        if value != self.value:
            self.value = value
            self.refresh()

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button == 1:
            self._dragging = True
            self.capture_mouse()
            self._set_from_x(event.x)
            event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._dragging:
            self._set_from_x(event.x)
            event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._dragging:
            self._dragging = False
            self.release_mouse()
            self._set_from_x(event.x)
            self.app.apply_notification_volume(self.value)
            event.stop()


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
        # Historial: solo registramos cuando el estado CAMBIA a CRITICAL/WARNING,
        # no en cada fetch (evita spamear el log cada 2s mientras sigue arriba
        # del umbral).
        self._last_status: str = "OK"

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

    # ── Renderizado (compacto — pensado para pantallas de celular chicas) ─────
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

        # Emoji grande y explícito de estado de alarma — activa vs. muteada.
        # Se mantiene siempre visible en el encabezado aunque el panel esté
        # compacto, porque es justo lo que se necesita ver de un vistazo.
        mute_icon = "🔇 MUTEADA" if self.muted else "🔔 ACTIVA"

        status_color = {
            "CRITICAL": "#ff0000", "WARNING": "#ffa500",
            "OK": "#00FF00", "ERROR": "#ff0000",
        }.get(status, "#00FF00")

        error_line = (
            f"[b #FF0000]ERR: {self._error_message}[/]\n" if self._error_message else ""
        )
        return (
            f"[b #00FF00]{symbol.split('/')[0]} {tf}[/]  {mute_icon}\n"
            # Cada dato en su PROPIA línea, siempre — así el layout nunca
            # depende de cuántos dígitos tenga el precio (BTC ~65000, 5
            # dígitos, vs ETH ~1900, 4 dígitos). Antes se agrupaban 2-3
            # valores por línea y con BTC se pasaban del ancho de columna,
            # Textual los envolvía a una línea nueva, y eso corría todo lo
            # de abajo (notional/estado/hora) fuera de la altura fija del
            # panel, cortándolo. Una línea por dato = ancho sobra siempre.
            f"P90[{color_for('p90')}]{stats['p90']:.1f}[/]\n"
            f"P95[{color_for('p95')}]{stats['p95']:.1f}[/]\n"
            f"P99[{color_for('p99')}]{stats['p99']:.1f}[/]\n"
            f"O:{o:.1f}\n"
            f"H:{h:.1f}\n"
            f"L:{l:.1f}\n"
            f"C:{c:.1f}\n"
            f"[b {status_color}]{notional:.2f}M ({cumplimiento:.0f}%)  {status}[/]\n"
            f"{error_line}"
            f"[dim]{utc_now}[/]"
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

        # Solo se registra en el historial la TRANSICIÓN hacia CRITICAL/WARNING,
        # no cada vez que el fetch confirma que sigue arriba del umbral.
        if status in ("CRITICAL", "WARNING") and status != self._last_status:
            self.app.record_alert_event(self.symbol, self.tf, status, utc_now)
        self._last_status = status

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
                f"[b #ff0000]{self.symbol.split('/')[0]} {self.tf}[/]\n"
                f"ERROR: {self._error_message}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  APP CORE
# ═══════════════════════════════════════════════════════════════════════════════

class SentinelApp(App):
    TITLE = "SentinelApp"
    # Grilla de 2 columnas x 4 filas (antes 4x2): en pantallas de celular
    # angostas, 4 columnas dejaban ~9 caracteres por panel y cortaba números
    # como "64209.12" a la mitad. Con 2 columnas cada panel tiene el doble
    # de ancho y el OHLCV entra completo. Se compensa con scroll vertical
    # (#main-scroll) para ver los 8 paneles sin que se corte la pantalla.
    CSS = """
    Screen { background: #000000; }
    #main-scroll {
        height: 1fr;
    }
    #main-grid {
        layout: grid;
        grid-size: 2 4;
        grid-gutter: 1;
        grid-rows: 12 12 12 12;
        height: auto;
    }
    #volume-row {
        height: 3;
        margin: 0 1;
    }
    #vol-minus, #vol-plus {
        width: 5;
        height: 3;
        min-width: 5;
        border: solid #1a2a1a;
        background: #080808;
        color: #00FF00;
    }
    #android-volume {
        height: 3;
        padding: 0 1;
        background: #080808;
        border: solid #1a2a1a;
        width: 1fr;
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
        with Horizontal(id="volume-row"):
            yield Button("−", id="vol-minus")
            yield AndroidVolumeControl()
            yield Button("+", id="vol-plus")
        with VerticalScroll(id="main-scroll"):
            with Grid(id="main-grid"):
                for symbol, tf in PAIRS:
                    wid   = make_widget_id(symbol, tf)
                    stats = STATS_CONFIG.get(f"{symbol}_{tf}", _ZERO)
                    yield VolumePanel(symbol, tf, stats, wid)
        yield Footer()

    def on_mount(self) -> None:
        self._exchange = ccxt.binanceusdm({
            "enableRateLimit": True,
            "timeout": 10_000,
            "options": {"adjustForTimeDifference": True},
        })
        self._alerts = TermuxAlertManager()
        self._sem   = asyncio.Semaphore(MAX_CONCURRENT_FETCH)
        self._failures: dict[tuple[str, str], int] = {}
        self._retry_after: dict[tuple[str, str], float] = {}
        self._connectivity_down = False
        # >>> MODIFICACIÓN INICIO — subtitle fija, sin countdown
        self.sub_title = (
            f"LIVE  │  refresco cada {self.LIVE_INTERVAL}s  │  Pares: {len(PAIRS)}"
        )
        if not self._alerts.push_configured:
            LOGGER.warning(
                "Push ntfy desactivado: configura SENTINEL_NTFY_TOPIC para habilitarlo"
            )
        # <<< MODIFICACIÓN FIN
        self.run_worker(self._load_notification_volume(), group="android-volume", exclusive=True)
        self.run_worker(self._update_loop(), exclusive=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Botones +/- junto a la barra de volumen: suben/bajan de a 1 y
        aplican el cambio al instante, sin depender del drag táctil."""
        vol = self.query_one(AndroidVolumeControl)
        if event.button.id == "vol-minus":
            vol.step(-1)
        elif event.button.id == "vol-plus":
            vol.step(1)

    def record_alert_event(self, symbol: str, tf: str, status: str, utc_now: str) -> None:
        """Dispara una notificación Android de HISTORIAL (acumulativa, no
        sonora) cada vez que una vela cruza a CRITICAL/WARNING. No usa
        exclusive=True: si dos pares cruzan casi al mismo tiempo, ambas
        notificaciones deben salir, no cancelarse entre sí.
        """
        self.run_worker(
            self._alerts.notify_history(symbol, tf, status, utc_now),
            group="alert-history",
        )

    @staticmethod
    def _read_notification_volume() -> Optional[tuple[int, int]]:
        """Lee el stream notification con Termux:API, si está disponible."""
        try:
            result = subprocess.run(
                ["termux-volume"],
                timeout=5,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                LOGGER.warning("termux-volume exited with status %s", result.returncode)
                return None
            streams = json.loads(result.stdout)
            for stream in streams:
                if stream.get("stream") == "notification":
                    return int(stream["volume"]), int(stream["max_volume"])
        except Exception as exc:
            LOGGER.warning("Could not read Android notification volume: %s", exc)
        return None

    async def _load_notification_volume(self) -> None:
        volume = await asyncio.to_thread(self._read_notification_volume)
        if volume is not None:
            self.query_one("#android-volume", AndroidVolumeControl).set_range(*volume)

    def apply_notification_volume(self, value: int) -> None:
        """Agenda el cambio al soltar la barra, sin bloquear la interfaz."""
        self.run_worker(
            self._set_notification_volume(value),
            group="android-volume-set",
            exclusive=True,
        )

    async def _set_notification_volume(self, value: int) -> None:
        def set_volume() -> bool:
            try:
                result = subprocess.run(
                    ["termux-volume", "notification", str(value)],
                    timeout=5,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                if result.returncode:
                    LOGGER.warning("termux-volume exited with status %s", result.returncode)
                    return False
                return True
            except Exception as exc:
                LOGGER.warning("Could not set Android notification volume: %s", exc)
                return False

        if await asyncio.to_thread(set_volume):
            self.notify(f"Volumen de alertas: {value}", severity="information")
        else:
            self.notify(
                "No se pudo cambiar volumen: verifica pkg install termux-api",
                severity="warning",
            )

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

                # Solo declaramos una caída general si todos los fetches intentados
                # fallaron por red/exchange; un símbolo aislado no dispara el aviso.
                connectivity_errors = [
                    error for _, _, _, error in results
                    if error and error.startswith(("RED:", "BLOQUEADO:"))
                ]
                all_attempts_lost = bool(results) and len(connectivity_errors) == len(results)
                if all_attempts_lost and not self._connectivity_down:
                    self._connectivity_down = True
                    await self._alerts.notify_connectivity(recovered=False)
                elif self._connectivity_down and any(error is None for _, _, _, error in results):
                    self._connectivity_down = False
                    await self._alerts.notify_connectivity(recovered=True)

                critical_details: list[str] = []
                warning_details: list[str] = []

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
                            critical_details.append(f"{symbol_r} {tf_r}")
                        elif not panel.muted and w:
                            warning_details.append(f"{symbol_r} {tf_r}")
                    except Exception as e:
                        LOGGER.exception("Panel update %s", wid)
                        self.notify(f"Panel update {wid}: {e}", severity="warning")

                await self._alerts.alert(
                    bool(critical_details), bool(warning_details),
                    critical_details or warning_details,
                )

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
