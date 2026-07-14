"""
kde_velas_diarias.py — Sentinel Quant
=====================================
KDE Report de velas diarias (1D) segmentadas por color UP/DOWN.

INPUT:  {ACTIVO}_USDT_1d.parquet   (columnas: timestamp, open, high, low, close,
                                     volume, quote_vol, taker_buy_quote_vol)
OUTPUT: kde_{activo}_up.png
        kde_{activo}_down.png
        kde_{activo}_comparacion.png   (UP vs DOWN superpuestos)
        doji_{activo}.csv              (log de velas excluidas por close==open)

Variables analizadas (todas derivadas de OHLC + quote_vol, sin camino intradía):
  - correcto_pct    : % de rendimiento a favor del color, medido en su punto más
                       extremo del día contra el open
                         UP   -> (high - open) / open * 100
                         DOWN -> (open - low)  / open * 100
  - contrario_pct   : % de desviación hacia el lado contrario antes de resolver
                       en el color final, medido siempre contra el open (valor
                       absoluto — mismo lado para UP y DOWN)
                         UP   -> abs((low  - open) / open) * 100
                         DOWN -> abs((high - open) / open) * 100
  - body_pct        : % del rango total que ocupa el cuerpo de la vela
                         abs(close - open) / (high - low) * 100
  - range_log       : log range simétrico (Parkinson) — ln(high / low)
                         (high - low) / open * 100
  - quote_vol       : volumen en quote asset (USDT), escala lineal
  - day_of_week     : histograma discreto Lun..Dom

Velas DOJI (close == open exacto) se EXCLUYEN de UP/DOWN y se guardan en un CSV
aparte con su fecha para revisión manual.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import gaussian_kde, skew, kurtosis
from pathlib import Path
import logging
import sys

# ─── CONFIG ────────────────────────────────────────────────────────────────────

DATA_DIR   = Path("/home/jonh/Documentos/proyectos_trading/sentinel/data")
OUTPUT_DIR = Path("/home/jonh/Documentos/proyectos_trading/ANALISIS VELAS DIARIAS/kde_output")

ACTIVOS = ["BTC", "ETH"]   # el pipeline ya trae ambos — se generan reportes para cada uno
TF      = "1d"

STYLE = {
    "bg_fig"    : "#0a0e1a",
    "bg_ax"     : "#0d1117",
    "color_kde" : "#00bcd4",
    "color_fill": "#00bcd4",
    "spine"     : "#333333",
    "tick"      : "white",
    "title"     : "white",
    "label"     : "white",
    "grid"      : "#1a1f2e",
}

HITO_COLORS = {
    "min"    : "#2196f3",
    "moda"   : "#e91e63",
    "mediana": "#4caf50",
    "media"  : "#ff9800",
    "p95"    : "#9e9e9e",
    "p99"    : "#616161",
    "max"    : "#37474f",
}

# Colores de comparación UP vs DOWN (Vista B)
COLOR_UP   = "#4caf50"
COLOR_DOWN = "#e91e63"

# ─── LOGGING ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ─── HELPERS DE ESTILO (reusados tal cual de kde_poblaciones.py) ──────────────

def estilo_ax(ax):
    ax.set_facecolor(STYLE["bg_ax"])
    ax.tick_params(colors=STYLE["tick"], labelsize=7)
    ax.spines[:].set_color(STYLE["spine"])
    ax.grid(True, color=STYLE["grid"], linewidth=0.4, alpha=0.6)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(STYLE["label"])


def calcular_hitos(datos):
    media = float(np.mean(datos))
    std   = float(np.std(datos))
    return {
        "moda"   : float(datos[np.argmax(gaussian_kde(datos)(datos))]) if len(np.unique(datos)) > 1 else float(np.median(datos)),
        "mediana": float(np.median(datos)),
        "media"  : media,
        "p01"    : float(np.percentile(datos, 1)),
        "p05"    : float(np.percentile(datos, 5)),
        "p95"    : float(np.percentile(datos, 95)),
        "p99"    : float(np.percentile(datos, 99)),
        "max"    : float(np.max(datos)),
        "min"    : float(np.min(datos)),
        "std"    : std,
        "cv"     : float(std / media) if media != 0 else np.nan,
        "skew"   : float(skew(datos, bias=False)) if len(datos) > 2 else np.nan,
        "kurt"   : float(kurtosis(datos, fisher=True, bias=False)) if len(datos) > 3 else np.nan,
        "n"      : len(datos),
    }


def panel_kde(ax, datos, titulo, unidad="", color=None):
    """Dibuja KDE + hitos + leyenda en un eje."""
    color = color or STYLE["color_kde"]
    datos = np.asarray(datos, dtype=float)
    datos = datos[np.isfinite(datos)]

    if len(datos) < 2 or len(np.unique(datos)) < 2:
        ax.text(0.5, 0.5, "Datos insuficientes", ha="center", va="center",
                color="white", transform=ax.transAxes)
        estilo_ax(ax)
        ax.set_title(titulo, fontsize=9, color=STYLE["title"], pad=4)
        return {}

    h = calcular_hitos(datos)

    try:
        kde = gaussian_kde(datos, bw_method="scott")
        xs  = np.linspace(h["min"], h["max"], 500)
        ys  = kde(xs)
        ax.fill_between(xs, ys, alpha=0.20, color=color)
        ax.plot(xs, ys, color=color, linewidth=1.6)
    except Exception:
        ax.hist(datos, bins=30, color=color, alpha=0.4, density=True)

    hitos_plot = [
        ("min",     h["min"],     f"Min    {h['min']:.2f}{unidad}"),
        ("moda",    h["moda"],    f"Moda   {h['moda']:.2f}{unidad}"),
        ("mediana", h["mediana"], f"Med    {h['mediana']:.2f}{unidad}"),
        ("media",   h["media"],   f"Media  {h['media']:.2f}{unidad}"),
        ("p95",     h["p95"],     f"P95    {h['p95']:.2f}{unidad}"),
        ("p99",     h["p99"],     f"P99    {h['p99']:.2f}{unidad}"),
        ("max",     h["max"],     f"Max    {h['max']:.2f}{unidad}"),
    ]
    handles = []
    for key, val, label in hitos_plot:
        c = HITO_COLORS[key]
        ax.axvline(val, color=c, linewidth=1.1, linestyle="--", alpha=0.85)
        handles.append(mlines.Line2D([], [], color=c, linewidth=1.1,
                                     linestyle="--", label=label))

    ax.legend(handles=handles, fontsize=6, loc="upper right",
              facecolor="#1a1f2e", edgecolor="#333", labelcolor="white",
              framealpha=0.85)

    cv_txt   = f"{h['cv']*100:.1f}%" if np.isfinite(h["cv"]) else "n/a"
    stats_txt = (
        f"Std     {h['std']:.2f}{unidad}\n"
        f"CV      {cv_txt}\n"
        f"Skew    {h['skew']:+.2f}\n"
        f"Kurt    {h['kurt']:+.2f}"
    )
    ax.text(0.02, 0.94, stats_txt, transform=ax.transAxes,
            fontsize=6, color="white", va="top", ha="left",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#1a1f2e",
                       edgecolor="#333", alpha=0.85))

    ax.set_title(f"{titulo}  (n={h['n']})", fontsize=9,
                 color=STYLE["title"], pad=4)
    estilo_ax(ax)
    return h


def panel_ecdf(ax, datos, titulo, unidad="", color=None):
    """Dibuja ECDF en un eje."""
    color = color or STYLE["color_kde"]
    datos = np.asarray(datos, dtype=float)
    datos = datos[np.isfinite(datos)]
    if len(datos) < 2:
        estilo_ax(ax)
        return
    xs = np.sort(datos)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.plot(xs, ys, color=color, linewidth=1.6)
    ax.axhline(0.5, color=HITO_COLORS["mediana"], linewidth=0.8,
               linestyle="--", alpha=0.7)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"ECDF — {titulo}", fontsize=9, color=STYLE["title"], pad=4)
    estilo_ax(ax)


def panel_hist_discreto(ax, datos, titulo, unidad="", color=None):
    """Histograma para variables discretas (day_of_week)."""
    color = color or STYLE["color_kde"]
    datos = np.asarray(datos)
    datos = datos[np.isfinite(datos.astype(float))].astype(int)
    vals, counts = np.unique(datos, return_counts=True)
    ax.bar(vals, counts / counts.sum(), color=color,
           alpha=0.7, width=0.6, edgecolor=STYLE["spine"])
    ax.set_xticks(vals)
    dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    ax.set_xticklabels([dias[v] for v in vals], fontsize=7)
    n = len(datos)
    ax.set_title(f"{titulo}  (n={n})", fontsize=9,
                 color=STYLE["title"], pad=4)
    estilo_ax(ax)


# ─── CÁLCULO DE MÉTRICAS ───────────────────────────────────────────────────────

VARIABLES = [
    # (columna,          titulo,                        unidad, discreta)
    ("correcto_pct",     "Rendimiento Sentido Correcto (%)",  "%", False),
    ("contrario_pct",    "Desviación Sentido Contrario (%)",  "%", False),
    ("body_pct",         "% Cuerpo vs Rango",                 "%", False),
    ("range_log",        "Rango Total (log)",                 "",  False),
    ("quote_vol",        "Volumen Quote (USDT)",              "",  False),
    ("day_of_week",      "Día de la semana",                  "",  True),
]

N_VARS = len(VARIABLES)


def calcular_metricas(df: pd.DataFrame, activo: str):
    """
    Agrega columnas derivadas, separa DOJI y devuelve (df_limpio, df_doji).
    """
    df = df.copy()

    # Detección de unidad del timestamp — evita el bug de interpretar ms como ns
    # (que colapsa todas las fechas cerca de 1970-01-01, un jueves, explicando
    # el "Día de la semana" mal graficado en la primera corrida).
    ts_col = df["timestamp"]
    if pd.api.types.is_datetime64_any_dtype(ts_col):
        df["timestamp"] = pd.to_datetime(ts_col, utc=True)
    else:
        sample = float(pd.to_numeric(ts_col, errors="coerce").dropna().iloc[0])
        if sample > 1e17:      # ya viene en nanosegundos
            unit = "ns"
        elif sample > 1e14:    # microsegundos
            unit = "us"
        elif sample > 1e11:    # milisegundos (típico Binance)
            unit = "ms"
        else:                  # segundos
            unit = "s"
        df["timestamp"] = pd.to_datetime(ts_col, unit=unit, utc=True)

    is_doji = df["close"] == df["open"]
    df_doji = df.loc[is_doji, ["timestamp", "open", "high", "low", "close"]].copy()
    if len(df_doji) > 0:
        log.warning(f"   ⚠️  {activo}: {len(df_doji)} vela(s) DOJI detectada(s) — excluidas, "
                     f"logueadas en doji_{activo.lower()}.csv")

    df = df.loc[~is_doji].copy()
    df["direccion"] = np.where(df["close"] > df["open"], "UP", "DOWN")

    up_mask   = df["direccion"] == "UP"
    down_mask = ~up_mask

    correcto_pct  = pd.Series(index=df.index, dtype=float)
    contrario_pct = pd.Series(index=df.index, dtype=float)

    # UP: correcto contra el high, contrario contra el low — ambos normalizados por open
    correcto_pct[up_mask]  = (df.loc[up_mask, "high"] - df.loc[up_mask, "open"]) / df.loc[up_mask, "open"] * 100
    contrario_pct[up_mask] = ((df.loc[up_mask, "low"] - df.loc[up_mask, "open"]) / df.loc[up_mask, "open"] * 100).abs()

    # DOWN: correcto contra el low, contrario contra el high — ambos normalizados por open
    correcto_pct[down_mask]  = (df.loc[down_mask, "open"] - df.loc[down_mask, "low"]) / df.loc[down_mask, "open"] * 100
    contrario_pct[down_mask] = ((df.loc[down_mask, "high"] - df.loc[down_mask, "open"]) / df.loc[down_mask, "open"] * 100).abs()

    df["correcto_pct"]  = correcto_pct
    df["contrario_pct"] = contrario_pct
    df["body_pct"]  = (df["close"] - df["open"]).abs() / (df["high"] - df["low"]) * 100

    # range_log: log range simétrico, sin necesidad de distinguir UP/DOWN
    # (equivalente al usado en el estimador de volatilidad de Parkinson)
    df["range_log"] = np.log(df["high"] / df["low"])

    df["quote_vol"] = df["quote_vol"]
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Lun ... 6=Dom

    return df, df_doji


# ─── EXPORT CSV DE MÉTRICAS (una fila por vela, separado UP/DOWN) ────────────

COLUMNAS_EXPORT = [
    "timestamp", "open", "high", "low", "close",
    "volume", "quote_vol", "taker_buy_quote_vol",
    "direccion",
    "correcto_pct", "contrario_pct", "body_pct", "range_log", "day_of_week",
]


def exportar_csv(df: pd.DataFrame, activo: str, output_dir: Path):
    """Guarda un CSV por dirección con todas las métricas calculadas, fila = 1 vela."""
    for direccion in ["UP", "DOWN"]:
        subset = df.loc[df["direccion"] == direccion, COLUMNAS_EXPORT].copy()
        subset = subset.sort_values("timestamp")
        fname = output_dir / f"metricas_{activo.lower()}_{direccion.lower()}.csv"
        subset.to_csv(fname, index=False)
        log.info(f"   📝 CSV guardado: {fname.name}  ({len(subset)} filas)")




def generar_reporte(df: pd.DataFrame, activo: str, direccion: str, output_dir: Path):
    subset = df[df["direccion"] == direccion].copy()
    n = len(subset)

    if n < 5:
        log.warning(f"⚠️  {activo}-{direccion}: solo {n} filas — saltando")
        return

    log.info(f"   Generando {activo}-{direccion} (n={n}) ...")

    fig, axes = plt.subplots(
        N_VARS, 2,
        figsize=(14, N_VARS * 2.8),
        facecolor=STYLE["bg_fig"]
    )

    fecha_min = df["timestamp"].min().strftime("%Y-%m-%d")
    fecha_max = df["timestamp"].max().strftime("%Y-%m-%d")

    fig.suptitle(
        f"KDE Report — Población {direccion} — Velas Diarias — {activo}/USDT 1D\n"
        f"n={n} velas | {fecha_min} → {fecha_max}",
        fontsize=12, color="white", y=0.995
    )

    for i, (col, titulo, unidad, discreta) in enumerate(VARIABLES):
        datos = subset[col].dropna().values
        ax_izq = axes[i, 0]
        ax_der = axes[i, 1]

        if discreta:
            panel_hist_discreto(ax_izq, datos, titulo, unidad)
            ax_der.set_visible(False)
        else:
            panel_kde(ax_izq, datos, f"KDE — {titulo}", unidad)
            panel_ecdf(ax_der, datos, titulo, unidad)

    plt.tight_layout(rect=[0, 0, 1, 0.995])

    fname = output_dir / f"kde_{activo.lower()}_{direccion.lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=STYLE["bg_fig"])
    plt.close()
    log.info(f"   ✅ Guardado: {fname.name}")


# ─── REPORTE COMPARATIVO UP vs DOWN (Vista B) ──────────────────────────────────

def generar_comparacion(df: pd.DataFrame, activo: str, output_dir: Path):
    n_up   = (df["direccion"] == "UP").sum()
    n_down = (df["direccion"] == "DOWN").sum()

    log.info(f"   Generando comparación {activo} (UP n={n_up} | DOWN n={n_down}) ...")

    continuas = [v for v in VARIABLES if not v[3]]
    n_cont = len(continuas)

    fig, axes = plt.subplots(
        n_cont + 1, 2,
        figsize=(14, (n_cont + 1) * 2.8),
        facecolor=STYLE["bg_fig"]
    )

    fecha_min = df["timestamp"].min().strftime("%Y-%m-%d")
    fecha_max = df["timestamp"].max().strftime("%Y-%m-%d")

    fig.suptitle(
        f"KDE Comparación — UP vs DOWN — Velas Diarias — {activo}/USDT 1D\n"
        f"n_UP={n_up} | n_DOWN={n_down} | {fecha_min} → {fecha_max}",
        fontsize=12, color="white", y=0.995
    )

    up_data   = df[df["direccion"] == "UP"]
    down_data = df[df["direccion"] == "DOWN"]

    for i, (col, titulo, unidad, _) in enumerate(continuas):
        ax_izq = axes[i, 0]
        ax_der = axes[i, 1]

        d_up   = up_data[col].dropna().values
        d_down = down_data[col].dropna().values

        # KDE superpuesto (sin caja de stats individual para no saturar)
        for datos, color, label in [(d_up, COLOR_UP, "UP"), (d_down, COLOR_DOWN, "DOWN")]:
            datos = np.asarray(datos, dtype=float)
            datos = datos[np.isfinite(datos)]
            if len(datos) < 2 or len(np.unique(datos)) < 2:
                continue
            kde = gaussian_kde(datos, bw_method="scott")
            xs  = np.linspace(datos.min(), datos.max(), 500)
            ys  = kde(xs)
            ax_izq.fill_between(xs, ys, alpha=0.15, color=color)
            ax_izq.plot(xs, ys, color=color, linewidth=1.6, label=label)

        ax_izq.legend(fontsize=7, loc="upper right", facecolor="#1a1f2e",
                      edgecolor="#333", labelcolor="white", framealpha=0.85)
        ax_izq.set_title(f"KDE — {titulo}", fontsize=9, color=STYLE["title"], pad=4)
        estilo_ax(ax_izq)

        # ECDF superpuesto
        for datos, color, label in [(d_up, COLOR_UP, "UP"), (d_down, COLOR_DOWN, "DOWN")]:
            datos = np.asarray(datos, dtype=float)
            datos = datos[np.isfinite(datos)]
            if len(datos) < 2:
                continue
            xs = np.sort(datos)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax_der.plot(xs, ys, color=color, linewidth=1.6, label=label)

        ax_der.legend(fontsize=7, loc="lower right", facecolor="#1a1f2e",
                      edgecolor="#333", labelcolor="white", framealpha=0.85)
        ax_der.set_ylim(0, 1.02)
        ax_der.set_title(f"ECDF — {titulo}", fontsize=9, color=STYLE["title"], pad=4)
        estilo_ax(ax_der)

    # Día de la semana: barras agrupadas UP vs DOWN
    ax_dow = axes[n_cont, 0]
    axes[n_cont, 1].set_visible(False)

    dow_up   = up_data["day_of_week"].dropna().astype(int).values
    dow_down = down_data["day_of_week"].dropna().astype(int).values
    dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    x = np.arange(7)
    width = 0.35

    counts_up   = np.array([np.sum(dow_up == d)   for d in range(7)]) / max(len(dow_up), 1)
    counts_down = np.array([np.sum(dow_down == d) for d in range(7)]) / max(len(dow_down), 1)

    ax_dow.bar(x - width/2, counts_up,   width, color=COLOR_UP,   alpha=0.8, label="UP")
    ax_dow.bar(x + width/2, counts_down, width, color=COLOR_DOWN, alpha=0.8, label="DOWN")
    ax_dow.set_xticks(x)
    ax_dow.set_xticklabels(dias, fontsize=8)
    ax_dow.legend(fontsize=7, loc="upper right", facecolor="#1a1f2e",
                  edgecolor="#333", labelcolor="white", framealpha=0.85)
    ax_dow.set_title("Día de la semana — UP vs DOWN", fontsize=9, color=STYLE["title"], pad=4)
    estilo_ax(ax_dow)

    plt.tight_layout(rect=[0, 0, 1, 0.995])

    fname = output_dir / f"kde_{activo.lower()}_comparacion.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=STYLE["bg_fig"])
    plt.close()
    log.info(f"   ✅ Guardado: {fname.name}")


# ─── CONSOLIDADO POR ACTIVO (tabla resumen UP vs DOWN) ────────────────────────

def generar_consolidado(df: pd.DataFrame, activo: str, output_dir: Path):
    """Tabla resumen: por cada métrica continua, hitos de UP vs DOWN lado a lado."""
    continuas = [v for v in VARIABLES if not v[3]]

    up_data   = df[df["direccion"] == "UP"]
    down_data = df[df["direccion"] == "DOWN"]
    n_up, n_down = len(up_data), len(down_data)

    filas = []
    for col, titulo, unidad, _ in continuas:
        for direccion, data in [("UP", up_data), ("DOWN", down_data)]:
            datos = data[col].dropna().values
            datos = np.asarray(datos, dtype=float)
            datos = datos[np.isfinite(datos)]
            if len(datos) < 2:
                continue
            h = calcular_hitos(datos)
            fmt = (lambda v: f"{v:,.2f}") if unidad != "" or col != "quote_vol" else (lambda v: f"{v:,.0f}")
            cv_txt = f"{h['cv']*100:.1f}%" if np.isfinite(h["cv"]) else "n/a"
            filas.append([
                titulo, direccion,
                fmt(h["min"]) + unidad, fmt(h["moda"]) + unidad, fmt(h["mediana"]) + unidad, fmt(h["media"]) + unidad,
                fmt(h["std"]) + unidad, cv_txt, f"{h['skew']:+.2f}", f"{h['kurt']:+.2f}",
                fmt(h["p95"]) + unidad, fmt(h["p99"]) + unidad, fmt(h["max"]) + unidad,
            ])

    columnas = ["Métrica", "Dir.", "Min", "Moda", "Mediana", "Media", "Std", "CV", "Skew", "Kurt",
                "P95", "P99", "Max"]

    fecha_min = df["timestamp"].min().strftime("%Y-%m-%d")
    fecha_max = df["timestamp"].max().strftime("%Y-%m-%d")

    fig_h = 1.2 + 0.42 * len(filas)
    fig, ax = plt.subplots(figsize=(17, fig_h), facecolor=STYLE["bg_fig"])
    ax.axis("off")

    fig.suptitle(
        f"Consolidado — {activo}/USDT 1D — UP vs DOWN\n"
        f"n_UP={n_up} | n_DOWN={n_down} | {fecha_min} → {fecha_max}",
        fontsize=13, color="white", y=0.99
    )

    tabla = ax.table(cellText := filas, colLabels=columnas, loc="center", cellLoc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(8)
    tabla.scale(1, 1.6)

    for (row, col_idx), cell in tabla.get_celld().items():
        cell.set_edgecolor(STYLE["spine"])
        if row == 0:
            cell.set_facecolor("#1a1f2e")
            cell.set_text_props(color="white", weight="bold")
        else:
            metrica, direccion = filas[row - 1][0], filas[row - 1][1]
            if col_idx == 1:
                cell.set_text_props(
                    color=COLOR_UP if direccion == "UP" else COLOR_DOWN, weight="bold"
                )
            else:
                cell.set_text_props(color="white")
            # sombreado alternado por bloque de métrica (UP+DOWN)
            bloque = (row - 1) // 2
            cell.set_facecolor(STYLE["bg_ax"] if bloque % 2 == 0 else "#11151f")

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    fname = output_dir / f"consolidado_{activo.lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=STYLE["bg_fig"])
    plt.close()
    log.info(f"   ✅ Guardado: {fname.name}")




def main():
    log.info("══ kde_velas_diarias.py — Sentinel Quant ══")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for activo in ACTIVOS:
        path = DATA_DIR / f"{activo}_USDT_{TF}.parquet"
        if not path.exists():
            log.warning(f"⚠️  {path.name} no encontrado — saltando {activo}")
            continue

        log.info(f"\n── {activo} — leyendo {path.name} ──")
        df_raw = pd.read_parquet(path)

        df, df_doji = calcular_metricas(df_raw, activo)
        log.info(f"   {activo}: {len(df)} velas válidas "
                 f"(UP={ (df['direccion']=='UP').sum() } | "
                 f"DOWN={ (df['direccion']=='DOWN').sum() }) "
                 f"| {len(df_doji)} DOJI excluidas")

        if len(df_doji) > 0:
            doji_path = OUTPUT_DIR / f"doji_{activo.lower()}.csv"
            df_doji.to_csv(doji_path, index=False)
            log.info(f"   📝 Log DOJI guardado: {doji_path.name}")

        exportar_csv(df, activo, OUTPUT_DIR)

        for direccion in ["UP", "DOWN"]:
            generar_reporte(df, activo, direccion, OUTPUT_DIR)

        generar_comparacion(df, activo, OUTPUT_DIR)
        generar_consolidado(df, activo, OUTPUT_DIR)

    log.info("\n══ DONE ══")
    log.info(f"   Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()