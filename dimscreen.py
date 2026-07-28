#!/usr/bin/env python3
"""
DimScreen casero - Control de brillo y temperatura de color para Linux Mint
-----------------------------------------------------------------------------
Usa xrandr (brillo) y redshift (temperatura de color) para cuidar la vista,
con valores basados en literatura de salud visual / ergonomía de pantallas:

  - Brillo: debe acercarse a la luz ambiental. En oficina (300-500 lux),
    50-70% de brillo es lo recomendado; en la tarde/noche, 20-40%.
    Fuente: revisión de brillo de pantalla y fatiga ocular, CareUEyes /
    literatura de optometría sobre Computer Vision Syndrome (CVS).

  - Temperatura de color: ~4000-4500K es el punto de mayor confort visual
    reportado en varios estudios y guías de fabricantes (BenQ), bajando a
    tonos más cálidos (~3400K) de noche para no interferir con el sueño
    (efecto de la luz azul sobre el ritmo circadiano).
    Fuente: EIZO Corporation - EyeCare Dimming Technology; BenQ Knowledge
    Center; "Protecting Your Vision" (revisión de temperatura de color).

  - Regla 20-20-20: cada 20 minutos, mirar algo a 20 pies (~6m) durante
    20 segundos, para relajar el músculo ciliar del ojo.
    Fuente: recomendación estándar en literatura de Computer Vision
    Syndrome (ej. Kaur et al. 2022, "Digital Eye Strain - A Comprehensive
    Review", Journal of Clinical and Diagnostic Research).

Estos son valores de referencia general, no un diagnóstico médico. Si
sientes molestia ocular persistente, consulta a un oftalmólogo/optometrista.

Requisitos (normalmente ya vienen instalados o se instalan fácil en Mint):
    sudo apt install python3-tk x11-xserver-utils redshift
    pip3 install pystray pillow   # opcional, para ocultar en la bandeja

Uso:
    python3 dimscreen.py
"""

import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox

try:
    import pystray
    from PIL import Image, ImageDraw
    TRAY_AVAILABLE = True
except ImportError:
    TRAY_AVAILABLE = False


# ---------- Presets basados en las fuentes citadas arriba ----------
PRESETS = {
    "Mañana / Día":  {"brightness": 70, "temp": 5500},
    "Tarde":         {"brightness": 45, "temp": 4500},
    "Noche":         {"brightness": 25, "temp": 3400},
}

TWENTY_TWENTY_TWENTY_MS = 20 * 60 * 1000  # 20 minutos, en milisegundos


def get_active_outputs():
    """Detecta las salidas de video activas (monitores conectados)."""
    try:
        result = subprocess.run(
            ["xrandr", "--query"], capture_output=True, text=True, check=True
        )
    except FileNotFoundError:
        print("ERROR: no se encontró 'xrandr'. Instálalo con:")
        print("    sudo apt install x11-xserver-utils")
        sys.exit(1)

    outputs = []
    for line in result.stdout.splitlines():
        if " connected" in line:
            outputs.append(line.split()[0])
    return outputs


def has_redshift():
    try:
        subprocess.run(
            ["redshift", "-V"], capture_output=True, text=True, check=True
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def set_brightness(outputs, value):
    for out in outputs:
        subprocess.run(["xrandr", "--output", out, "--brightness", f"{value:.2f}"])


def set_color_temp(redshift_available, kelvin):
    if not redshift_available:
        return
    # -O aplica una temperatura fija de forma inmediata (one-shot)
    subprocess.run(["redshift", "-O", str(kelvin), "-P"])


def reset_color_temp(redshift_available):
    if not redshift_available:
        return
    subprocess.run(["redshift", "-x"])  # quita el ajuste de redshift


def make_tray_image():
    """Ícono simple: círculo amarillo sobre fondo oscuro."""
    img = Image.new("RGB", (64, 64), color=(30, 30, 30))
    d = ImageDraw.Draw(img)
    d.ellipse((14, 14, 50, 50), fill=(255, 205, 60))
    return img


def build_tray_icon(root, apply_preset_fn, quit_fn):
    def show_window(icon, item):
        root.after(0, root.deiconify)

    def preset_action(name):
        return lambda icon, item: root.after(0, lambda: apply_preset_fn(name))

    def quit_action(icon, item):
        icon.stop()
        root.after(0, quit_fn)

    menu = pystray.Menu(
        pystray.MenuItem("Mostrar panel", show_window, default=True),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Mañana / Día", preset_action("Mañana / Día")),
        pystray.MenuItem("Tarde", preset_action("Tarde")),
        pystray.MenuItem("Noche", preset_action("Noche")),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Salir", quit_action),
    )
    return pystray.Icon("dimscreen", make_tray_image(), "Cuidado visual", menu)


def main():
    outputs = get_active_outputs()
    if not outputs:
        print("No se detectó ninguna salida de video activa con xrandr.")
        sys.exit(1)

    redshift_available = has_redshift()

    root = tk.Tk()
    root.title("Cuidado visual - Pantalla")
    root.attributes("-topmost", True)
    root.resizable(False, False)
    root.geometry("340x300+40+40")

    ttk.Label(root, text="Brillo", font=("Sans", 10)).pack(pady=(12, 0))
    brightness_value = ttk.Label(root, text="", font=("Sans", 10, "bold"))
    brightness_value.pack()

    def on_brightness_change(val):
        v = float(val) / 100
        set_brightness(outputs, v)
        brightness_value.config(text=f"{int(float(val))}%")

    brightness_slider = ttk.Scale(
        root, from_=10, to=100, orient="horizontal",
        command=on_brightness_change, length=300
    )
    brightness_slider.pack(pady=4)

    if redshift_available:
        ttk.Label(root, text="Temperatura de color (K)", font=("Sans", 10)).pack(pady=(10, 0))
        temp_value = ttk.Label(root, text="", font=("Sans", 10, "bold"))
        temp_value.pack()

        def on_temp_change(val):
            k = int(float(val))
            set_color_temp(redshift_available, k)
            temp_value.config(text=f"{k}K")

        temp_slider = ttk.Scale(
            root, from_=3000, to=6500, orient="horizontal",
            command=on_temp_change, length=300
        )
        temp_slider.pack(pady=4)
    else:
        ttk.Label(
            root,
            text="(redshift no instalado: sin control de temperatura de color.\n"
                 "sudo apt install redshift)",
            font=("Sans", 8), foreground="#888"
        ).pack(pady=(6, 0))
        temp_slider = None

    ttk.Label(root, text="Presets según hora del día", font=("Sans", 10)).pack(pady=(14, 4))

    preset_frame = ttk.Frame(root)
    preset_frame.pack()

    def apply_preset(name):
        p = PRESETS[name]
        brightness_slider.set(p["brightness"])
        on_brightness_change(p["brightness"])
        if temp_slider is not None:
            temp_slider.set(p["temp"])
            on_temp_change(p["temp"])

    for name in PRESETS:
        ttk.Button(
            preset_frame, text=name, command=lambda n=name: apply_preset(n)
        ).pack(side="left", padx=4)

    tray_state = {"icon": None}

    def hide_to_tray():
        if TRAY_AVAILABLE:
            root.withdraw()
            if tray_state["icon"] is None:
                icon = build_tray_icon(root, apply_preset, on_close)
                tray_state["icon"] = icon
                threading.Thread(target=icon.run, daemon=True).start()
        else:
            messagebox.showinfo(
                "Falta una librería",
                "Para ocultar en la bandeja del sistema, instala:\n\n"
                "    pip3 install pystray pillow\n\n"
                "Por ahora solo minimizo la ventana a la barra de tareas."
            )
            root.iconify()

    ttk.Button(root, text="Ocultar (bandeja)", command=hide_to_tray).pack(pady=(10, 0))

    # ---------- Recordatorio regla 20-20-20 ----------
    def remind_20_20_20():
        messagebox.showinfo(
            "Descansa la vista",
            "Regla 20-20-20:\n\n"
            "Mira algo a 20 pies (~6 metros) de distancia\n"
            "durante 20 segundos para relajar tus ojos."
        )
        root.after(TWENTY_TWENTY_TWENTY_MS, remind_20_20_20)

    root.after(TWENTY_TWENTY_TWENTY_MS, remind_20_20_20)

    # Preset inicial cómodo
    apply_preset("Mañana / Día")

    def on_close():
        set_brightness(outputs, 1.0)
        reset_color_temp(redshift_available)
        if tray_state["icon"] is not None:
            try:
                tray_state["icon"].stop()
            except Exception:
                pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
