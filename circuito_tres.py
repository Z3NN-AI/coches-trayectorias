"""Tres circuitos de Sugiyama lado a lado (vídeo para el portafolio):
  1. IDM + MOBIL de manual (parámetros estándar de Treiber, idénticos, sin ruido).
     En un circuito de un carril MOBIL no tiene a dónde cambiar: solo actúa el IDM.
  2. Nuestros conductores promedio (aprendidos de vídeo real).
  3. Los mismos conductores promedio + 1 coche robot amortiguador (Stern et al. 2018).

Uso: .venv/bin/python circuito_tres.py [--duracion 600]
"""
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

from circuito import DT, Circuito, params
from comportamiento import MANUAL

W, H, FPS, ACEL = 1920, 1080, 20, 4
F = cv2.FONT_HERSHEY_SIMPLEX


def color(kmh, vmax=25):
    t = np.uint8([[[int(np.clip(kmh / vmax * 255, 0, 255))]]])
    return tuple(int(q) for q in cv2.applyColorMap(t, cv2.COLORMAP_TURBO)[0, 0])


def texto_c(img, s, cx, y, e, g=1, col=(255, 255, 255)):
    (w, _), _ = cv2.getTextSize(s, F, e, g)
    cv2.putText(img, s, (int(cx - w / 2), y), F, e, col, g, cv2.LINE_AA)


def anillo(img, cx, cy, R, c, titulo, sub, ref, hist, k_km):
    cv2.circle(img, (cx, cy), R + 16, (70, 70, 70), 34)
    cv2.circle(img, (cx, cy), R + 33, (230, 230, 230), 2)
    cv2.circle(img, (cx, cy), R, (230, 230, 230), 2)
    L = 4.5 / c.C * 2 * np.pi
    for i in range(c.n):
        ang = c.x[i] / c.C * 2 * np.pi + np.pi / 2
        pts = [(cx + r_ * np.cos(a_), cy - r_ * np.sin(a_))
               for a_, r_ in ((ang, R + 6), (ang, R + 27), (ang - L, R + 27), (ang - L, R + 6))]
        cv2.fillPoly(img, [np.int32(pts)], color(c.v[i] * 3.6), cv2.LINE_AA)
        if c.robot[i]:
            cv2.polylines(img, [np.int32(pts)], True, (255, 255, 255), 4, cv2.LINE_AA)
    texto_c(img, titulo, cx, 70, 1.0, 2)
    texto_c(img, sub, cx, 105, 0.6, 1, (210, 210, 210))
    texto_c(img, ref, cx, 140, 0.7, 2, (120, 220, 255))
    vm = c.v.mean() * 3.6
    parados = int((c.v * 3.6 < 5).sum())
    flujo = k_km * vm
    # estado del tráfico (semáforo) por la dispersión de velocidades del último minuto
    h = np.array(hist[-int(60 / DT):]) if hist else np.zeros((1, 3))
    oscila = h[:, 2].mean() if len(h) else 0
    if oscila > 3 or parados > 0:
        estado, col = "STOP-AND-GO (phantom jam)", (40, 40, 230)
    elif oscila > 1:
        estado, col = "UNSTABLE", (0, 190, 255)
    else:
        estado, col = "SMOOTH FLOW", (70, 200, 70)
    cv2.circle(img, (cx - 120, cy - 70), 16, col, -1, cv2.LINE_AA)
    cv2.putText(img, estado, (cx - 95, cy - 62), F, 0.65, col, 2, cv2.LINE_AA)
    for j, s in enumerate([f"average speed  {vm:5.1f} km/h",
                           f"traffic flow   {flujo:5.0f} veh/h",
                           f"stopped        {parados:2d} of {c.n}",
                           f"slowest car    {c.v.min() * 3.6:5.1f} km/h"]):
        cv2.putText(img, s, (cx - 135, cy - 15 + 34 * j), F, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    # traza: velocidad media (blanco) y del más lento (rojo), último minuto
    x0, y0, w, hh = cx - 260, 1000, 520, 120
    cv2.rectangle(img, (x0, y0 - hh), (x0 + w, y0), (35, 35, 35), -1)
    n = int(60 / DT)
    hh_ = np.array(hist[-n:]) if hist else np.zeros((1, 3))
    for k, cc in ((0, (255, 255, 255)), (1, (60, 60, 255))):
        pts = [(x0 + int(i * w / n), int(y0 - np.clip(v / 25, 0, 1) * hh)) for i, v in enumerate(hh_[:, k])]
        if len(pts) > 1:
            cv2.polylines(img, [np.int32(pts)], False, cc, 2, cv2.LINE_AA)
    texto_c(img, "last 60 s: average speed (white), slowest car (red), 0-25 km/h", cx, y0 + 26, 0.5, 1,
            (200, 200, 200))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=22)
    ap.add_argument("--largo", type=float, default=230.0)
    ap.add_argument("--duracion", type=float, default=600.0)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--salida", default="salidas/lista/circuito_tres_modelos.mp4")
    a = ap.parse_args()
    salida = Path(a.salida); base, k = salida, 2
    while salida.exists():
        salida = base.with_name(f"{base.stem}_v{k}{base.suffix}"); k += 1
    p, T = params("salidas/lista/comportamiento.json", "01_wqctLW0Hb_0")
    v_obj = 30 / 3.6
    rngs = [np.random.default_rng(a.semilla) for _ in range(3)]
    libro = Circuito(a.n, a.largo, dict(MANUAL), {"p50": MANUAL["T"]}, False, v_obj, rngs[0])
    prom = Circuito(a.n, a.largo, p, T, True, v_obj, rngs[1])
    robot = Circuito(a.n, a.largo, p, T, True, v_obj, rngs[2])
    robot.meter_robot(0, 2.0)
    anillos = [
        (libro, "TEXTBOOK IDM + MOBIL", "classic algorithms, identical perfect drivers (MOBIL idle: 1 lane)",
         "Treiber et al. 2000 (IDM) / Kesting et al. 2007 (MOBIL)"),
        (prom, "OUR AVERAGE DRIVERS", "learned from real motorway video: each one different, imperfect",
         "reproduces Sugiyama et al. 2008 (phantom jam)"),
        (robot, "AVERAGE DRIVERS + 1 ROBOT", "same humans + 1 damping robot car (white outline)",
         "reproduces Stern et al. 2018 (one AV dissolves the jam)"),
    ]
    hist = [[], [], []]
    k_km = a.n / (a.largo / 1000)
    tmp = salida.with_suffix(".tmp.mp4")
    out = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    pasos = int(round(ACEL / (FPS * DT)))
    for f in range(int(a.duracion * FPS / ACEL)):
        for _ in range(pasos):
            for j, (c, *_) in enumerate(anillos):
                c.paso()
                hist[j].append((c.v.mean() * 3.6, c.v.min() * 3.6, c.v.std() * 3.6))
        img = np.full((H, W, 3), (38, 72, 38), np.uint8)
        cv2.rectangle(img, (0, 0), (W, 22), (20, 20, 20), -1)
        cv2.putText(img, f"PHANTOM TRAFFIC JAM TEST  |  {a.n} cars on a {a.largo:.0f} m ring "
                         f"({k_km:.0f} veh/km), all asked to drive at 30 km/h, one car starts 1 m out of place "
                         f"(Sugiyama 2008)  |  t = {anillos[0][0].t:4.0f} s  (video x{ACEL})",
                    (10, 16), F, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for j, (c, tit, sub, ref) in enumerate(anillos):
            anillo(img, 320 + 640 * j, 490, 225, c, tit, sub, ref, hist[j], k_km)
        # leyenda de color
        for i in range(240):
            cv2.line(img, (W - 300 + i, H - 22), (W - 300 + i, H - 10), color(i * 25 / 240), 1)
        cv2.putText(img, "car colour = speed: 0", (W - 520, H - 11), F, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "25 km/h", (W - 55, H - 11), F, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        out.write(img)
    out.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-crf", "22",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(salida)], check=True)
    tmp.unlink()
    for (c, tit, *_), h in zip(anillos, hist):
        h = np.array(h[len(h) // 2:])
        print(f"{tit:28s} media {h[:, 0].mean():5.1f} km/h  flujo {k_km * h[:, 0].mean():5.0f} veh/h")
    print(f"Vídeo -> {salida}")


if __name__ == "__main__":
    main()
