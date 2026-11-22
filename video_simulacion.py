"""Vídeo realidad vs. conductor promedio.

Izquierda: el vídeo real. Derecha: vista cenital donde, desde el instante inicial,
los vehículos de dentro de la zona los mueve el SIMULADOR (rellenos, color = velocidad)
y en contorno blanco se ve dónde estaba de verdad cada uno. Los que entran en escena
lo hacen cuando y como entraron en la realidad; dentro decide el conductor promedio.

Uso: .venv/bin/python video_simulacion.py salidas/lista/01_wqctLW0Hb_0 --trozo 004 --duracion 60
"""
import argparse
import json
import subprocess
from argparse import Namespace
from pathlib import Path

import cv2
import joblib
import numpy as np

from calibrar import fondo
from simulador import escena
from trayectorias import Cenital, Suelo


def color_vel(kmh):
    t = np.uint8([[[int(np.clip(kmh / 120 * 255, 0, 255))]]])
    return tuple(int(c) for c in cv2.applyColorMap(t, cv2.COLORMAP_TURBO)[0, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("camara")
    ap.add_argument("--trozo", default="004")
    ap.add_argument("--desde", type=float, default=0.0)
    ap.add_argument("--duracion", type=float, default=60.0)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--comportamiento", default="salidas/lista/comportamiento.json")
    ap.add_argument("--salida", default=None)
    a = ap.parse_args()
    cam = Path(a.camara)
    salida = Path(a.salida or cam / f"realidad_vs_simulacion_{a.trozo}.mp4")
    base, k = salida, 2
    while salida.exists():   # nunca pisar un vídeo anterior
        salida = base.with_name(f"{base.stem}_v{k}{base.suffix}"); k += 1

    info = json.load(open(a.comportamiento))[cam.name]
    modelo = joblib.load(Path(a.comportamiento).with_suffix(".decision.joblib"))
    crudo = json.load(open(cam / f"estados_{a.trozo}.json"))
    pistas = json.load(open(cam / f"pistas_{a.trozo}.json"))
    fps, W, H = pistas["fps"], pistas["ancho"], pistas["alto"]
    suelo = Suelo(json.load(open(cam / "calibracion.json")), W)
    th = crudo["rumbo_carretera"]
    u = np.array([np.cos(th), np.sin(th)]); n = np.array([-u[1], u[0]])

    # ---- simular los dos sentidos
    sims = {}
    for sen in ("alejan", "acercan"):
        ns = Namespace(camara=str(cam), trozo=a.trozo, sentido=sen, desde=a.desde,
                       duracion=a.duracion, semilla=a.semilla, comportamiento=a.comportamiento)
        sims[sen] = escena(ns, info, modelo)
    # fotograma del vídeo = fotograma concatenado - desplazamiento del trozo
    raw_min = min(e[0] for e in crudo["estados"])
    f0_video = int(raw_min + a.desde * fps)

    # registro simulado -> por instante (múltiplos de 0,1 s)
    por_t = {}
    for sen, r in sims.items():
        reg = r["registro"]
        c = np.asarray(r["centros"])
        carr = np.arange(1, len(c) + 1)
        izq = np.interp(reg.lat.values, carr, c)
        sg = 1 if sen == "alejan" else -1
        P = np.outer(sg * reg.x.values, u) + np.outer(sg * izq, n)
        rumbo = th if sen == "alejan" else th + np.pi
        for t, vid, X, Z, v, clase in zip(reg.t.values, reg.id.values, P[:, 0], P[:, 1], reg.v.values,
                                          reg.clase.values):
            por_t.setdefault(round(float(t), 1), {})[(sen, vid)] = (X, Z, v, clase, rumbo)
    reales = {}
    for f, vid, clase, X, Z, v, rb in crudo["estados"]:
        reales.setdefault(f, []).append((X, Z, rb, clase))

    video = pistas["video"]
    cen = Cenital(suelo, H, fondo(video))
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0_video)
    tmp = salida.with_suffix(".tmp.mp4")
    out = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W + cen.W, H))
    n_frames = int(a.duracion * fps)
    for k in range(n_frames):
        ok, img = cap.read()
        if not ok:
            break
        t = k / fps
        ta = np.floor(t * 10) / 10; tb = round(ta + 0.1, 1); w = (t - ta) / 0.1
        A, B = por_t.get(round(ta, 1), {}), por_t.get(tb, {})
        cen.nuevo()
        # realidad: contorno blanco
        for X, Z, rb, clase in reales.get(f0_video + k, []):
            cen.vehiculo((X, Z), rb, clase, (255, 255, 255), relleno=False)
        # simulación: relleno, color = velocidad
        for key, (X, Z, v, clase, rumbo) in A.items():
            if key in B:
                X2, Z2, v2, *_ = B[key]
                X, Z, v = X + (X2 - X) * w, Z + (Z2 - Z) * w, v + (v2 - v) * w
            cen.vehiculo((X, Z), rumbo, clase, color_vel(v * 3.6), texto=f"{v * 3.6:.0f}")
        derecha = cen.final()
        zona = derecha[22:96]; zona[:] = (zona * 0.3).astype(np.uint8)
        for j, (txt, e) in enumerate([("AVERAGE-DRIVER SIMULATION", 0.55),
                                      ("filled = simulated (colour = speed)", 0.45),
                                      ("white outline = what really happened", 0.45),
                                      (f"t = {t:4.1f} s since simulation start", 0.45)]):
            cv2.putText(derecha, txt, (8, 40 + 17 * j), cv2.FONT_HERSHEY_SIMPLEX, e, (255, 255, 255),
                        1, cv2.LINE_AA)
        img[:32, :300] = (img[:32, :300] * 0.35).astype(np.uint8)
        cv2.putText(img, "REAL VIDEO", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)
        out.write(np.hstack([img, derecha]))
    out.release(); cap.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-crf", "22",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(salida)], check=True)
    tmp.unlink()
    print(f"Vídeo -> {salida}")


if __name__ == "__main__":
    main()
