"""Paso 1: detecta y sigue vehículos en un vídeo y guarda las pistas en JSON.

Uso: .venv/bin/python detectar.py videos/trafico_1min.mp4 [salidas/pistas.json]
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Ruta a los pesos; si no existe, ultralytics descarga yolov8m-seg.pt automáticamente
MODELO = os.environ.get("YOLO_MODELO", "yolov8m-seg.pt")
CLASES = {2: "coche", 3: "moto", 5: "bus", 7: "camion"}  # ids COCO


def pie(poli, banda=3.0):
    """Punto de apoyo en el suelo según la máscara: la zona más baja del contorno
    (rueda / esquina del parachoques). Es un punto físico fijo del vehículo, a
    diferencia del centro de la caja, que se desplaza al verse más o menos lateral."""
    if poli is None or len(poli) < 3:
        return None
    y_max = poli[:, 1].max()
    bajos = poli[poli[:, 1] >= y_max - banda]
    return [round(float(bajos[:, 0].mean()), 1), round(float(y_max), 1)]


def pie_franja(poli, fraccion=0.2):
    """Centro de la franja inferior de la máscara (ruedas + parachoques): menos
    sensible que el punto extremo a que el contorno salte de una rueda a otra."""
    if poli is None or len(poli) < 3:
        return None
    x0, y0 = np.floor(poli.min(axis=0)).astype(int)
    x1, y1 = np.ceil(poli.max(axis=0)).astype(int)
    m = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    cv2.fillPoly(m, [np.round(poli - [x0, y0]).astype(np.int32)], 1)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    corte = ys.max() - fraccion * (ys.max() - ys.min())
    sel = ys >= corte
    return [round(float(xs[sel].mean() + x0), 1), round(float(ys.max() + y0), 1)]


def main(video: str, salida: str) -> None:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    modelo = YOLO(MODELO)
    frames = []
    t0 = time.time()
    resultados = modelo.track(
        source=video, stream=True, persist=True, tracker="bytetrack.yaml",
        classes=list(CLASES), conf=0.25, imgsz=1280, device="mps",
        retina_masks=True, verbose=False,
    )
    for i, r in enumerate(resultados):
        dets = []
        b = r.boxes
        contornos = r.masks.xy if r.masks is not None else [None] * len(b)
        if b is not None and b.id is not None:
            for tid, c, cf, xyxy, poli in zip(b.id.int().tolist(), b.cls.int().tolist(),
                                              b.conf.tolist(), b.xyxy.tolist(), contornos):
                d = {"id": tid, "clase": CLASES[c], "conf": round(cf, 3),
                     "caja": [round(v, 1) for v in xyxy]}
                p, q = pie(poli), pie_franja(poli)
                if p is not None:
                    d["pie"] = p
                if q is not None:
                    d["pie20"] = q
                dets.append(d)
        frames.append(dets)
        if i % 250 == 0:
            print(f"  frame {i}/{total}  ({(i + 1) / (time.time() - t0):.1f} fps)", flush=True)

    Path(salida).parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w") as f:
        json.dump({"video": video, "fps": fps, "ancho": ancho, "alto": alto,
                   "frames": frames}, f)
    ids = {d["id"] for fr in frames for d in fr}
    print(f"Hecho: {len(frames)} frames, {len(ids)} vehículos distintos, "
          f"{time.time() - t0:.0f} s -> {salida}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "salidas/pistas.json")
