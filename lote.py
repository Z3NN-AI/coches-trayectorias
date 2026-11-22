"""Procesa una lista de vídeos por trozos: detectar -> calibrar (una vez por cámara)
-> trayectorias --estados. Reanudable: salta lo que ya existe.

Uso: .venv/bin/python lote.py [nombre ...]      (sin nombres: todos los de LISTA)
"""
import os
import subprocess
import sys
from pathlib import Path

PY = ".venv/bin/python"
os.environ.setdefault("YOLO_MODELO", os.path.expanduser("~/Desktop/AI/modelos/yolov8m-seg/yolov8m-seg.pt"))
TROZO = 300  # s

# normativa de marcas viales: ciclo línea+hueco (m), ancho de carril (m), mano de circulación
UK = dict(ciclo=9.0, carril=3.65, mano="izquierda")
FR = dict(ciclo=13.0, carril=3.5, mano="derecha")
NA = dict(ciclo=12.2, carril=3.66, mano="derecha")   # EE. UU./Canadá: 10 ft + 30 ft
LISTA = {  # en orden de prioridad
    "01_wqctLW0Hb_0": UK,
    "05_nt3D26lrkho": FR,
    "09_MNn9qKG2UFI": NA,
    "11_MZNbLWtJgnw": UK,
    "02_QuUxHIVUoaY": NA,
    "04_KBsqQez-O4w": NA,
    "03_TW3EH4cnFZo": NA,
}


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode == 0


def procesar(nombre, cfg):
    src = Path("videos/lista") / f"{nombre}.mp4"
    tdir = Path("videos/lista/trozos") / nombre
    out = Path("salidas/lista") / nombre
    tdir.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    if not any(tdir.glob("*.mp4")):
        run(["ffmpeg", "-v", "error", "-i", str(src), "-c", "copy", "-an", "-f", "segment",
             "-segment_time", str(TROZO), "-reset_timestamps", "1", str(tdir / "%03d.mp4")])
    (out / "config.txt").write_text(f"{cfg}\n")
    calib = out / "calibracion.json"
    for t in sorted(tdir.glob("*.mp4")):
        pistas = out / f"pistas_{t.stem}.json"
        estados = out / f"estados_{t.stem}.json"
        if not pistas.exists():
            if not run([PY, "detectar.py", str(t), str(pistas)]):
                print(f"!! fallo detectando {t}"); continue
        if not calib.exists():
            if not run([PY, "calibrar.py", str(pistas), str(calib),
                        "--ciclo", str(cfg["ciclo"]), "--carril", str(cfg["carril"])]):
                print(f"!! fallo calibrando {nombre}"); return
        if not estados.exists():
            run([PY, "trayectorias.py", str(pistas), "--calib", str(calib),
                 "--sin-video", "--estados", str(estados)])


if __name__ == "__main__":
    nombres = sys.argv[1:] or list(LISTA)
    for n in nombres:
        print(f"\n===== {n} =====", flush=True)
        procesar(n, LISTA[n])
