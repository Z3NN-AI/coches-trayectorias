"""Calibración automática del plano de la carretera para una cámara fija.

Modelo: un punto del suelo en la fila y está a Z = K / (y - y0) metros, y la escala
allí es ppm = f / Z píxeles por metro.

  y0 (horizonte)     -> el valor que hace que cada vehículo vaya a velocidad constante
  xv (punto de fuga) -> donde se cortan las rectas que recorren los vehículos
  K  (distancia)     -> ciclo de las líneas discontinuas detectadas (longitud por normativa)
  f  (lateral)       -> separación entre líneas discontinuas vecinas = ancho de carril

Si no encuentra líneas discontinuas, recurre al ancho de los coches (~1,8 m).

Uso: .venv/bin/python calibrar.py salidas/pistas.json [--ciclo 9 --carril 3.65]
     Reino Unido autovía: ciclo 9 m (2+7), carril 3.65 m
     España autovía:      ciclo 12.5 m (3.5+9), carril 3.5 m
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


# ------------------------------------------------------------------ utilidades
def fondo(video, n=60):
    """Mediana temporal de n fotogramas: la carretera sin coches."""
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fr = []
    for i in np.linspace(0, total - 2, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            fr.append(f)
    return np.median(np.array(fr), axis=0).astype(np.uint8)


def pistas_suelo(d, margen=5):
    """Puntos de contacto con el suelo (centro inferior de la caja) por vehículo."""
    W, H = d["ancho"], d["alto"]
    tr = defaultdict(list)
    for i, fr in enumerate(d["frames"]):
        for o in fr:
            x1, _, x2, y2 = o["caja"]
            if margen < x1 and x2 < W - margen and y2 < H - margen:
                tr[o["id"]].append((i, (x1 + x2) / 2, y2))
    return [np.array(v, float) for v in tr.values() if len(v) >= 40]


def horizonte_por_velocidad(pistas, fps, H):
    """y0 que mejor convierte cada pista en movimiento uniforme (Z lineal en t)."""
    def coste(y0):
        r = []
        for p in pistas:
            m = p[:, 2] > y0 + 25
            if m.sum() < 30:
                continue
            t, Z = p[m, 0] / fps, 1 / (p[m, 2] - y0)
            A = np.vstack([t, np.ones_like(t)]).T
            c, *_ = np.linalg.lstsq(A, Z, rcond=None)
            r.append(np.sqrt(np.mean((Z - A @ c) ** 2)) / np.ptp(Z))
        return np.median(r) if r else np.inf
    ys = np.arange(H * 0.05, H * 0.6, 1.0)
    costes = np.array([coste(y) for y in ys])
    return float(ys[int(np.argmin(costes))])


def punto_de_fuga(pistas, y0):
    xs = []
    for p in pistas:
        m = p[:, 2] > y0 + 40
        if m.sum() < 30 or np.ptp(p[m, 2]) < 80:
            continue
        a, b = np.polyfit(p[m, 2], p[m, 1], 1)
        xs.append(a * y0 + b)
    return float(np.median(xs)), float(np.subtract(*np.percentile(xs, [75, 25])))


def marcas(bg, y0, xv):
    """Trozos de pintura blanca alineados con el punto de fuga."""
    g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    m = (th > 40).astype(np.uint8)
    m[: int(y0) + 15] = 0
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if a < 6 or h < 3 or h > 0.5 * (y + h - y0):  # demasiado largo -> línea continua
            continue
        ys, xs = np.nonzero(lab[y:y + h, x:x + w] == k)
        if len(xs) < 6:
            continue
        # orientación del trozo vs dirección hacia el punto de fuga
        c = np.cov(np.vstack([xs, ys]))
        ev, evec = np.linalg.eigh(c)
        eje = evec[:, 1]
        if ev[1] < 4 * max(ev[0], 1e-6):  # poco alargado
            continue
        cx, cy = cen[k]
        rayo = np.array([cx - xv, cy - y0]); rayo /= np.linalg.norm(rayo)
        if abs(eje @ rayo) < np.cos(np.radians(12)):
            continue
        out.append({"tan": (cx - xv) / (cy - y0), "y_lejos": y, "y_cerca": y + h,
                    "caja": (x, y, w, h)})
    return out


def agrupar_lineas(ms, salto=0.04, minimo=4):
    ms = sorted(ms, key=lambda m: m["tan"])
    grupos, actual = [], [ms[0]] if ms else []
    for m in ms[1:]:
        if m["tan"] - actual[-1]["tan"] > salto:
            grupos.append(actual); actual = [m]
        else:
            actual.append(m)
    if actual:
        grupos.append(actual)
    return [g for g in grupos if len(g) >= minimo]


def paso_base(u):
    """Separación base de una progresión con huecos (marcas no detectadas)."""
    d = np.diff(np.sort(u))
    d = d[d > 1e-9]
    mejor, err = None, np.inf
    for b in np.unique(np.concatenate([d, d / 2, d / 3])):
        r = d / b
        if np.any(r < 0.7):
            continue
        e = np.mean((r - np.round(r)) ** 2)
        if e < err - 1e-4:
            mejor, err = b, e
    return mejor, err


# ------------------------------------------------------------------ principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pistas")
    ap.add_argument("salida", nargs="?", default="salidas/calibracion.json")
    ap.add_argument("--ciclo", type=float, default=9.0, help="línea + hueco (m)")
    ap.add_argument("--carril", type=float, default=3.65, help="ancho de carril (m)")
    a = ap.parse_args()

    d = json.load(open(a.pistas))
    fps, W, H = d["fps"], d["ancho"], d["alto"]
    pistas = pistas_suelo(d)
    y0 = horizonte_por_velocidad(pistas, fps, H)
    xv, iqr = punto_de_fuga(pistas, y0)
    print(f"horizonte y0={y0:.0f}px (por velocidad constante), "
          f"punto de fuga x={xv:.0f}px (IQR {iqr:.0f}px)")

    bg = fondo(d["video"])
    vis = bg.copy()
    lineas = agrupar_lineas(marcas(bg, y0, xv))
    Ks, tans = [], []
    for g in lineas:
        u = np.array([1 / (m["y_lejos"] - y0) for m in g])
        b, err = paso_base(u)
        if b is None or err > 0.02:
            continue
        Ks.append(a.ciclo / b)
        tans.append(np.median([m["tan"] for m in g]))
        col = (0, 255, 0)
        for m in g:
            x, y, w, h = m["caja"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), col, 1)
        yb = H
        cv2.line(vis, (int(xv), int(y0)), (int(xv + tans[-1] * (yb - y0)), yb), col, 1)

    metodo = "líneas"
    if len(Ks) >= 3:  # descartar líneas falsas que no concuerdan con el resto
        Ks, tans = np.array(Ks), np.array(tans)
        ok = np.abs(Ks / np.median(Ks) - 1) < 0.15
        if (~ok).any():
            print(f"descartadas {int((~ok).sum())} líneas incoherentes: K={np.round(Ks[~ok])}")
        Ks, tans = list(Ks[ok]), list(tans[ok])
    if len(Ks) >= 1:
        K = float(np.median(Ks))
        tans = np.sort(tans)
        izq, der = tans[tans < 0], tans[tans > 0]
        vecinas = [np.diff(lado) for lado in (izq, der) if len(lado) >= 2]
        if vecinas:
            dtan = float(np.median(np.concatenate(vecinas)))
            f = K * dtan / a.carril
        else:
            f = W / (2 * np.tan(np.radians(30)))
            metodo = "líneas (f supuesta: no hay dos líneas vecinas)"
        print(f"{len(Ks)} líneas discontinuas: K por línea = {np.round(Ks, 0)}")
    else:
        metodo = "ancho de coches (sin líneas discontinuas)"
        f = W / (2 * np.tan(np.radians(30)))
        ppm = []
        for fr in d["frames"]:
            for o in fr:
                x1, _, x2, y2 = o["caja"]
                if o["clase"] == "coche" and o["conf"] > 0.5 and y2 > y0 + 20:
                    ppm.append((x2 - x1) / 1.8 / (y2 - y0))
        K = f / float(np.median(ppm))  # ppm = f (y-y0) / K

    fov = float(np.degrees(2 * np.arctan(W / 2 / f)))
    print(f"Método: {metodo}\nK={K:.0f}  f={f:.0f}px (FOV {fov:.0f}°)  "
          f"altura cámara≈{K / f:.1f} m")
    cv2.line(vis, (0, int(y0)), (W, int(y0)), (255, 255, 255), 1)
    cv2.circle(vis, (int(xv), int(y0)), 5, (0, 0, 255), -1)
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(Path(a.salida).with_suffix(".jpg")), vis)
    json.dump({"y0": y0, "xv": xv, "K": K, "f": f, "metodo": metodo,
               "ciclo": a.ciclo, "carril": a.carril}, open(a.salida, "w"), indent=1)


if __name__ == "__main__":
    main()
