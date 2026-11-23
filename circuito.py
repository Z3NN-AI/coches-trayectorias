"""El experimento de Sugiyama (2008) en el ordenador: N coches en un circuito circular
de un carril, todos intentando ir a la misma velocidad. ¿Sale un atasco fantasma?

Dos poblaciones con EXACTAMENTE el mismo pedal (IDM calibrado en vídeo real):
  - PERFECTOS: todos iguales (distancia mediana real), sin ruido.
  - PROMEDIO: cada uno con su distancia (distribución real), su "humor" de velocidad
    que oscila y la imprecisión del pie (los mismos que en simulador.py).
Las dos empiezan igual: repartidos uniformemente y con UN coche 1 m fuera de sitio.

Uso: .venv/bin/python circuito.py [--n 22] [--largo 230] [--duracion 300] [--barrido]
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from simulador import RUIDO_PEDAL, SIGMA_V0, TAU_V0

DT = 0.1
TAU_ROBOT = 30.0   # s, ventana de la velocidad media que sigue el robot


def params(ruta, camara):
    info = json.load(open(ruta))[camara]
    # el seguimiento se ve donde la gente va pegada: el sentido denso
    p = dict(info["idm"]["alejan/ligero"])
    T = info["perfil"]["alejan/ligero"]["T_estable"]
    return p, T


def muestra(rng, d):
    qs = np.array([5, 25, 50, 75, 95]) / 100
    vals = np.array([d[f"p{q}"] for q in (5, 25, 50, 75, 95)])
    return float(np.interp(rng.uniform(0.02, 0.98), qs, vals))


class Circuito:
    def __init__(self, n, largo, p, T_dist, humanos, v_obj, rng):
        self.n, self.C, self.rng, self.humanos = n, largo, rng, humanos
        self.x = np.arange(n) * largo / n
        self.x[0] += 1.0                         # la única perturbación inicial
        self.v = np.zeros(n)
        self.p = {k: np.full(n, float(v)) for k, v in p.items()}
        self.v0_base = np.full(n, v_obj)
        if humanos:
            self.p["T"] = np.array([muestra(rng, T_dist) for _ in range(n)])
            self.osc = rng.uniform(0, 2, n) * SIGMA_V0
            self.eps = rng.normal(0, 1, n) * self.osc
        else:
            self.p["T"] = np.full(n, T_dist["p50"])
            self.osc = np.zeros(n); self.eps = np.zeros(n)
        self.ruido = np.full(n, 1.0 if humanos else 0.0)
        self.robot = np.zeros(n, bool)
        self.v_med = np.zeros(n)        # media móvil de la propia velocidad (para el robot)
        self.t = 0.0

    def meter_robot(self, i, a_):
        """Convierte el coche i en un conductor atento sin ruido (como el coche autónomo
        de Stern et al. 2018): misma distancia mediana, reacción viva."""
        self.p["a"][i] = self.p["b"][i] = a_
        self.osc[i] = self.eps[i] = 0.0
        self.ruido[i] = 0.0
        self.robot[i] = True

    def paso(self):
        orden = np.argsort(self.x)
        lider = np.empty(self.n, int)
        lider[orden] = np.roll(orden, -1)
        s = (self.x[lider] - self.x) % self.C - 4.5
        p = self.p
        if self.humanos:
            k = DT / TAU_V0
            self.eps += -self.eps * k + self.osc * np.sqrt(2 * k) * self.rng.normal(size=self.n)
        v0 = self.v0_base * (1 + self.eps)
        s = np.maximum(s, 0.1)
        s_star = p["s0"] + np.maximum(0, self.v * p["T"] + self.v * (self.v - self.v[lider])
                                      / (2 * np.sqrt(p["a"] * p["b"])))
        a = p["a"] * (1 - (self.v / v0) ** 4 - (s_star / s) ** 2)
        if self.humanos:
            a = a + RUIDO_PEDAL * self.ruido * self.rng.normal(size=self.n)
        if self.robot.any():
            # AMORTIGUADOR (Stern 2018): ir a la velocidad media reciente, sin copiar los
            # acelerones del de delante; solo frena si el hueco se vuelve corto de verdad
            r = self.robot
            # velocidad media de todo el circuito (la saca de su tiempo por vuelta)
            self.v_med[r] += (self.v.mean() - self.v_med[r]) * DT / TAU_ROBOT
            objetivo = np.maximum(self.v_med[r], 2.0)
            a_suave = 0.3 * (objetivo - self.v[r])
            seguridad = p["a"][r] * (1 - ((p["s0"][r] + self.v[r] * 0.6) / s[r]) ** 2)
            a[r] = np.minimum(a_suave, np.maximum(seguridad, -9))
        a = np.clip(a, -9, 4)
        v2 = np.maximum(self.v + a * DT, 0)
        self.x = (self.x + (self.v + v2) / 2 * DT) % self.C
        self.v = v2
        self.t += DT


def medir(c, dur):
    vs = []
    while c.t < dur:
        c.paso()
        if c.t > dur / 2:
            vs.append(c.v.copy())
    vs = np.array(vs) * 3.6
    return dict(v_media=vs.mean(), desv=vs.std(axis=1).mean(), parados=(vs < 5).mean() * 100,
                v_min=vs.min())


def dibujar_anillo(img, cx, cy, R, c, titulo, hist):
    cv2.circle(img, (cx, cy), R + 14, (70, 70, 70), 28)
    cv2.circle(img, (cx, cy), R + 28, (230, 230, 230), 1)
    cv2.circle(img, (cx, cy), R, (230, 230, 230), 1)
    for i in range(c.n):
        ang = c.x[i] / c.C * 2 * np.pi
        kmh = c.v[i] * 3.6
        t = np.uint8([[[int(np.clip(kmh / 40 * 255, 0, 255))]]])
        col = tuple(int(q) for q in cv2.applyColorMap(t, cv2.COLORMAP_TURBO)[0, 0])
        L = 4.5 / c.C * 2 * np.pi
        pts = []
        for a_, r_ in ((ang, R + 6), (ang, R + 22), (ang - L, R + 22), (ang - L, R + 6)):
            pts.append((cx + r_ * np.cos(a_), cy - r_ * np.sin(a_)))
        cv2.fillPoly(img, [np.int32(pts)], col, cv2.LINE_AA)
        if c.robot[i]:
            cv2.polylines(img, [np.int32(pts)], True, (255, 255, 255), 3, cv2.LINE_AA)
    vm = c.v.mean() * 3.6
    parados = int((c.v * 3.6 < 5).sum())
    cv2.putText(img, titulo, (cx - 190, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    for j, s in enumerate([f"average {vm:4.1f} km/h", f"slowest {c.v.min() * 3.6:4.1f} km/h",
                           f"stopped: {parados} of {c.n}"]):
        cv2.putText(img, s, (cx - 85, cy - 20 + 26 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)
    # traza de velocidad media y mínima
    h = np.array(hist[-600:]) if hist else np.zeros((1, 2))
    x0, y0, w, hh = cx - 200, 650, 400, 90
    cv2.rectangle(img, (x0, y0 - hh), (x0 + w, y0), (40, 40, 40), -1)
    for k, col in ((0, (255, 255, 255)), (1, (60, 60, 255))):
        pts = [(x0 + int(i * w / 600), int(y0 - np.clip(v / 40, 0, 1) * hh)) for i, v in enumerate(h[:, k])]
        if len(pts) > 1:
            cv2.polylines(img, [np.int32(pts)], False, col, 1, cv2.LINE_AA)
    cv2.putText(img, "last 60 s: average (white) / slowest car (red), 0-40 km/h", (x0, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=22)
    ap.add_argument("--largo", type=float, default=230.0)
    ap.add_argument("--v-obj", type=float, default=30.0, help="km/h que se pide a todos")
    ap.add_argument("--duracion", type=float, default=300.0)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--comportamiento", default="salidas/lista/comportamiento.json")
    ap.add_argument("--camara", default="01_wqctLW0Hb_0")
    ap.add_argument("--perfecto-a", type=float, default=2.0,
                    help="aceleración/frenada del conductor perfecto (reacción viva, tipo ACC)")
    ap.add_argument("--robot", action="store_true",
                    help="izquierda: solo humanos; derecha: los mismos humanos + 1 robot amortiguador")
    ap.add_argument("--barrido", action="store_true", help="solo tabla: varias densidades")
    ap.add_argument("--salida", default="salidas/lista/circuito_atasco_fantasma.mp4")
    a = ap.parse_args()
    p, T = params(a.comportamiento, a.camara)
    print(f"Pedal (IDM calibrado): a={p['a']:.2f} b={p['b']:.2f} s0={p['s0']:.2f}  T mediana real {T['p50']:.2f} s")

    if a.barrido:
        print(f"{'coches':>6} {'veh/km':>6} | {'PERFECTOS media  desv  %parados':>32} | {'PROMEDIO media  desv  %parados':>31}")
        for n in (10, 14, 18, 22, 26, 30, 34):
            r = []
            for hum in (False, True):
                c = Circuito(n, a.largo, p, T, hum, a.v_obj / 3.6, np.random.default_rng(a.semilla))
                r.append(medir(c, a.duracion))
            print(f"{n:>6} {n / a.largo * 1000:>6.0f} | " + " | ".join(
                f"{x['v_media']:>13.1f} {x['desv']:>5.1f} {x['parados']:>9.1f}" for x in r))
        return

    salida = Path(a.salida); base, k = salida, 2
    while salida.exists():
        salida = base.with_name(f"{base.stem}_v{k}{base.suffix}"); k += 1
    W, H, FPS = 1280, 720, 20
    if a.robot:
        salida = salida.with_name(salida.stem.replace("circuito_atasco_fantasma", "circuito_robot") + salida.suffix)
        base, k = salida, 2
        while salida.exists():
            salida = base.with_name(f"{base.stem}_v{k}{base.suffix}"); k += 1
        perf = Circuito(a.n, a.largo, p, T, True, a.v_obj / 3.6, np.random.default_rng(a.semilla))
        hum = Circuito(a.n, a.largo, p, T, True, a.v_obj / 3.6, np.random.default_rng(a.semilla))
        hum.meter_robot(0, 2.0)
        tit = ("ONLY AVERAGE HUMANS", "SAME HUMANS + 1 ROBOT CAR (white)")
    else:
        p_perf = dict(p, a=a.perfecto_a, b=a.perfecto_a)
        perf = Circuito(a.n, a.largo, p_perf, T, False, a.v_obj / 3.6, np.random.default_rng(a.semilla))
        hum = Circuito(a.n, a.largo, p, T, True, a.v_obj / 3.6, np.random.default_rng(a.semilla))
        tit = (f"PERFECT DRIVERS (alert: a=b={a.perfecto_a:.1f}, no noise)",
               f"AVERAGE HUMANS (learned: a={p['a']:.2f}, b={p['b']:.2f})")
    tmp = salida.with_suffix(".tmp.mp4")
    out = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    hp, hh = [], []
    acel = 4   # el vídeo va 4x más rápido que la realidad (si no, 5 minutos de nada)
    pasos = int(round(acel / (FPS * DT)))
    for f in range(int(a.duracion * FPS / acel)):
        for _ in range(pasos):
            perf.paso(); hum.paso()
            hp.append((perf.v.mean() * 3.6, perf.v.min() * 3.6))
            hh.append((hum.v.mean() * 3.6, hum.v.min() * 3.6))
        img = np.full((H, W, 3), (40, 80, 40), np.uint8)
        dibujar_anillo(img, 320, 330, 190, perf, tit[0], hp)
        dibujar_anillo(img, 960, 330, 190, hum, tit[1], hh)
        cv2.putText(img, f"{a.n} cars on a {a.largo:.0f} m ring, all asked to drive at {a.v_obj:.0f} km/h "
                         f"(Sugiyama 2008)  |  t = {perf.t:5.0f} s  (video x{acel})",
                    (20, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        out.write(img)
    out.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-crf", "22",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(salida)], check=True)
    tmp.unlink()
    print(f"Vídeo -> {salida}")


if __name__ == "__main__":
    main()
