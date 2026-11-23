"""Vídeo de la autovía sintética: SOLO conductores promedio (agentes estocásticos).

Dos escenarios seguidos en el mismo vídeo:
  1. muy denso + 3 camiones en el carril 1 y un 4.º en el carril 2 adelantándolos
     despacio (la "carrera de elefantes")
  2. ligero: pocos coches y pocos camiones

Vista cenital de una ventana de 600 m que acompaña al grupo, partida en 3 tramos de
200 m (se lee como un texto: de izquierda a derecha y bajando). Antes de grabar se
simula un calentamiento para que la carretera ya esté llena y nadie aparezca de la
nada; los vehículos entran 1 km antes de la ventana. El ángulo de cada vehículo sale
de su propio movimiento (lateral / avance), así que no giran sobre sí mismos.

Uso: .venv/bin/python video_sintetico.py [--camara 01_wqctLW0Hb_0] [--duracion 120]
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import joblib
import numpy as np

from simulador import DT, Poblacion, Trafico

ANCHO, ALTO, FPS = 1280, 720, 20
VENTANA, TRAMOS = 600.0, 3            # m visibles, filas
PPM = ANCHO / (VENTANA / TRAMOS)      # píxeles por metro (6.4)
CARRIL_M = 3.65
LAT = 2.2                             # los carriles se dibujan 2.2 veces más anchos (se ve mejor)
LARGO_VIA = 6000.0
ESCENARIOS = {
    "dense": dict(titulo="VERY DENSE TRAFFIC + 3 trucks in lane 1, 1 truck overtaking in lane 2",
                  q=1700, cuota=0.08, elefantes=True, calentar=90),
    "light": dict(titulo="LIGHT TRAFFIC: few cars, few trucks",
                  q=350, cuota=0.06, elefantes=False, calentar=120),
}


def color_vel(kmh):
    t = np.uint8([[[int(np.clip(kmh / 130 * 255, 0, 255))]]])
    return tuple(int(c) for c in cv2.applyColorMap(t, cv2.COLORMAP_TURBO)[0, 0])


class Escenario:
    def __init__(self, info, modelo, cfg, semilla):
        self.rng = np.random.default_rng(semilla)
        self.pob = Poblacion(info, "acercan", modelo, self.rng, idm_de="acercan", perfil_de="acercan")
        self.sim = Trafico(self.pob, LARGO_VIA)
        self.cfg = cfg
        self.nc = self.pob.nc
        self.uso = {g: (self.pob.perfil[g] or self.pob.perfil[0])["uso_carriles"] for g in (0, 1)}
        self.tasa = cfg["q"] * self.nc / 3600
        self.sig = self.rng.exponential(1 / self.tasa)
        self.cola, self.n_id = [], 0
        self.elefantes = []
        self.lat_prev = {}

    def llegadas(self):
        s, rng = self.sim, self.rng
        while s.t >= self.sig:
            g = int(rng.uniform() < self.cfg["cuota"])
            pr = np.array([self.uso[g].get(str(c), self.uso[g].get(c, 0)) for c in range(1, self.nc + 1)], float)
            c = int(rng.choice(np.arange(1, self.nc + 1), p=pr / pr.sum()))
            self.cola.append((g, c)); self.sig += rng.exponential(1 / self.tasa)
        resto = []
        for g, c in self.cola:
            ult = [m for m in s.veh if m["carril"] == c and m["x"] < 60]
            hueco = min([m["x"] - m["L"] for m in ult], default=1e3)
            if hueco > 8:
                v_in = min([m["v"] for m in ult], default=None)
                s.entra(self.n_id, "camion" if g else "coche", 0.0, c, 0.0)
                me = s.veh[-1]
                me["v"] = min(me["p"]["v0"], v_in if v_in is not None else me["p"]["v0"])
                self.n_id += 1
            else:
                resto.append((g, c))
        self.cola = resto

    def poner_elefantes(self, x0):
        """3 camiones en el carril 1 a 85 km/h y uno en el 2 a 88 km/h, junto a ellos.
        Se quita a quien estorbe (antes de grabar: nadie lo ve)."""
        s = self.sim
        plan = [(1, x0 + 60, 85), (1, x0 + 30, 85), (1, x0, 85), (2, x0 + 15, 88)]
        for c, x, kmh in plan:
            s.veh = [m for m in s.veh if not (m["carril"] == c and abs(m["x"] - x) < 25)]
        for c, x, kmh in plan:
            s.entra(f"T{len(self.elefantes)}", "camion", x, c, kmh / 3.6, v0=kmh / 3.6)
            s.veh[-1]["p"]["v0"] = kmh / 3.6
            self.elefantes.append(s.veh[-1]["id"])

    def paso(self):
        self.llegadas()
        self.sim.paso()


def dibujar(esc, x_ini, t, cab, n_cambios0, w=1.0):
    """w: fase entre dos pasos de simulación (0..1) para dibujar posiciones intermedias."""
    img = np.full((ALTO, ANCHO, 3), (40, 90, 40), np.uint8)
    nc = esc.nc
    alto_via = int(nc * CARRIL_M * PPM * LAT)
    y_top = 110
    sep = (ALTO - y_top - 40 - TRAMOS * alto_via) // (TRAMOS - 1)
    largo_tramo = VENTANA / TRAMOS
    # mano izquierda (UK): carril 1 abajo de cada tramo
    def a_px(x, lat):
        k = int((x - x_ini) // largo_tramo)
        if k < 0 or k >= TRAMOS:
            return None
        px = (x - x_ini - k * largo_tramo) * PPM
        y0 = y_top + k * (alto_via + sep)
        py = y0 + alto_via - (lat - 0.5) * CARRIL_M * PPM * LAT
        return px, py
    for k in range(TRAMOS):
        y0 = y_top + k * (alto_via + sep)
        cv2.rectangle(img, (0, y0), (ANCHO, y0 + alto_via), (70, 70, 70), -1)
        cv2.line(img, (0, y0), (ANCHO, y0), (230, 230, 230), 2)
        cv2.line(img, (0, y0 + alto_via), (ANCHO, y0 + alto_via), (230, 230, 230), 2)
        for c in range(1, nc):
            yy = int(y0 + c * CARRIL_M * PPM * LAT)
            xm = x_ini + k * largo_tramo
            for xd in np.arange(np.ceil(xm / 9) * 9, xm + largo_tramo, 9.0):  # UK: 2 m + 7 m
                p0 = int((xd - xm) * PPM)
                cv2.line(img, (p0, yy), (p0 + int(2 * PPM), yy), (220, 220, 220), 1)
        m0 = x_ini + k * largo_tramo
        cv2.putText(img, f"{m0:.0f} m", (4, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1,
                    cv2.LINE_AA)
        if k < TRAMOS - 1:
            cv2.putText(img, "continues below ->", (ANCHO - 170, y0 + alto_via + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    visibles = []
    for m in esc.sim.veh:
        lp_ = esc.lat_prev.get(m["id"], m["lat"])
        p = a_px(m["x"] - m["v"] * DT * (1 - w), lp_ + (m["lat"] - lp_) * w)
        if p is None:
            continue
        visibles.append(m)
        # ángulo del propio movimiento: lateral por segundo frente a avance por segundo
        lp = esc.lat_prev.get(m["id"], m["lat"])
        dlat = (m["lat"] - lp) / DT * CARRIL_M
        ang = np.arctan2(dlat, max(m["v"], 1.0))
        L, W = m["L"], (2.5 if m["g"] else 1.8)
        c, s_ = np.cos(ang), np.sin(ang)
        pts = []
        for dx, dy in ((0, -W / 2), (0, W / 2), (-L, W / 2), (-L, -W / 2)):
            X = dx * c - dy * s_; Y = dx * s_ + dy * c
            pts.append((p[0] + X * PPM, p[1] - Y * PPM * LAT))
        pts = np.int32(pts)
        col = color_vel(m["v"] * 3.6)
        cv2.fillPoly(img, [pts], col, cv2.LINE_AA)
        borde = (255, 255, 255) if m["id"] in esc.elefantes else (20, 20, 20)
        cv2.polylines(img, [pts], True, borde, 2 if m["id"] in esc.elefantes else 1, cv2.LINE_AA)
        if m["prog"] < 1.0:   # cambiando de carril: flecha
            d = np.sign(m["carril"] - m["lat0"])
            cx, cy = pts.mean(0)
            cv2.arrowedLine(img, (int(cx), int(cy)), (int(cx), int(cy - d * 18)), (255, 255, 255), 2,
                            tipLength=0.5)
        txt = f"{m['v'] * 3.6:.0f}" + (" TRUCK" if m["g"] else "")
        cv2.putText(img, txt, (int(pts[:, 0].min()), int(pts[:, 1].min()) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
    # cabecera
    cv2.rectangle(img, (0, 0), (ANCHO, 80), (25, 25, 25), -1)
    v = [m["v"] * 3.6 for m in visibles]
    vm = len(v) / np.sum(1 / np.maximum(v, 1)) if v else float("nan")
    dens = len(visibles) / (VENTANA / 1000) / nc
    camiones = sum(m["g"] for m in visibles)
    cambios = len(esc.sim.cambios) - n_cambios0
    lineas = [(cab, 0.6, 2),
              (f"ONLY AVERAGE DRIVERS (learned from real video)  |  t = {t:5.1f} s  |  "
               f"view moves with the group, {VENTANA:.0f} m shown in {TRAMOS} rows", 0.45, 1),
              (f"in view: {len(visibles)} vehicles ({camiones} trucks)  |  {dens:.0f} veh/km/lane  |  "
               f"average {vm:.0f} km/h  |  lane changes so far: {cambios}", 0.45, 1)]
    for j, (s, e, g) in enumerate(lineas):
        cv2.putText(img, s, (10, 24 + 24 * j), cv2.FONT_HERSHEY_SIMPLEX, e, (255, 255, 255), g, cv2.LINE_AA)
    # leyenda de color
    for i in range(200):
        cv2.line(img, (ANCHO - 230 + i, ALTO - 28), (ANCHO - 230 + i, ALTO - 16), color_vel(i * 130 / 200), 1)
    cv2.putText(img, "0", (ANCHO - 245, ALTO - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(img, "130 km/h", (ANCHO - 26 - 50, ALTO - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    nota = ("white outline = the 4 trucks of the scenario   " if esc.elefantes else "") + \
        "arrow = changing lane   (lanes drawn 2x wide)"
    cv2.putText(img, nota,
                (10, ALTO - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camara", default="01_wqctLW0Hb_0")
    ap.add_argument("--comportamiento", default="salidas/lista/comportamiento.json")
    ap.add_argument("--duracion", type=float, default=120.0)
    ap.add_argument("--semilla", type=int, default=1)
    ap.add_argument("--salida", default="salidas/lista/simulacion_conductores_promedio.mp4")
    a = ap.parse_args()
    salida = Path(a.salida); base, k = salida, 2
    while salida.exists():   # nunca pisar un vídeo anterior
        salida = base.with_name(f"{base.stem}_v{k}{base.suffix}"); k += 1
    info = json.load(open(a.comportamiento))[a.camara]
    modelo = joblib.load(Path(a.comportamiento).with_suffix(".decision.joblib"))
    tmp = salida.with_suffix(".tmp.mp4")
    out = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (ANCHO, ALTO))
    cada = int(round(FPS * DT))          # fotogramas por paso de simulación (tiempo real)
    for nombre, cfg in ESCENARIOS.items():
        esc = Escenario(info, modelo, cfg, a.semilla)
        # calentamiento: la carretera se llena antes de grabar
        while esc.sim.t < cfg["calentar"]:
            esc.paso()
        x_ini = 1000.0
        if cfg["elefantes"]:
            esc.poner_elefantes(x_ini + 150)
            for _ in range(20):
                esc.paso()
        n_c0 = len(esc.sim.cambios)
        # título del escenario
        for _ in range(FPS * 2):
            img = np.full((ALTO, ANCHO, 3), 25, np.uint8)
            cv2.putText(img, cfg["titulo"], (40, ALTO // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                        2, cv2.LINE_AA)
            out.write(img)
        t0 = esc.sim.t
        v_ventana = 24.0 if cfg["elefantes"] else 100 / 3.6
        nfr = int(a.duracion * FPS)
        for f in range(nfr):
            if f % cada == 0:
                esc.lat_prev = {m["id"]: m["lat"] for m in esc.sim.veh}
                esc.paso()
            w = (f % cada + 1) / cada
            # la ventana sigue a los camiones (o a la velocidad media) sin saltos
            el = [m for m in esc.sim.veh if m["id"] in esc.elefantes]
            objetivo = (np.mean([m["x"] for m in el]) - VENTANA * 0.35) if el else None
            vis = [m["v"] for m in esc.sim.veh if x_ini <= m["x"] <= x_ini + VENTANA]
            v_obj = 100 / 3.6 if not esc.elefantes else (np.median(vis) if vis else v_ventana)
            if objetivo is not None:
                v_obj = np.median([m["v"] for m in el]) + 0.2 * (objetivo - x_ini)
            v_ventana += np.clip(v_obj - v_ventana, -0.3, 0.3)
            x_ini += max(v_ventana, 0) / FPS
            if x_ini + VENTANA > LARGO_VIA - 50:
                break
            out.write(dibujar(esc, x_ini, esc.sim.t - t0,
                              f"SCENARIO {list(ESCENARIOS).index(nombre) + 1}: {cfg['titulo']}", n_c0, w))
        print(f"{nombre}: {len(esc.sim.cambios) - n_c0} cambios de carril, {f + 1} fotogramas")
    out.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-crf", "22",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(salida)], check=True)
    tmp.unlink()
    print(f"Vídeo -> {salida}")


if __name__ == "__main__":
    main()
