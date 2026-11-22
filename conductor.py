"""El conductor promedio en directo: aplica en cada fotograma lo aprendido por
comportamiento.py (IDM calibrado, carriles, modelo de cambio de carril) y calcula
la fluidez por sentido y el riesgo de colisión.

Lo usa trayectorias.py con --comportamiento salidas/lista/comportamiento.json --camara NOMBRE.
"""
import json
from collections import deque
from pathlib import Path

import numpy as np

from comportamiento import LARGO, MANUAL, idm_acc, rasgos

# Fluidez: densidad (veh/km/carril). Por debajo de ~18 cada uno va a su ritmo;
# por encima de ~28 (densidad crítica de Greenshields en estos datos) se forma cola.
K_DENSO, K_CONGESTION = 18.0, 28.0
TTC_ALTO, TTC_MEDIO = 2.5, 4.0       # s, tiempo hasta colisión con el de delante
HW_ALTO = 0.5                        # s, distancia temporal peligrosa
P_CAMBIO = 0.35                      # prob. a partir de la que se marca un cambio


class Conductor:
    def __init__(self, ruta_json, camara, fps, rumbo, ruta_modelo=None):
        info = json.load(open(ruta_json))[camara]
        self.fps, self.rumbo = fps, rumbo
        self.u = np.array([np.cos(rumbo), np.sin(rumbo)])
        self.n = np.array([-self.u[1], self.u[0]])
        self.mano = info["config"]["mano"]
        self.centros = {k: np.array(v) for k, v in info["centros"].items()}
        self.idm = {s: info["idm"].get(s, MANUAL) for s in ("alejan", "acercan")}
        self.err = info.get("error_idm", {})
        ruta_modelo = ruta_modelo or Path(ruta_json).with_suffix(".volante.joblib")
        self.modelo = None
        if Path(ruta_modelo).exists():
            import joblib
            m = joblib.load(ruta_modelo)
            self.modelo, self.columnas = m["modelo"], m["columnas"]
        self.carril = {}                 # id -> carril actual (con histéresis)
        self.pendiente = {}              # id -> (carril candidato, fotogramas seguidos)
        self.hist = {s: deque(maxlen=int(60 * fps)) for s in ("alejan", "acercan")}
        self.pasados = {s: {} for s in ("alejan", "acercan")}   # id -> clase
        self.vel_veh = {s: {} for s in ("alejan", "acercan")}   # id -> [suma, n]

    # ------------------------------------------------------------ geometría
    def sentido(self, th):
        return "alejan" if np.cos(th - self.rumbo) > 0 else "acercan"

    def coords(self, v):
        X, Z, vel, th = v.s[:4]
        sen = self.sentido(th)
        sg = 1 if sen == "alejan" else -1
        P = np.array([X, Z])
        return sen, sg * float(P @ self.u), sg * float(P @ self.n), vel * np.sin(th - self.rumbo) * sg

    def asignar(self, vid, sen, izq, histeresis=0.6, persistencia=0.3):
        c = self.centros.get(sen)
        if c is None:
            return 0
        cand = int(np.argmin(np.abs(c - izq)))
        cur = self.carril.get(vid)
        if cur is None:
            self.carril[vid] = cand
            return cand + 1
        if cand != cur and abs(izq - c[cur]) > abs(izq - c[cand]) + histeresis:
            pc, k = self.pendiente.get(vid, (cand, 0))
            k = k + 1 if pc == cand else 1
            self.pendiente[vid] = (cand, k)
            if k >= persistencia * self.fps:
                self.carril[vid] = cand
                self.pendiente.pop(vid, None)
        else:
            self.pendiente.pop(vid, None)
        return self.carril[vid] + 1

    # ------------------------------------------------------------ cada fotograma
    def paso(self, vehs):
        """vehs: vehículos activos (con .id, .s, .clase). Devuelve (por_vehiculo, fluidez)."""
        por_sen = {"alejan": [], "acercan": []}
        res = {}
        for v in vehs:
            sen, x, izq, vl = self.coords(v)
            c = self.asignar(v.id, sen, izq)
            por_sen[sen].append((v, x, c, vl))
            if v.perdido == 0:
                self.pasados[sen][v.id] = v.clase
                acu = self.vel_veh[sen].setdefault(v.id, [0.0, 0])
                acu[0] += v.s[2]; acu[1] += 1
            res[v.id] = dict(sentido=sen, carril=c, p_izq=0.0, p_der=0.0, ttc=None,
                             hw=None, lider=None, corte=None)
        fluidez = {}
        for sen, lst in por_sen.items():
            if not lst:
                self.hist[sen].append((0, np.nan))
                fluidez[sen] = self.fluidez(sen)
                continue
            ids = np.array([v.id for v, *_ in lst])
            X = np.array([x for _, x, _, _ in lst])
            V = np.array([v.s[2] for v, *_ in lst])
            C = np.array([c for _, _, c, _ in lst])
            L = np.array([LARGO.get(v.clase, 4.5) for v, *_ in lst])
            VL = np.array([vl for *_, vl in lst])
            p = self.idm[sen]
            # riesgo longitudinal: líder en el mismo carril
            for i in range(len(lst)):
                m = (C == C[i]) & (X > X[i])
                if not m.any() or C[i] < 1:
                    continue
                j = np.where(m)[0][np.argmin(X[m])]
                hueco = X[j] - X[i] - L[j]
                r = res[ids[i]]
                r["lider"] = ids[j]
                r["hw"] = hueco / max(V[i], 0.5)
                cierre = V[i] - V[j]
                r["ttc"] = hueco / cierre if cierre > 0.3 and hueco > 0 else None
            # volante
            if self.modelo is not None and sen in self.centros:
                F = rasgos(ids, X, V, C, L, VL, self.centros[sen], p)
                if F:
                    P = self.modelo.predict_proba(
                        np.array([[f[k] for k in self.columnas] for f in F]))[:, 1]
                    for f, pr in zip(F, P):
                        r = res[ids[f["i"]]]
                        lado = "p_izq" if f["a_la_izquierda"] else "p_der"
                        r[lado] = max(r[lado], float(pr))
                        # riesgo de corte: quiere cambiar y el de detrás en ese carril
                        # viene cerca o más rápido
                        if pr > P_CAMBIO and (f["hueco_det"] < max(1.0 * V[f["i"]], 8)
                                              or (f["dv_det"] > 2 and f["hueco_det"] < 30)):
                            r["corte"] = "izq" if f["a_la_izquierda"] else "der"
            # fluidez (densidad en los 20-80 m observados)
            enrango = [(v, x) for v, x, c, _ in lst if 20 <= v.s[1] <= 80]
            nc = len(self.centros.get(sen, [1]))
            k = len(enrango) / 0.060 / max(nc, 1)
            vel = 3.6 / np.mean([1 / max(v.s[2], 0.5) for v, _ in enrango]) if enrango else np.nan
            self.hist[sen].append((k, vel))
            fluidez[sen] = self.fluidez(sen)
        return res, fluidez

    def fluidez(self, sen):
        h = np.array(self.hist[sen], float)
        if len(h) < self.fps:
            return None
        n5, n10 = int(5 * self.fps), int(10 * self.fps)
        k5 = np.nanmean(h[-n5:, 0]); v5 = np.nanmean(h[-n5:, 1])
        k_largo = np.nanmean(h[:, 0]); v_largo = np.nanmean(h[:, 1])
        k10 = np.nanmean(h[-n10:, 0]); v10 = np.nanmean(h[-n10:, 1])
        if k5 >= K_CONGESTION:
            estado, col = "CONGESTED", (40, 40, 230)
        elif k5 >= K_DENSO:
            estado, col = "DENSE", (0, 190, 255)
        else:
            estado, col = "FREE FLOW", (70, 200, 70)
        # predicción: la densidad sube y la velocidad baja -> se forma un embudo
        embudo = (len(h) > 2 * n10 and k10 > 1.2 * k_largo and k10 > 0.8 * K_DENSO
                  and v10 < 0.9 * v_largo)
        mejora = len(h) > 2 * n10 and k10 < 0.8 * k_largo and v10 > 1.1 * v_largo
        tend = "bottleneck risk" if embudo else ("clearing" if mejora else "stable")
        return dict(estado=estado, color=col, k=k5, v=v5, tendencia=tend)

    def resumen(self, sen):
        """Vehículos que han pasado por clase y velocidad media (media de cada vehículo)."""
        from collections import Counter
        n = Counter(self.pasados[sen].values())
        vm = [a / k for a, k in self.vel_veh[sen].values() if k > 0]
        return n, (3.6 * float(np.mean(vm)) if vm else float("nan"))

    def banda(self, sen, horizonte):
        """(p10, p90) del error longitudinal del IDM calibrado, escalado al horizonte."""
        e = self.err.get(sen) or next(iter(self.err.values()), None)
        if not e:
            return None
        f = (horizonte / e["h"]) ** 1.5   # el error crece algo más que lineal
        return e["q10"] * f, e["q90"] * f
