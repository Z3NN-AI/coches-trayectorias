"""Paso 3: filtra cada vehículo con un EKF (CTRA adaptativo) sobre el plano de la
carretera y dibuja en cada fotograma la trayectoria prevista con su cono.

Uso: .venv/bin/python trayectorias.py salidas/pistas.json [salidas/trayectorias.mp4]
       [--modelo ctra|ctrv|cv] [--horizonte 2.0] [--sin-prior] [--sin-video]

Física del cono: el giro (w) que un conductor acepta está limitado por la aceleración
lateral (a_lat = v * w). A más velocidad, menos giro posible -> el cono se estrecha.
El centro de la predicción solo se limita por adherencia (~0,8 g), para que un
vehículo que pierde el control (p. ej. sigue recto en una rotonda) se vea tal cual.
"""
import argparse
import json
import subprocess
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

CHI2_99_2D = 9.21  # umbral de "maniobra": innovación anormalmente grande
A_FRICCION = 8.0   # m/s², límite físico de adherencia (turismos)
A_VUELCO = 3.5     # m/s², camiones y buses vuelcan antes de derrapar (~0,35 g)


# ---------------------------------------------------------------- suelo
class Suelo:
    """Z = K / (y - y0);  ppm = f / Z;  X = (x - cx) / ppm."""

    def __init__(self, c, ancho):
        self.y0, self.K, self.f, self.cx = c["y0"], c["K"], c["f"], ancho / 2
        self.xv = c.get("xv")

    def a_suelo(self, x, y):
        Z = self.K / (y - self.y0)
        return np.array([(x - self.cx) * Z / self.f, Z])

    def a_imagen(self, X, Z):
        return self.cx + X * self.f / Z, self.y0 + self.K / Z

    def jac(self, x, y, e=0.5):
        a = self.a_suelo(x, y)
        return np.column_stack([(self.a_suelo(x + e, y) - a) / e,
                                (self.a_suelo(x, y + e) - a) / e])

    def rumbo_carretera(self):
        """Ángulo (en el plano X-Z) de la dirección de la carretera, si se conoce."""
        if self.xv is None:
            return None
        return float(np.arctan2(1.0, (self.xv - self.cx) / self.f))


# ---------------------------------------------------------------- dinámica CTRA
# estado: [X, Z, v, theta, w, a]
def mover(s, dt, sub=2):
    s = s.copy()
    h = dt / sub
    for _ in range(sub):
        X, Z, v, th, w, a = s
        v2 = max(v + a * h, 0.0)
        th2 = th + w * h
        vm, thm = (v + v2) / 2, (th + th2) / 2
        s[:] = [X + vm * np.cos(thm) * h, Z + vm * np.sin(thm) * h, v2, th2, w, a]
        if v2 == 0.0:
            s[5] = max(s[5], 0.0)  # parado: no sigue "frenando" hacia atrás
    return s


def jac_num(fn, s, e=1e-5):
    f0 = fn(s)
    J = np.zeros((len(f0), len(s)))
    for i in range(len(s)):
        d = np.zeros(len(s)); d[i] = e
        J[:, i] = (fn(s + d) - f0) / e
    return J


# Vehículos pesados: giran más despacio y su límite de vuelco es bajo (~0,35 g)
INERCIA = {"coche": 1.0, "moto": 1.0, "bus": 0.5, "camion": 0.5}


def alat_confort(v):
    """Aceleración lateral que un conductor acepta (m/s²): baja con la velocidad."""
    return float(np.clip(4.0 - 0.02 * v * 3.6, 1.2, 4.0))


class Vehiculo:
    SIGMA_PX = 2.5
    SIGMA_JERK = 3.0    # m/s³
    SIGMA_ALFA = 0.1    # rad/s²
    SIGMA_V = 4.0       # m/s² equivalentes
    N_INIT = 6

    def __init__(self, tid, clase, modelo, rumbo_prior):
        self.id, self.modelo = tid, modelo
        self.votos = Counter([clase])
        self.atipicos = 0          # innovaciones grandes seguidas
        self.ancho_ema = None      # ancho de caja en metros (suavizado)
        self.rechazos_ancho = 0
        self.reanclar = False
        self.perdido = 0           # fotogramas sin verse
        self.sin_medida = 0        # fotogramas sin medida válida (tapado o perdido)
        self.oculto = False
        self.ultima_z = None
        self.robusto = True
        self.rumbo_prior = rumbo_prior
        self.buf = []
        self.s = self.P = None
        self.n = 0
        self.maniobra = 0
        self.rastro = deque(maxlen=30)
        # estados que el modelo no usa se fijan a 0 con covarianza 0
        self.libres = np.array([1, 1, 1, 1, modelo != "cv", modelo == "ctra"], float)

    @property
    def clase(self):
        return self.votos.most_common(1)[0][0]

    @property
    def inercia(self):
        return INERCIA.get(self.clase, 1.0)

    # --- inicialización con una recta sobre las primeras medidas
    def iniciar(self, dt):
        b = np.array([m for m, _ in self.buf])
        t = np.arange(len(b)) * dt
        vx, vz = np.polyfit(t, b[:, 0], 1)[0], np.polyfit(t, b[:, 1], 1)[0]
        v = float(np.hypot(vx, vz))
        th = float(np.arctan2(vz, vx))
        sig_th = 0.3
        if self.rumbo_prior is not None and v > 2:
            cand = [self.rumbo_prior, self.rumbo_prior + np.pi]
            th = min(cand, key=lambda c: abs(np.angle(np.exp(1j * (c - th)))))
            sig_th = 0.06
        R = self.buf[-1][1]
        self.s = np.array([*b[-1], v, th, 0.0, 0.0])
        self.P = np.diag([R[0, 0], R[1, 1], 9.0, sig_th**2, 0.05**2, 1.5**2])
        self._fijar()

    def _fijar(self):
        L = self.libres
        self.s *= L
        self.P *= np.outer(L, L)

    def Q(self, dt):
        th = self.s[3]
        c, s_ = np.cos(th), np.sin(th)
        G = np.array([[dt**3 / 6 * c, 0], [dt**3 / 6 * s_, 0], [dt**2 / 2, 0],
                      [0, dt**2 / 2], [0, dt], [dt, 0]])
        sj = self.SIGMA_JERK if self.modelo == "ctra" else 3.0 * self.SIGMA_JERK / dt / 25
        alfa = self.SIGMA_ALFA * (self.inercia if self.robusto else 1.0)
        Qm = G @ np.diag([sj**2, alfa**2]) @ G.T
        # ruido directo en v: cambios de ritmo que la 'a' filtrada no capta a tiempo
        Qm[2, 2] += (self.SIGMA_V * dt) ** 2
        return Qm

    def predecir(self, dt):
        F = jac_num(lambda s: mover(s, dt), self.s)
        Qm = self.Q(dt) * (10.0 if self.maniobra > 0 else 1.0)
        self.s = mover(self.s, dt)
        self.P = F @ self.P @ F.T + Qm
        self.maniobra = max(self.maniobra - 1, 0)
        self._fijar()
        return Qm

    def paso(self, z, R, dt):
        if self.s is None:
            if z is not None:
                self.buf.append((z, R))
                if len(self.buf) >= self.N_INIT:
                    self.iniciar(dt)
            return
        Qm = self.predecir(dt)
        if z is not None and self.reanclar:
            # salto del punto medido, no del vehículo: mover solo la posición
            self.reanclar = False
            self.s[:2] = z
            self.P[:2, :] = 0; self.P[:, :2] = 0; self.P[:2, :2] = R
            self.rastro.clear(); self.rastro.append(self.s[:2].copy())
            return
        if z is not None:
            Hm = np.zeros((2, 6)); Hm[0, 0] = Hm[1, 1] = 1
            y = z - self.s[:2]
            S = Hm @ self.P @ Hm.T + R
            nis = float(y @ np.linalg.solve(S, y))
            if nis > CHI2_99_2D and self.n > 5:
                self.atipicos += 1
                if self.robusto and self.atipicos < 3:
                    # salto aislado (oclusión, caja mala): se descarta la medida
                    self.rastro.append(self.s[:2].copy())
                    return
                # desvío persistente = maniobra real: inflar y seguir a las medidas
                self.P = self.P + 9.0 * Qm * 25
                self._fijar()
                S = Hm @ self.P @ Hm.T + R
                self.maniobra = 12
            else:
                self.atipicos = 0
            K = self.P @ Hm.T @ np.linalg.inv(S)
            self.s = self.s + K @ y
            self.P = (np.eye(6) - K @ Hm) @ self.P
            self.P = (self.P + self.P.T) / 2
            self._fijar()
            self.n += 1
        self.rastro.append(self.s[:2].copy())

    def fiable(self):
        return (self.s is not None and self.n >= 4
                and np.sqrt(self.P[3, 3]) < np.radians(5) and self.s[2] > 1.0)

    K_SIG_TH = 0.5  # sigmas de rumbo que abre el cono
    TAU_A = 0.5  # s: la aceleración actual se desvanece en la predicción

    def futuro(self, horizonte, paso=0.1, lider=None):
        """(centro, borde_izq, borde_der) como listas de (X, Z).
        lider: (puntos (X, Z) futuros del vehículo de delante, su velocidad, su largo)."""
        v, w, a = self.s[2], self.s[4], self.s[5]
        v_ref = max(v, 2.0)
        a_max = A_VUELCO if (self.robusto and self.inercia < 1) else A_FRICCION
        w_fis = a_max / v_ref
        w_conf = alat_confort(v) * (self.inercia if self.robusto else 1.0) / v_ref
        sig_w = np.sqrt(max(self.P[4, 4], 0))
        sig_th = np.sqrt(max(self.P[3, 3], 0))
        dw = float(np.clip(2 * sig_w, 0.3 * w_conf, w_conf))
        caminos = []
        for sgn in (0, 1, -1):
            s = self.s.copy()
            s[3] += sgn * self.K_SIG_TH * sig_th
            s[4] = np.clip(w + sgn * dw, -w_fis, w_fis)
            a0 = float(np.clip(a, -9.0, 4.0))
            pts = [s[:2].copy()]
            if lider is not None:
                lp, lv, ll = lider
                u0 = np.array([np.cos(s[3]), np.sin(s[3])])
                T_i = t_comodo(s[2], float((np.asarray(lp[0]) - s[:2]) @ u0) - ll)
            for k in range(int(round(horizonte / paso))):
                s[5] = a0 * np.exp(-(k + 0.5) * paso / self.TAU_A)
                if lider is not None:
                    lp, lv, ll = lider
                    u = np.array([np.cos(s[3]), np.sin(s[3])])
                    hueco = float((np.asarray(lp[min(k, len(lp) - 1)]) - s[:2]) @ u) - ll
                    s[5] = idm(s[2], lv, hueco, T_i) if hueco < 60 else s[5]
                s = mover(s, paso); pts.append(s[:2].copy())
            caminos.append(pts)
        return caminos



# ---------------------------------------------------------------- interacción
# IDM (Intelligent Driver Model): cuánto frena quien alcanza al de delante.
IDM_A, IDM_B, IDM_T, IDM_S0 = 1.5, 2.0, 1.2, 2.0  # m/s², m/s², s, m


def idm(v, v_lider, hueco, T=IDM_T):
    """Aceleración por la interacción con el de delante (IDM sin velocidad deseada:
    si nadie molesta, mantiene su ritmo). T = distancia temporal que ESE conductor
    considera cómoda: se estima de cómo va ahora, así solo frena si se acerca."""
    if hueco <= 0.1:
        return -9.0
    dv = v - v_lider  # >0: me acerco
    s_star = IDM_S0 + max(0.0, v * T + v * dv / (2 * np.sqrt(IDM_A * IDM_B)))
    return float(np.clip(IDM_A * (1 - (s_star / hueco) ** 2), -9.0, 0.5))


def t_comodo(v, hueco):
    """Distancia temporal que el conductor mantiene ahora mismo (s)."""
    return float(np.clip((hueco - IDM_S0) / max(v, 1.0), 0.3, 3.0))


def buscar_lider(v, otros, carril=1.6, alcance=80.0):
    """Vehículo más cercano por delante, en el mismo carril y sentido.
    Devuelve (líder, distancia entre puntos de referencia) o None."""
    u = np.array([np.cos(v.s[3]), np.sin(v.s[3])])
    n = np.array([-u[1], u[0]])
    mejor = None
    for o in otros:
        if o is v or np.cos(o.s[3] - v.s[3]) < 0.8:
            continue
        d = o.s[:2] - v.s[:2]
        delante, lateral = float(d @ u), float(d @ n)
        if 0 < delante < alcance and abs(lateral) < carril:
            if mejor is None or delante < mejor[1]:
                mejor = (o, delante)
    return mejor


def heredar(fantasmas, z, R, dt):
    """Un número nuevo del seguidor, ¿es un vehículo en memoria (oculto o perdido)?
    Los fantasmas ya están en su posición más probable (se mueven cada fotograma),
    así que basta comparar. La distancia tolerada crece con el tiempo oculto."""
    mejor, d2min = None, CHI2_99_2D
    for v in fantasmas:
        e = z - mover(v.s, dt)[:2]
        dist = float(np.hypot(*e))
        if dist > 2.0 + 1.5 * v.perdido * dt:
            continue
        d2 = float(e @ np.linalg.solve(v.P[:2, :2] + R, e))
        if d2 < d2min:
            mejor, d2min = v, d2
    return mejor


# ---------------------------------------------------------------- vista cenital
DIMENSIONES = {"coche": (4.5, 1.8), "moto": (2.0, 0.8), "bus": (12.0, 2.55),
               "camion": (10.0, 2.5)}  # largo, ancho (m)


class Cenital:
    """Plano visto desde arriba. La calibración es una homografía imagen->suelo,
    así que el fondo de la carretera se puede 'enderezar' con warpPerspective."""

    def __init__(self, suelo, alto, bg, zmin=15.0, zmax=85.0, xr=20.0):
        self.suelo, self.zmin, self.zmax, self.xr = suelo, zmin, zmax, xr
        self.s = alto / (zmax - zmin)
        self.W = int(2 * xr * self.s) // 2 * 2
        self.H = alto
        K, f, cx, y0 = suelo.K, suelo.f, suelo.cx, suelo.y0
        img_a_suelo = np.array([[K / f, 0, -cx * K / f], [0, 0, K], [0, 1, -y0]])
        suelo_a_pan = np.array([[self.s, 0, xr * self.s], [0, -self.s, zmax * self.s],
                                [0, 0, 1]])
        M = suelo_a_pan @ img_a_suelo
        bg = bg.copy()
        bg[: int(y0) + 3] = 0  # por encima del horizonte no hay suelo
        base = cv2.warpPerspective(bg, M, (self.W, self.H), flags=cv2.INTER_LINEAR)
        base = (base * 0.55).astype(np.uint8)
        for z in range(int(np.ceil(zmin / 10) * 10), int(zmax) + 1, 10):
            v = self.px([(0, z)])[0][1]
            cv2.line(base, (0, v), (self.W, v), (90, 90, 90), 1)
            cv2.putText(base, f"{z} m", (4, v - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (170, 170, 170), 1, cv2.LINE_AA)
        cv2.putText(base, "VISTA CENITAL", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(base, "camara", (self.W // 2 - 24, self.H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        self.base = base

    def px(self, pts):
        return np.array([((X + self.xr) * self.s, (self.zmax - Z) * self.s)
                         for X, Z in pts], np.int32)

    def nuevo(self):
        self.img = self.base.copy()
        self.capa = self.base.copy()

    def vehiculo(self, p, th, clase, col, texto=None, cono=None, centro=None, rastro=None,
                 relleno=True):
        L, A = DIMENSIONES.get(clase, (4.5, 1.8))
        u = np.array([np.cos(th), np.sin(th)])
        lejos = u if u[1] >= 0 else -u           # el punto medido es el lado más cercano
        c = np.asarray(p) + lejos * L / 2
        n = np.array([-u[1], u[0]])
        esq = [c + u * L / 2 + n * A / 2, c + u * L / 2 - n * A / 2,
               c - u * L / 2 - n * A / 2, c - u * L / 2 + n * A / 2]
        if rastro is not None and len(rastro) > 2:
            cv2.polylines(self.img, [self.px(rastro)], False, col, 1, cv2.LINE_AA)
        if cono is not None:
            izq, der = cono
            cv2.fillPoly(self.capa, [np.vstack([self.px(izq), self.px(der)[::-1]])], col)
        if centro is not None:
            cc = self.px(centro)
            cv2.polylines(self.img, [cc], False, col, 2, cv2.LINE_AA)
            for k in range(5, len(cc), 5):
                cv2.circle(self.img, tuple(cc[k]), 3, col, -1, cv2.LINE_AA)
        if relleno:
            cv2.fillPoly(self.img, [self.px(esq)], col)
            cv2.polylines(self.img, [self.px(esq)], True, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.polylines(self.img, [self.px(esq)], True, col, 2, cv2.LINE_AA)
        if texto:
            q = self.px([c])[0]
            cv2.putText(self.img, texto, (int(q[0]) + 8, int(q[1]) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    def final(self):
        return cv2.addWeighted(self.capa, 0.3, self.img, 0.7, 0)


# ---------------------------------------------------------------- dibujo
def color_id(tid):
    rng = np.random.default_rng(tid * 7919)
    h = int(rng.integers(0, 180))
    c = cv2.cvtColor(np.uint8([[[h, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(v) for v in c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pistas")
    ap.add_argument("salida", nargs="?", default="salidas/trayectorias.mp4")
    ap.add_argument("--calib", default="salidas/calibracion.json")
    ap.add_argument("--modelo", default="ctrv", choices=["ctrv", "ctra", "cv"])
    ap.add_argument("--horizonte", type=float, default=2.0, help="segundos a predecir")
    ap.add_argument("--sin-prior", action="store_true",
                    help="no usar la dirección de la carretera al iniciar (rotondas, cruces)")
    ap.add_argument("--sin-video", action="store_true", help="solo evaluar, sin dibujar")
    ap.add_argument("--sin-cenital", action="store_true", help="no añadir la vista cenital")
    ap.add_argument("--punto", default="mixto", choices=["caja", "pie", "pie20", "mixto"],
                    help="punto de apoyo: centro inferior de la caja, punto más bajo de la máscara, centro de su franja inferior, o mixto (franja en ligeros, caja en pesados)")
    ap.add_argument("--sin-idm", action="store_true", help="sin interacción con el de delante")
    ap.add_argument("--quitar", default="", help="desactivar comprobaciones: tapado,ancho")
    ap.add_argument("--basico", action="store_true",
                    help="sin oclusión, saltos de caja, atípicos ni masa (para comparar)")
    ap.add_argument("--jerk", type=float, default=Vehiculo.SIGMA_JERK)
    ap.add_argument("--tau", type=float, default=Vehiculo.TAU_A)
    ap.add_argument("--sv", type=float, default=Vehiculo.SIGMA_V)
    ap.add_argument("--alfa", type=float, default=Vehiculo.SIGMA_ALFA)
    ap.add_argument("--ksig", type=float, default=Vehiculo.K_SIG_TH)
    a = ap.parse_args()
    Vehiculo.SIGMA_JERK, Vehiculo.TAU_A, Vehiculo.SIGMA_V = a.jerk, a.tau, a.sv
    Vehiculo.SIGMA_ALFA, Vehiculo.K_SIG_TH = a.alfa, a.ksig

    datos = json.load(open(a.pistas))
    fps, W, Hh = datos["fps"], datos["ancho"], datos["alto"]
    suelo = Suelo(json.load(open(a.calib)), W)
    prior = None if a.sin_prior else suelo.rumbo_carretera()
    dt = 1 / fps
    y_min = suelo.y0 + suelo.K / 80.0  # no seguir más allá de 80 m (ruido de profundidad)
    print(f"Modelo {a.modelo.upper()}, horizonte {a.horizonte}s, "
          f"prior de carretera: {'sí' if prior is not None else 'no'}")

    cap = out = None
    if not a.sin_video:
        cap = cv2.VideoCapture(datos["video"])
        tmp = Path(a.salida).with_suffix(".tmp.mp4")
        cen = None
        if not a.sin_cenital:
            from calibrar import fondo
            cen = Cenital(suelo, Hh, fondo(datos["video"]))
        Wt = W + (cen.W if cen else 0)
        out = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (Wt, Hh))
    vehs: dict[int, Vehiculo] = {}    # número del seguidor -> vehículo
    memoria: set = set()              # vehículos con estado (vistos o fantasmas)
    velocidades = defaultdict(list)
    medidas = defaultdict(dict)       # id vehículo -> {frame: (X, Z)}
    predicciones = []
    reapariciones = []                # (segundos sin verse, err física, err quieto, oculto?)
    hs = [1.0, 2.0]
    MEM_OCULTO, MEM_PERDIDO = int(3.0 * fps), int(0.5 * fps)

    ocluidas = Counter()
    herencias = 0
    giros = defaultdict(list)   # clase -> |w| de vehículos fiables (aquí todos van rectos)

    def px(pts):
        return np.array([suelo.a_imagen(X, Z) for X, Z in pts if Z > 1.0], np.int32)

    for i, dets in enumerate(datos["frames"]):
        img = None
        if cap is not None:
            ok, img = cap.read()
            if not ok:
                break

        # ---- 1. medidas de este fotograma
        obs = []
        for d in dets:
            x1, y1, x2, y2 = d["caja"]
            if y2 < y_min:
                continue
            punto = a.punto
            if punto == "mixto":
                punto = "caja" if d["clase"] in ("camion", "bus") else "pie20"
            if punto != "caja" and punto in d:
                xb, yb = d[punto]
            else:
                xb, yb = (x1 + x2) / 2, y2
            cortado = x1 < 3 or x2 > W - 3 or y2 > Hh - 3
            tapado = 0.0
            if not a.basico:
                area = (x2 - x1) * (y2 - y1)
                tapado = sum(max(0, min(x2, o["caja"][2]) - max(x1, o["caja"][0])) *
                             max(0, min(y2, o["caja"][3]) - max(y1, o["caja"][1]))
                             for o in dets if o is not d and o["caja"][3] > y2 + 2) / area
            J = suelo.jac(xb, yb)
            R = J @ np.diag([Vehiculo.SIGMA_PX**2] * 2) @ J.T + np.eye(2) * 0.05
            obs.append((d, xb, yb, cortado, tapado, R))

        # ---- 2. asociar cada medida a un vehículo (o heredar uno en memoria)
        vistos = {}
        fantasmas = [v for v in memoria if v.perdido > 0]
        for d, xb, yb, cortado, tapado, R in obs:
            tid = d["id"]
            v = vehs.get(tid)
            if v is not None and v not in memoria and v.s is not None:
                v = None  # se olvidó: empieza de nuevo
            if v is None and not a.basico and fantasmas:
                v = heredar(fantasmas, suelo.a_suelo(xb, yb), R, dt)
                if v is not None:
                    fantasmas.remove(v)
                    herencias += 1
            if v is None:
                v = Vehiculo(tid, d["clase"], a.modelo, prior)
                v.robusto = not a.basico
            else:
                v.votos[d["clase"]] += 1
            vehs[tid] = v
            if any(v is w for w, _ in vistos.values()):
                continue
            x1, y1, x2, y2 = d["caja"]
            ppm = suelo.f * (yb - suelo.y0) / suelo.K
            if not a.basico:
                if "tapado" not in a.quitar and tapado > 0.12:
                    cortado = True; ocluidas["tapado"] += 1
                ancho_m = (x2 - x1) / ppm
                if ("ancho" not in a.quitar and v.ancho_ema is not None
                        and abs(ancho_m / v.ancho_ema - 1) > 0.25):
                    v.rechazos_ancho += 1
                    if v.rechazos_ancho >= 5:
                        # la caja cambió de forma: nueva referencia de posición,
                        # pero se conserva la velocidad, el rumbo y el giro
                        v.ancho_ema, v.rechazos_ancho = None, 0
                        v.reanclar = True
                    else:
                        cortado = True; ocluidas["salto de caja"] += 1
                if not cortado:
                    v.rechazos_ancho = 0
                    v.ancho_ema = ancho_m if v.ancho_ema is None else 0.8 * v.ancho_ema + 0.2 * ancho_m
            z = None if cortado else suelo.a_suelo(xb, yb)
            if z is not None:
                medidas[v.id][i] = z
                v.caja_m = ((x2 - x1) / ppm, (y2 - y1) / ppm, ((x1 + x2) / 2 - xb) / ppm)
                if v.s is not None and v.sin_medida >= 5 and v.ultima_z is not None:
                    pred = mover(v.s, dt)[:2]
                    reapariciones.append(((v.sin_medida + 1) * dt, float(np.hypot(*(z - pred))),
                                          float(np.hypot(*(z - v.ultima_z))), v.perdido > 0))
                v.ultima_z = z
                v.sin_medida = 0
            else:
                v.sin_medida += 1
            v.paso(z, R, dt)
            v.perdido, v.oculto = 0, False
            memoria.add(v)
            vistos[tid] = (v, d["caja"])

        # ---- 3. los que no se ven: siguen la física (trayectoria más probable)
        cajas = [o[0]["caja"] for o in obs]
        en_vista = {id(v) for v, _ in vistos.values()}
        for v in list(memoria):
            if id(v) in en_vista:
                continue
            if v.s is None or a.basico or not v.fiable():
                memoria.discard(v); continue
            v.predecir(dt)
            v.perdido += 1
            v.sin_medida += 1
            xp, yp = suelo.a_imagen(*v.s[:2]) if v.s[1] > 1 else (-1e9, -1e9)
            # ¿hay un vehículo más cercano justo delante de donde debería estar?
            v.oculto = any(c[0] - 10 <= xp <= c[2] + 10 and c[1] <= yp <= c[3] + 10
                           for c in cajas)
            fuera = not (-50 < xp < W + 50 and y_min < yp < Hh + 20)
            limite = MEM_OCULTO if v.oculto else MEM_PERDIDO
            if fuera or v.perdido > limite:
                memoria.discard(v)

        # ---- 4. dibujo y registro
        capa = img.copy() if img is not None else None
        if img is not None and cen is not None:
            cen.nuevo()
            for v, _ in vistos.values():  # aún sin predicción fiable: en gris
                if not v.fiable() and i in medidas[v.id]:
                    th = prior if prior is not None else (v.s[3] if v.s is not None else np.pi / 2)
                    cen.vehiculo(medidas[v.id][i], th, v.clase, (140, 140, 140))
        dibujar = [(v, caja) for v, caja in vistos.values() if v.fiable()]
        dibujar += [(v, None) for v in memoria if v.perdido > 0]
        activos = [v for v, _ in dibujar]
        futuros = {id(v): v.futuro(a.horizonte) for v in activos}
        lideres = {id(v): buscar_lider(v, activos) for v in activos}
        for _ in range(0 if a.sin_idm else 3):  # la frenada se propaga por la cola
            nuevos = {}
            for v in activos:
                L = lideres[id(v)]
                if L is None:
                    nuevos[id(v)] = futuros[id(v)]
                    continue
                lv, _ = L
                dl = DIMENSIONES.get(lv.clase, (4.5, 1.8))[0]
                nuevos[id(v)] = v.futuro(a.horizonte, lider=(futuros[id(lv)][0], lv.s[2], dl))
            futuros = nuevos
        for v, caja in dibujar:
            kmh = v.s[2] * 3.6
            centro, izq, der = futuros[id(v)]
            L = lideres.get(id(v))
            con_lider = L is not None and L[1] < 40
            if caja is not None:
                velocidades[v.id].append((i, round(kmh, 1)))
                giros["pesado" if v.clase in ("camion", "bus") else "ligero"].append(abs(v.s[4]))
                for h in hs:
                    k = int(round(h / 0.1))
                    if k < len(centro):
                        hw = float(np.linalg.norm(np.subtract(izq[k], der[k]))) / 2
                        predicciones.append((v.id, i, h, *centro[k], v.s[3], v.n, hw, con_lider))
            if img is None:
                continue
            col = color_id(v.id)
            fantasma = caja is None
            if fantasma:  # caja estimada en la posición predicha
                xp, yp = suelo.a_imagen(*v.s[:2])
                ppm = suelo.f / v.s[1]
                wm, hm, dx = getattr(v, "caja_m", (1.8, 1.5, 0.0))
                cxp = xp + dx * ppm
                caja = (cxp - wm * ppm / 2, yp - hm * ppm, cxp + wm * ppm / 2, yp)
            x1, y1, x2, y2 = map(int, caja)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 1 if fantasma else 2)
            if len(v.rastro) > 2:
                cv2.polylines(img, [px(v.rastro)], False, col, 1, cv2.LINE_AA)
            pi, pd = px(izq), px(der)
            if len(pi) > 1 and len(pd) > 1:
                cv2.fillPoly(capa, [np.vstack([pi, pd[::-1]])], col)
            c = px(centro)
            if len(c) > 1:
                cv2.polylines(img, [c], False, col, 1 if fantasma else 2, cv2.LINE_AA)
                for k in range(5, len(c), 5):
                    cv2.circle(img, tuple(c[k]), 3, col, -1, cv2.LINE_AA)
            if cen is not None:
                cen.vehiculo(v.s[:2], v.s[3], v.clase, col, f"{kmh:.0f}", (izq, der),
                             centro, list(v.rastro), relleno=not fantasma)
                if con_lider and not a.sin_idm:  # enlace con el vehículo que le condiciona
                    q1, q2 = cen.px([v.s[:2], L[0].s[:2]])
                    cv2.line(cen.img, tuple(map(int, q1)), tuple(map(int, q2)), (230, 230, 230), 1,
                             cv2.LINE_AA)
            txt = f"#{v.id} {kmh:.0f} km/h"
            if fantasma:
                txt = f"#{v.id} {'oculto' if v.oculto else 'perdido'} {v.perdido * dt:.1f}s"
            elif v.maniobra:
                txt += " !"
            (tw, th_), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, y1 - th_ - 6), (x1 + tw + 4, y1), col, -1)
            cv2.putText(img, txt, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 1, cv2.LINE_AA)
        if img is not None:
            img = cv2.addWeighted(capa, 0.25, img, 0.75, 0)
            cv2.putText(img, f"{a.modelo.upper()}  |  prediccion {a.horizonte:.1f} s "
                        f"(punto = 0.5 s)  |  ! = maniobra detectada", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            if cen is not None:
                img = np.hstack([img, cen.final()])
            out.write(img)

    if out is not None:
        out.release(); cap.release()
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264",
                        "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        a.salida], check=True)
        tmp.unlink()
        json.dump(velocidades, open(Path(a.salida).with_suffix(".velocidades.json"), "w"))

    print(f"Medidas descartadas: {dict(ocluidas)}  |  identidades heredadas: {herencias}")
    if reapariciones:
        r = np.array(reapariciones)
        print("Al reaparecer: distancia entre dónde está y dónde se suponía que estaba (m)")
        for lo, hi in [(0.2, 0.5), (0.5, 1.0), (1.0, 3.1)]:
            m = (r[:, 0] >= lo) & (r[:, 0] < hi)
            if m.any():
                print(f"  sin verse {lo:.1f}-{hi:.1f}s n={m.sum():3d}  con física {np.median(r[m, 1]):5.2f}"
                      f"  | si se quedara quieto {np.median(r[m, 2]):5.2f}")
    for g, w in giros.items():
        w = np.degrees(np.array(w))
        print(f"Giro aparente {g:7s} (debería ser ~0): mediana {np.median(w):.2f}°/s, "
              f"p95 {np.percentile(w, 95):.2f}°/s, máx {w.max():.1f}°/s  (n={len(w)})")
    # ---- evaluación: predicción vs posición medida h segundos después
    print("Error de predicción (posición medida después vs predicha), metros:")
    for h in hs:
        for nombre, cond in [("se alejan", lambda th: np.sin(th) > 0),
                             ("se acercan", lambda th: np.sin(th) < 0)]:
            lon, lat, tot, lat_nuevo, dentro, anchos = [], [], [], [], [], []
            lon_lider = []
            for tid, f, hh, X, Z, th, edad, hw, cl in predicciones:
                if hh != h or not cond(th):
                    continue
                real = medidas[tid].get(f + int(round(h * fps)))
                if real is None:
                    continue
                e = real - np.array([X, Z])
                u = np.array([np.cos(th), np.sin(th)])
                lon.append(e @ u); lat.append(e @ np.array([-u[1], u[0]]))
                if cl:
                    lon_lider.append(lon[-1])
                if edad < 20:
                    lat_nuevo.append(lat[-1])
                tot.append(np.hypot(*e))
                dentro.append(abs(lat[-1]) <= hw); anchos.append(hw)
            if tot:
                print(f"  {h:.0f}s {nombre:10s} n={len(tot):5d}  total mediana {np.median(tot):5.2f}"
                      f"  p90 {np.percentile(tot, 90):5.2f}  | longitudinal sesgo "
                      f"{np.median(lon):+5.2f}  | lateral mediana abs {np.median(np.abs(lat)):4.2f}"
                      + f"\n        cono: semiancho mediano {np.median(anchos):.2f} m, contiene el {100 * np.mean(dentro):.0f}% de las posiciones reales"
                      + (f"  | recién vistos: lateral con signo {np.median(lat_nuevo):+.2f} m (n={len(lat_nuevo)})" if lat_nuevo else "")
                      + (f"\n        con líder a <40 m (n={len(lon_lider)}): error longitudinal mediana abs "
                         f"{np.median(np.abs(lon_lider)):.2f} m, sesgo {np.median(lon_lider):+.2f} m" if lon_lider else ""))


if __name__ == "__main__":
    main()
