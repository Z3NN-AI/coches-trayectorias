"""Simulador de autovía con el CONDUCTOR PROMEDIO aprendido de los vídeos.

Cada vehículo es un agente con muy pocas opciones: seguir en su carril (acelerando o
frenando según el IDM calibrado) o pasar a un carril vecino. Nada de reglas a mano:
- pedal: IDM con los parámetros calibrados por sentido y clase, y la personalidad de
  cada conductor (su velocidad deseada y su distancia) sacada de la distribución real;
- volante: la probabilidad de cambiar de carril la da el modelo aprendido de los
  cambios reales (comportamiento.py, sin deriva lateral). Sale lo que hace la gente,
  incluido el que se queda a vivir en el carril rápido;
- física y normas solo como límite: no se cambia si el de detrás tendría que frenar
  más de 4 m/s²; en UK los camiones no usan el carril rápido de 3+ carriles.

Modos:
  escena    arranca con los vehículos reales de un trozo de vídeo; los que entran en
            escena lo hacen cuando y como entraron en la realidad, pero DENTRO lo
            decide el conductor promedio. Se compara con lo que pasó de verdad.
  sintetico autovía de 2 km con distintas demandas (ligero ... muy denso) y camiones;
            mide el diagrama fundamental y busca atascos fantasma.

Uso:
  .venv/bin/python simulador.py escena salidas/lista/01_wqctLW0Hb_0 --trozo 004 --sentido alejan
  .venv/bin/python simulador.py sintetico --camara 01_wqctLW0Hb_0 --sentido alejan
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from comportamiento import CONTEXTO, LARGO, MANUAL, asignar_carril, cargar, rasgos

DT = 0.1                 # s, paso de integración
PASO_DECISION = 0.5      # s, cada cuánto se plantea cambiar de carril
B_SEGURO = 4.0           # m/s², frenada máxima que un cambio puede imponer al de detrás
DURACION_CAMBIO = {0: 4.0, 1: 6.0}   # s (ligero, pesado)
SIGMA_V0 = 0.05          # oscilación típica de la velocidad preferida (±5 %, hasta ±10 % según el conductor)
TAU_V0 = 30.0            # s, cuánto tarda en cambiar ese "humor"
RUIDO_PEDAL = 0.15       # m/s², imprecisión instantánea del pie
V_SIN_ADELANTAR_DENTRO = 60 / 3.6   # por debajo (cola) cada carril va a su ritmo
GANAS_BASE = 5.0         # calibración global: iguala la tasa real de cambios por veh·km
SIGMA_GANAS = 1.0        # dispersión entre conductores de las ganas de cambiar de carril
P_VUELVE = 0.0           # fracción de conductores de coche que vuelven al carril 1 (el resto lo tapa)
T_VOLVER = 6.0           # s, espera media de un camión para volver al carril 1 (regla)
ESPERA_CAMBIO = 3.0      # s sin volver a plantearse otro cambio
# Pesados: lo que los datos no identifican bien se toma de la física
A_MAX_PESADO = 0.8       # m/s², un camión cargado no acelera más
CLASES = ["coche", "camion", "bus", "moto"]


# ---------------------------------------------------------------- conductores
class Poblacion:
    """Parámetros de la población de una cámara y sentido (de comportamiento.json)."""

    def __init__(self, info, sentido, modelo, rng, idm_de=None, perfil_de=None):
        """idm_de: sentido del que tomar el seguimiento (mejor el denso, donde se ve la
        interacción); perfil_de: del que tomar la velocidad deseada (mejor el fluido)."""
        self.rng = rng
        self.nc = len(info["centros"][sentido])
        self.centros = np.array(info["centros"][sentido])
        self.mano = info["config"]["mano"]
        idm, si, sp = info["idm"], idm_de or sentido, perfil_de or sentido
        base = idm.get(si, MANUAL)
        self.idm = {g: dict(idm.get(f"{si}/{n}", base)) for g, n in ((0, "ligero"), (1, "pesado"))}
        self.idm[1]["a"] = min(self.idm[1]["a"], A_MAX_PESADO)
        self.perfil = {g: info["perfil"].get(f"{sp}/{n}") for g, n in ((0, "ligero"), (1, "pesado"))}
        # distancia temporal de cada conductor: la que la gente mantiene de verdad en
        # seguimiento estable (sentido denso, donde se sigue a alguien)
        self.T_dist = {g: (info["perfil"].get(f"{si}/{n}") or {}).get("T_estable")
                       for g, n in ((0, "ligero"), (1, "pesado"))}
        self.modelo, self.base = modelo["modelo"], modelo["base"]
        self.restringe_pesados = self.mano == "izquierda" and self.nc >= 3

    def muestra(self, p):
        """Un valor de una distribución dada por sus percentiles 5/25/50/75/95."""
        qs = np.array([5, 25, 50, 75, 95]) / 100
        vals = np.array([p[f"p{q}"] for q in (5, 25, 50, 75, 95)])
        return float(np.interp(self.rng.uniform(0.02, 0.98), qs, vals))

    def v_libre(self, g):
        """Velocidad deseada de un conductor nuevo: muestreo de la distribución real."""
        return self.muestra((self.perfil.get(g) or self.perfil[0])["v_libre_kmh"]) / 3.6

    def personalidad(self, g, v0=None, carril=None):
        p = dict(self.idm[g])
        por_carril = (self.perfil.get(g) or {}).get("v_libre_carril") or {}
        dist = por_carril.get(str(carril)) or por_carril.get(carril)
        if v0 is not None:
            p["v0"] = v0
        elif dist and carril is not None:   # el que entra por el carril rápido quiere ir rápido
            p["v0"] = self.muestra(dist) / 3.6
        else:
            p["v0"] = self.v_libre(g)
        # TIPOS DE CONDUCTOR (supuestos, no medibles en 55 m de cámara):
        # ganas de cambiar de carril: multiplica las odds del modelo aprendido
        # (lognormal: la mayoría cerca de 1, algunos que no cambian nunca, otros que no paran)
        p["ganas"] = float(self.rng.lognormal(0.0, SIGMA_GANAS))
        # IMPRECISIÓN HUMANA: la velocidad preferida oscila (Ornstein-Uhlenbeck), cada uno
        # con su amplitud: el del control de crucero casi nada, el despistado bastante
        p["v0_base"] = p["v0"]
        p["osc"] = float(self.rng.uniform(0.0, 2.0) * SIGMA_V0)
        p["eps"] = float(self.rng.normal(0.0, p["osc"]))
        # vuelve al carril 1 cuando no adelanta (los camiones siempre, por norma)
        p["vuelve"] = bool(g == 1 or self.rng.uniform() < P_VUELVE)
        Td = self.T_dist.get(g) or self.T_dist.get(0)
        if Td:   # cada uno con su distancia real: el prudente, el normal, el que se pega
            p["T"] = self.muestra(Td)
        else:
            p["T"] *= float(self.rng.lognormal(0, 0.25))
        return p

    def p_cambio(self, F):
        """P(cambia en 3 s) corregida del entrenamiento equilibrado -> prob. real."""
        pb = self.modelo.predict_proba(np.array([[f[k] for k in CONTEXTO] for f in F]))[:, 1]
        pb = np.clip(pb, 1e-6, 1 - 1e-6)
        odds = pb / (1 - pb) * self.base / (1 - self.base)
        return odds / (1 + odds)


def idm_vec(v, vl, s, p):
    s = np.maximum(s, 0.1)
    s_star = p["s0"] + np.maximum(0, v * p["T"] + v * (v - vl) / (2 * np.sqrt(p["a"] * p["b"])))
    return p["a"] * (1 - (v / p["v0"]) ** 4 - (s_star / s) ** 2)


class Trafico:
    """Vehículos de un sentido. x: avance (m); carril 1 = arcén."""

    def __init__(self, pob, x_fin, frontera=None):
        self.pob, self.x_fin, self.frontera = pob, x_fin, frontera
        self.veh = []          # dicts
        self.t = 0.0
        self.cambios = []      # (t, id, de, a)
        self.vetos = {}        # decisiones del modelo vetadas por reglas a mano
        self.salidas = {}      # id -> t de salida
        self.sig_decision = 0.0

    def cabe(self, x, carril, L):
        """¿Hay sitio en ese carril y posición? (para no meter un vehículo encima de otro)."""
        for m in self.veh:
            if m["carril"] == carril:
                if m["x"] >= x and m["x"] - m["L"] - x < 3.0:
                    return False
                if m["x"] < x and x - L - m["x"] < 3.0:
                    return False
        return True

    def entra(self, vid, clase, x, carril, v, v0=None, t_entrada=None):
        """v0: velocidad deseada; si no se da, la de su personalidad (nunca menor que
        la que ya lleva)."""
        g = 1 if clase in ("camion", "bus") else 0
        if self.pob.restringe_pesados and g == 1 and carril == self.pob.nc:
            carril = self.pob.nc - 1
        p = self.pob.personalidad(g, v0, carril)
        p["v0"] = max(p["v0"], 1.05 * v, 5.0)
        self.veh.append(dict(id=vid, clase=clase, g=g, x=x, carril=int(carril), v=v,
                             p=p, L=LARGO.get(clase, 4.5),
                             lat=float(carril), lat0=float(carril), prog=1.0, espera=0.0,
                             t0=self.t if t_entrada is None else t_entrada))

    def lider(self, i, carril=None):
        me = self.veh[i]
        c = me["carril"] if carril is None else carril
        mejor = None
        for j, o in enumerate(self.veh):
            if j != i and o["carril"] == c and o["x"] > me["x"]:
                if mejor is None or o["x"] < self.veh[mejor]["x"]:
                    mejor = j
        return mejor

    def detras(self, i, carril):
        me = self.veh[i]
        mejor = None
        for j, o in enumerate(self.veh):
            if j != i and o["carril"] == carril and o["x"] <= me["x"]:
                if mejor is None or o["x"] > self.veh[mejor]["x"]:
                    mejor = j
        return mejor

    def acc(self, i, lider):
        me = self.veh[i]
        if lider is None:
            if self.frontera is not None:      # lo que hay más allá de lo que se ve
                vl = self.frontera(self.t, me["carril"])
                if vl is not None:
                    return float(idm_vec(me["v"], vl, self.x_fin + 15 - me["x"], me["p"]))
            return float(idm_vec(me["v"], me["v"], 1e4, me["p"]))
        o = self.veh[lider]
        return float(idm_vec(me["v"], o["v"], o["x"] - me["x"] - o["L"], me["p"]))

    def lideres(self):
        """Líder de cada vehículo ordenando cada carril una vez por paso."""
        out = [None] * len(self.veh)
        por = {}
        for i, m in enumerate(self.veh):
            por.setdefault(m["carril"], []).append(i)
        for lst in por.values():
            lst.sort(key=lambda i: self.veh[i]["x"])
            for a_, b_ in zip(lst[:-1], lst[1:]):
                out[a_] = b_
        return out

    def no_adelantar_por_dentro(self, acc):
        """NORMA (no aprendida): en fluido no se adelanta por el carril interior. Si el
        vehículo del carril de fuera (carril + 1), algo por delante, va más lento, se
        ajusta la velocidad a la suya (modelo europeo de Treiber, umbral 60 km/h)."""
        por = {}
        for i, m in enumerate(self.veh):
            por.setdefault(m["carril"], []).append(i)
        for c, lst in por.items():
            fuera = sorted(por.get(c + 1, []), key=lambda j: self.veh[j]["x"])
            if not fuera:
                continue
            xs = np.array([self.veh[j]["x"] for j in fuera])
            for i in lst:
                me = self.veh[i]
                k = np.searchsorted(xs, me["x"] - 2.0)   # el primero de fuera a mi altura o delante
                if k >= len(fuera):
                    continue
                o = self.veh[fuera[k]]
                d = o["x"] - me["x"]
                if d > 60 or o["v"] >= me["v"] or o["v"] < V_SIN_ADELANTAR_DENTRO:
                    continue
                # seguirle de lado: como un líder virtual a distancia 'd' + hueco mínimo
                a_lado = float(idm_vec(me["v"], o["v"], max(d, 0.0) + me["p"]["s0"] + 2.0, me["p"]))
                acc[i] = min(acc[i], a_lado)
        return acc

    def paso(self):
        lid = self.lideres()
        acc = [np.clip(self.acc(i, lid[i]), -9.0, 4.0) for i in range(len(self.veh))]
        acc = [float(np.clip(a_, -9.0, 4.0)) for a_ in self.no_adelantar_por_dentro(acc)]
        rng = self.pob.rng
        k = DT / TAU_V0
        for i, me in enumerate(self.veh):
            p = me["p"]
            if "v0_base" in p:   # deriva lenta de la velocidad preferida + ruido de pedal
                p["eps"] += -p["eps"] * k + p["osc"] * np.sqrt(2 * k) * rng.normal()
                p["v0"] = p["v0_base"] * (1 + p["eps"])
                acc[i] += RUIDO_PEDAL * rng.normal()
            v2 = max(me["v"] + acc[i] * DT, 0.0)
            me["x"] += (me["v"] + v2) / 2 * DT
            me["v"] = v2
            me["espera"] = max(0.0, me["espera"] - DT)
            if me["prog"] < 1.0:
                me["prog"] = min(1.0, me["prog"] + DT / DURACION_CAMBIO[me["g"]])
                me["lat"] = me["lat0"] + (me["carril"] - me["lat0"]) * (3 - 2 * me["prog"]) * me["prog"] ** 2
        if self.t + 1e-9 >= self.sig_decision:
            self.decidir()
            self.sig_decision += PASO_DECISION
        for me in [m for m in self.veh if m["x"] > self.x_fin]:
            self.salidas[me["id"]] = (self.t, me["carril"])
            self.veh.remove(me)
        self.t += DT

    def decidir(self):
        cand = [i for i, m in enumerate(self.veh) if m["prog"] >= 1.0 and m["espera"] <= 0]
        if not cand or len(self.veh) == 0:
            return
        X = np.array([m["x"] for m in self.veh]); V = np.array([m["v"] for m in self.veh])
        C = np.array([m["carril"] for m in self.veh]); L = np.array([m["L"] for m in self.veh])
        ids = np.array([m["id"] for m in self.veh])
        p_ref = self.pob.idm[0]
        F = rasgos(ids, X, V, C, L, np.zeros(len(X)), self.pob.centros, p_ref, elegidos=cand)
        if not F:
            return
        P3 = self.pob.p_cambio(F)
        ganas = np.array([self.veh[f["i"]]["p"].get("ganas", 1.0) for f in F])
        odds = P3 / (1 - P3 + 1e-12) * ganas * GANAS_BASE
        P3 = odds / (1 + odds)
        h = 1 - (1 - P3) ** (PASO_DECISION / 3.0)    # prob. en este medio segundo
        tira = self.pob.rng.uniform(size=len(F))
        hechos = set()
        for f, hh, u in sorted(zip(F, h, tira), key=lambda z: -z[1]):
            i, nuevo = f["i"], f["a"]
            if u >= hh or i in hechos:
                continue
            me = self.veh[i]
            if self.pob.restringe_pesados and me["g"] == 1 and nuevo == self.pob.nc:
                self.vetos["camion_carril3"] = self.vetos.get("camion_carril3", 0) + 1
                continue
            if not self.seguro(i, nuevo):
                self.vetos["seguridad"] = self.vetos.get("seguridad", 0) + 1
                continue
            self.vetos["aceptados"] = self.vetos.get("aceptados", 0) + 1
            self.cambios.append((self.t, me["id"], me["carril"], nuevo))
            me["lat0"], me["carril"], me["prog"] = me["lat"], nuevo, 0.0
            me["espera"] = ESPERA_CAMBIO + DURACION_CAMBIO[me["g"]]
            hechos.add(i)
        self.volver_al_1(cand, hechos)

    def volver_al_1(self, cand, hechos):
        """REGLA (no aprendida): un pesado (o un coche de los que cumplen) fuera del carril 1 vuelve a él cuando no está
        adelantando a nadie allí y el hueco es seguro. En vídeo nunca se ve a un camión
        volver en fluido (pasa fuera de cámara), pero el 73 % van por el carril 1."""
        for i in cand:
            me = self.veh[i]
            if i in hechos or not me["p"].get("vuelve", me["g"] == 1) or me["carril"] <= 1 or me["prog"] < 1.0:
                continue
            dl = self.lider(i, me["carril"] - 1)
            if dl is not None:
                o = self.veh[dl]
                hueco = o["x"] - o["L"] - me["x"]
                # sigue adelantando si allí delante hay alguien cerca y más lento que él
                if me["g"] == 1:   # camión: deja de adelantar en cuanto pasa al más lento
                    sigue = hueco < max(2.0 * me["v"], 30.0) and o["v"] < me["v"] - 1.0
                else:              # coche: no vuelve si enseguida tendría que frenar
                    sigue = hueco < max(5.0 * me["v"], 80.0) and o["v"] < me["p"]["v0"] - 2.0
                if sigue:
                    continue
            if self.pob.rng.uniform() > PASO_DECISION / T_VOLVER:   # no al instante
                continue
            if not self.seguro(i, me["carril"] - 1):
                continue
            self.vetos["regla_volver_al_1"] = self.vetos.get("regla_volver_al_1", 0) + 1
            self.cambios.append((self.t, me["id"], me["carril"], me["carril"] - 1))
            me["lat0"], me["carril"], me["prog"] = me["lat"], me["carril"] - 1, 0.0
            me["espera"] = ESPERA_CAMBIO + DURACION_CAMBIO[me["g"]]

    def seguro(self, i, nuevo):
        me = self.veh[i]
        dl = self.lider(i, nuevo)
        if dl is not None and self.veh[dl]["x"] - me["x"] - self.veh[dl]["L"] < 2.0:
            return False
        dt_ = self.detras(i, nuevo)
        if dt_ is None:
            return True
        o = self.veh[dt_]
        hueco = me["x"] - o["x"] - me["L"]
        return hueco > 2.0 and idm_vec(o["v"], me["v"], hueco, o["p"]) > -B_SEGURO


# ---------------------------------------------------------------- modo escena
def modelo_sin_trozo(a, cam, trozo):
    """Reentrena la decisión sin el trozo que se va a simular (si no, el simulador
    'recordaría' los cambios que tiene que adivinar)."""
    ruta = Path(a.comportamiento).with_suffix(".volante.csv")
    if not ruta.exists():
        return None
    from sklearn.ensemble import HistGradientBoostingClassifier
    mb = pd.read_csv(ruta)
    vid = mb.id.str.split("@").str[0].astype(int)
    fuera = (mb.camara == cam) & (vid // 100000 == trozo)
    tr = mb[~fuera]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         l2_regularization=1.0, class_weight="balanced",
                                         random_state=0).fit(tr[CONTEXTO], tr.y)
    print(f"  decisión reentrenada sin el trozo {trozo} de {cam}: {int(tr.y.sum())} cambios de "
          f"entrenamiento ({int(mb[fuera].y.sum())} apartados)")
    return {"modelo": clf, "columnas": CONTEXTO, "base": float(tr.y.mean())}


def escena(a, info, modelo):
    df, fps, cfg = cargar(a.camara)
    cam = Path(a.camara).name
    centros = {k: np.array(v) for k, v in info["centros"].items()}
    df, cambios_reales = asignar_carril(df, centros, fps)
    trozo = int(a.trozo)
    g = df[(df.id // 100000 == trozo) & (df.sentido == a.sentido) & (df.carril > 0)].copy()
    if g.empty:
        raise SystemExit("trozo/sentido sin datos")
    # t = 0 común a los dos sentidos: primer fotograma del trozo
    f0 = int(df[df.id // 100000 == trozo].frame.min()) + int(a.desde * fps)
    f1 = f0 + int(a.duracion * fps)
    x_ini, x_fin = g.x.quantile(0.03), g.x.quantile(0.97)
    rng = np.random.default_rng(a.semilla)
    modelo = modelo_sin_trozo(a, cam, trozo) or modelo
    # velocidad deseada: la del sentido más fluido (en la cola nadie va a la que quiere)
    fluido = max(("alejan", "acercan"), key=lambda s_: (info["perfil"].get(f"{s_}/ligero") or
                                                         {"v_libre_kmh": {"p50": 0}})["v_libre_kmh"]["p50"])
    pob = Poblacion(info, a.sentido, modelo, rng, perfil_de=fluido)
    print(f"  velocidad deseada tomada del sentido '{fluido}'; seguimiento del sentido '{a.sentido}'")

    # frontera de salida: velocidad real en los últimos 10 m de cada carril (lo que
    # pasa más allá de lo que vemos condiciona a los de dentro, p. ej. una cola)
    sal = g[g.x > x_fin - 10]
    v_sal = {c: s_.groupby("frame").v.mean() for c, s_ in sal.groupby("carril")}

    def frontera(t, carril):
        s_ = v_sal.get(carril)
        if s_ is None or s_.empty:
            return None
        fr = f0 + int(t * fps)
        k = s_.index.searchsorted(fr)
        if k >= len(s_) or abs(s_.index[min(k, len(s_) - 1)] - fr) > 2 * fps:
            return None
        return float(s_.iloc[min(k, len(s_) - 1)])

    sim = Trafico(pob, x_fin, frontera)
    primeros = g.sort_values("frame").groupby("id").first()
    # velocidad de entrada: mediana de su primer segundo (la primera medida de un
    # vehículo recién detectado, sobre todo un camión, sale baja mientras el filtro arranca)
    primeros["v"] = g.sort_values("frame").groupby("id").head(int(fps)).groupby("id").v.median()
    presentes = g[g.frame == f0].set_index("id")
    for vid, r in presentes.iterrows():
        if sim.cabe(r.x, r.carril, LARGO.get(r.clase, 4.5)):
            sim.entra(vid, r.clase, r.x, r.carril, r.v, t_entrada=0.0)
    llegadas = primeros[(primeros.frame > f0) & (primeros.frame < f1)].sort_values("frame")
    llegadas = list(llegadas.itertuples())
    real_pos = {}   # (id, frame) -> x, carril
    for r in g[(g.frame >= f0) & (g.frame <= f1)][["id", "frame", "x", "carril"]].itertuples(index=False):
        real_pos[(r.id, r.frame)] = (r.x, r.carril)
    registro = []   # (t, id, x, lat, v, clase) para dibujar
    k, espera = 0, []
    while sim.t < a.duracion:
        fr = f0 + int(round(sim.t * fps))
        while k < len(llegadas) and llegadas[k].frame <= fr:
            espera.append(llegadas[k]); k += 1
        resto = []
        for r in espera:   # entra cuando hay sitio (si no, espera: como una cola real)
            x_e = min(r.x, x_ini + 5) if r.x < x_ini + 5 else r.x
            if sim.cabe(x_e, r.carril, LARGO.get(r.clase, 4.5)):
                # entra a su velocidad real salvo que el de delante esté tan cerca que
                # tenga que ir como él (distancia < 1 s a su velocidad)
                ult = [m["v"] for m in sim.veh if m["carril"] == r.carril
                       and 0 < m["x"] - m["L"] - x_e < max(r.v * 1.0, 6.0)]
                sim.entra(r.Index, r.clase, x_e, r.carril, min([r.v] + ult),
                          t_entrada=(r.frame - f0) / fps)
            else:
                resto.append(r)
        espera = resto
        sim.paso()
        if True:
            registro += [(round(sim.t, 2), m["id"], m["x"], m["lat"], m["v"], m["clase"]) for m in sim.veh]

    # ---- comparación con la realidad
    reales = g[(g.frame >= f0) & (g.frame <= f1)]
    print(f"\n=== {cam} trozo {a.trozo} · {a.sentido} · {a.duracion:.0f} s desde {a.desde:.0f} s · "
          f"zona {x_ini:.0f}-{x_fin:.0f} m · {pob.nc} carriles")
    # tiempo dentro de la zona (de su entrada a su salida) para los que la cruzan entera
    t_real, t_sim, t_cv = [], [], []
    for vid, r in reales.groupby("id"):
        if vid not in sim.salidas:
            continue
        ent, sal_ = r.iloc[0], r.iloc[-1]
        if sal_.x < x_fin - 3:
            continue
        te = max((ent.frame - f0) / fps, 0.0)
        tr = (sal_.frame - f0) / fps - te
        ts = sim.salidas[vid][0] - te
        tc = (x_fin - ent.x) / max(ent.v, 0.5)          # si siguiera a su velocidad de entrada
        if tr > 0.5:
            t_real.append(tr); t_sim.append(ts); t_cv.append(tc)
    t_real, t_sim, t_cv = map(np.array, (t_real, t_sim, t_cv))
    if len(t_real):
        print(f"  tiempo en atravesar la zona (n={len(t_real)}): real mediana {np.median(t_real):.1f} s")
        print(f"    error por vehículo: conductor promedio {np.median(np.abs(t_sim - t_real)):.2f} s "
              f"(sesgo {np.median(t_sim - t_real):+.2f})  |  velocidad constante "
              f"{np.median(np.abs(t_cv - t_real)):.2f} s (sesgo {np.median(t_cv - t_real):+.2f})")
    # cambios de carril
    cr = [c for c in cambios_reales if c[0] in set(reales.id) and f0 <= c[1] <= f1]
    vkm_real = reales.groupby("id").x.agg(lambda s_: s_.max() - s_.min()).sum() / 1000
    print(f"  cambios de carril: real {len(cr)}  |  simulado {len(sim.cambios)}   "
          f"({len(cr) / max(vkm_real, 1e-6):.1f} vs {len(sim.cambios) / max(vkm_real, 1e-6):.1f} por veh·km)")
    print(f"  decisiones del modelo: {sim.vetos}")
    # carril de salida y velocidad por carril
    sal_real = reales.sort_values("frame").groupby("id").last()
    comunes = [v for v in sim.salidas if v in sal_real.index and sal_real.loc[v].x >= x_fin - 3]
    if comunes:
        acierto = np.mean([sim.salidas[v][1] == sal_real.loc[v].carril for v in comunes])
        print(f"  sale por el mismo carril que en la realidad: {100 * acierto:.0f}% (n={len(comunes)})")
    reg = pd.DataFrame(registro, columns=["t", "id", "x", "lat", "v", "clase"])
    vs = reg.assign(c=reg.lat.round().astype(int)).groupby("c").v.mean() * 3.6
    vr = reales.groupby("carril").v.mean() * 3.6
    print("  velocidad media por carril (real / simulado): " +
          "  ".join(f"C{c} {vr.get(c, np.nan):.0f}/{vs.get(c, np.nan):.0f}" for c in range(1, pob.nc + 1)))
    out = Path(a.camara) / f"sim_escena_{a.trozo}_{a.sentido}.csv"
    reg.to_csv(out, index=False)
    print(f"  trayectorias simuladas -> {out}")
    return dict(x_ini=x_ini, x_fin=x_fin, registro=reg, f0=f0, fps=fps, trozo=trozo,
                centros=centros[a.sentido])


# ---------------------------------------------------------------- modo sintético
DEMANDAS = {"ligero": 500, "medio": 1000, "denso": 1500, "muy denso": 1900}   # veh/h/carril


def sintetico(a, info, modelo):
    print(f"\n=== Autovía sintética {a.largo:.0f} m, {a.duracion:.0f} s, rampa {a.rampa:.0f} veh/h · conductores promedio de {a.camara} "
          f"(seguimiento: {a.idm_de or a.sentido}, velocidad deseada: {a.perfil_de or a.sentido})")
    filas = []
    for nombre, q in DEMANDAS.items():
        rng = np.random.default_rng(a.semilla)
        pob = Poblacion(info, a.sentido, modelo, rng, idm_de=a.idm_de, perfil_de=a.perfil_de)
        sim = Trafico(pob, a.largo)
        cuota_p = (pob.perfil[1] or {}).get("cuota", 0.08) if a.camiones is None else a.camiones
        uso = {g: (pob.perfil[g] or pob.perfil[0])["uso_carriles"] for g in (0, 1)}
        tasa = q * pob.nc / 3600          # llegadas por segundo
        cola, sig, n_id = [], rng.exponential(1 / tasa), 0
        x_rampa = a.largo * 0.6
        sig_r = rng.exponential(3600 / a.rampa) if a.rampa > 0 else np.inf
        cola_r = []
        detector = []                     # (t, id, v) al cruzar x = mitad
        mitad = a.largo / 2
        pasados = set()
        espacio_tiempo = []
        while sim.t < a.duracion:
            while sim.t >= sig:
                g = int(rng.uniform() < cuota_p)
                pr = np.array([uso[g].get(str(c), uso[g].get(c, 0)) for c in range(1, pob.nc + 1)], float)
                c = int(rng.choice(np.arange(1, pob.nc + 1), p=pr / pr.sum()))
                cola.append((g, c)); sig += rng.exponential(1 / tasa)
            resto = []
            for g, c in cola:   # entra si hay sitio en su carril
                ult = [m for m in sim.veh if m["carril"] == c and m["x"] < 60]
                hueco = min([m["x"] - m["L"] for m in ult], default=1e3)
                if hueco > 8:
                    v_in = min([m["v"] for m in ult], default=None)
                    clase = "camion" if g else "coche"
                    sim.entra(n_id, clase, 0.0, c, 0.0)
                    me = sim.veh[-1]
                    me["v"] = min(me["p"]["v0"], v_in if v_in is not None else me["p"]["v0"])
                    n_id += 1
                else:
                    resto.append((g, c))
            cola = resto
            # incorporación: entra al carril 1 en los 200 m de la rampa si cabe; cuanto más
            # espera, menos hueco exige (como la gente al final del carril de aceleración)
            while sim.t >= sig_r:
                cola_r.append(sim.t); sig_r += rng.exponential(3600 / a.rampa)
            resto_r = []
            for t_r in cola_r:
                esperado = sim.t - t_r
                exige = max(3.0 - 0.5 * esperado, 0.5)
                ok = False
                for xr in np.linspace(x_rampa, x_rampa + 200, 9):
                    v_ref = [m["v"] for m in sim.veh if m["carril"] == 1 and abs(m["x"] - xr) < 40]
                    v_in = min(v_ref) if v_ref else 22.0
                    lider_ok = all(not (m["carril"] == 1 and 0 <= m["x"] - m["L"] - xr < exige + 0.5 * v_in)
                                   for m in sim.veh)
                    det_ok = all(not (m["carril"] == 1 and 0 <= xr - 4.5 - m["x"] < exige + 0.3 * m["v"])
                                 for m in sim.veh)
                    if lider_ok and det_ok:
                        g_r = int(rng.uniform() < cuota_p)
                        sim.entra(n_id, "camion" if g_r else "coche", xr, 1, v_in); n_id += 1
                        ok = True
                        break
                if not ok:
                    resto_r.append(t_r)
            cola_r = resto_r
            antes = {m["id"]: m["x"] for m in sim.veh}
            sim.paso()
            for m in sim.veh:
                if m["id"] not in pasados and antes.get(m["id"], 0) < mitad <= m["x"]:
                    pasados.add(m["id"]); detector.append((sim.t, m["id"], m["v"]))
            if int(round(sim.t / DT)) % 10 == 0:
                espacio_tiempo += [(sim.t, m["x"], m["v"], m["carril"]) for m in sim.veh]
        det = pd.DataFrame(detector, columns=["t", "id", "v"])
        det = det[det.t > a.duracion * 0.3]       # descartar el arranque
        et = pd.DataFrame(espacio_tiempo, columns=["t", "x", "v", "carril"])
        et = et[et.t > a.duracion * 0.3]
        dur = a.duracion * 0.7
        flujo = len(det) / dur * 3600 / pob.nc
        v_media = 3.6 / np.mean(1 / np.maximum(det.v, 0.5)) if len(det) else np.nan
        dens = flujo / v_media if v_media > 0 else np.nan
        # atasco fantasma: tramos que se paran (<20 km/h) sin que haya nada que lo cause
        lento = et[et.v * 3.6 < 20]
        frac_lento = len(lento) / max(len(et), 1)
        # velocidad justo antes de la rampa (donde se forma el embudo)
        antes_r = et[(et.x > x_rampa - 400) & (et.x < x_rampa)]
        filas.append(dict(demanda=nombre, q_entrada=q, flujo=flujo, vel=v_media, densidad=dens,
                          vel_antes_rampa=antes_r.v.mean() * 3.6 if len(antes_r) else np.nan,
                          cola_entrada=len(cola), cola_rampa=len(cola_r), cambios_h_km=len(sim.cambios) / (a.duracion / 3600) / (a.largo / 1000),
                          frac_parado=frac_lento))
        et.to_csv(Path(a.salida_dir) / f"sim_sintetico_{nombre.replace(' ', '_')}_rampa{a.rampa:.0f}.csv",
                  index=False)
    r = pd.DataFrame(filas)
    print(r.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["escena", "sintetico"])
    ap.add_argument("camara", nargs="?", default=None, help="carpeta de la cámara (modo escena)")
    ap.add_argument("--comportamiento", default="salidas/lista/comportamiento.json")
    ap.add_argument("--sentido", default="alejan", choices=["alejan", "acercan"])
    ap.add_argument("--trozo", default="004")
    ap.add_argument("--desde", type=float, default=0.0, help="s desde el inicio del trozo")
    ap.add_argument("--duracion", type=float, default=120.0)
    ap.add_argument("--largo", type=float, default=2000.0)
    ap.add_argument("--camiones", type=float, default=None, help="cuota de pesados (sintético)")
    ap.add_argument("--rampa", type=float, default=0.0, help="veh/h que se incorporan por una rampa (sintético)")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--idm-de", default=None, help="sintético: sentido del que tomar el seguimiento")
    ap.add_argument("--perfil-de", default=None, help="sintético: sentido del que tomar la velocidad deseada")
    ap.add_argument("--salida-dir", default="salidas/lista")
    a = ap.parse_args()
    todo = json.load(open(a.comportamiento))
    modelo = joblib.load(Path(a.comportamiento).with_suffix(".decision.joblib"))
    if a.modo == "escena":
        info = todo[Path(a.camara).name]
        escena(a, info, modelo)
    else:
        a.camara = a.camara or "01_wqctLW0Hb_0"
        sintetico(a, todo[Path(a.camara).name], modelo)


if __name__ == "__main__":
    main()
