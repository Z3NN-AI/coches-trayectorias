"""Paso 4: comportamiento del conductor promedio, a partir de los estados filtrados
(trayectorias.py --estados). Cada argumento es una cámara: una carpeta con
estados_*.json (+ config.txt con la mano de circulación) o un único estados.json.

Por cámara:
 1. Carriles (número automático) y cambios de carril.
 2. Régimen de tráfico por sentido (densidad, velocidad, flujo) y delta frente al
    sentido más fluido.
 3. Pedal: IDM calibrado sobre trayectorias con la primera mitad y evaluado en la
    segunda (posición del seguidor a h s, conocida la del líder).
Con todas las cámaras juntas:
 4. Volante: P(cambia al carril vecino en los próximos 3 s). Incentivo MOBIL solo
    vs. un clasificador con el contexto (huecos, velocidades de los carriles) vs. el
    mismo + deriva lateral. Validación por vehículo y dejando una cámara fuera.

Uso: .venv/bin/python comportamiento.py salidas/lista/01_* salidas/lista/05_* [...]
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

LARGO = {"coche": 4.5, "moto": 2.0, "bus": 12.0, "camion": 10.0}
VENTANA_CAMBIO = 3.0   # s: "va a cambiar de carril en los próximos 3 s"
PASO_MUESTRA = 0.5     # s entre muestras de un mismo vehículo
ALCANCE = 60.0         # m: vecinos que cuentan


# ---------------------------------------------------------------- carga
def cargar(camara):
    p = Path(camara)
    rutas = sorted(p.glob("estados_*.json")) if p.is_dir() else [p]
    cfg = {"mano": "izquierda"}
    if p.is_dir() and (p / "config.txt").exists():
        cfg.update(ast.literal_eval((p / "config.txt").read_text()))
    trozos, fps, fin = [], None, 0
    for k, r in enumerate(rutas):
        d = json.load(open(r))
        if not d["estados"]:
            continue
        fps = d["fps"]
        df = pd.DataFrame(d["estados"], columns=d["campos"])
        if "z_max" in cfg:   # solo el tramo recto cercano (más allá la carretera curva)
            df = df[(df.Z <= cfg["z_max"]) & (df.Z >= cfg.get("z_min", 0))]
        th = d["rumbo_carretera"]
        u = np.array([np.cos(th), np.sin(th)])
        n = np.array([-u[1], u[0]])   # izquierda de quien se aleja
        P = df[["X", "Z"]].values
        df["s"], df["d"] = P @ u, P @ n
        df["sentido"] = np.where(np.cos(df.rumbo - th) > 0, "alejan", "acercan")
        if cfg.get("enderezar"):
            # deriva lateral sistemática (curva suave o sesgo de perspectiva de la caja):
            # pendiente mediana d(s) de cada vehículo, por sentido, y se descuenta
            for sen in ("alejan", "acercan"):
                g = df[df.sentido == sen]
                pend = [np.polyfit(h.s, h.d, 1)[0] for _, h in g.groupby("id")
                        if len(h) > 15 and np.ptp(h.s) > 15]
                if len(pend) > 20:
                    pd_ = float(np.median(pend))
                    df.loc[g.index, "d"] = g.d - pd_ * (g.s - g.s.median())
        # x: avance en el sentido de marcha; izq: hacia la izquierda del conductor
        df["x"] = np.where(df.sentido == "alejan", df.s, -df.s)
        df["izq"] = np.where(df.sentido == "alejan", df.d, -df.d)
        # velocidad lateral hacia la izquierda del conductor, del rumbo filtrado (causal:
        # lo mismo que se puede calcular en directo)
        df["v_lat"] = df.v * np.sin(df.rumbo - th) * np.where(df.sentido == "alejan", 1, -1)
        df["frame"] += fin            # trozos seguidos en una sola línea de tiempo
        df["id"] = df["id"] + 100000 * k
        fin = df.frame.max() + 2 * int(fps)  # hueco entre trozos: nadie cruza el corte
        trozos.append(df)
    if not trozos:
        return None, None, cfg
    df = pd.concat(trozos, ignore_index=True).sort_values(["id", "frame"]).reset_index(drop=True)
    a = []
    for _, g in df.groupby("id", sort=False):
        if len(g) >= 15:
            a.append(savgol_filter(g.v.values, 15, 2, deriv=1, delta=1 / fps))
        else:
            a.append(np.full(len(g), np.nan))
    df["a"] = np.concatenate(a)
    df["largo"] = df.clase.map(LARGO).fillna(4.5)
    return df, fps, cfg


# ---------------------------------------------------------------- carriles
def carriles(df, carril_m, mano):
    """Centros de carril (en 'izq') por k-medias 1D; el número de carriles sale del
    ancho ocupado. Carril 1 = el del arcén (izquierda en UK, derecha en el continente);
    el número crece hacia la mediana (carril rápido)."""
    centros = {}
    for sen, g in df.groupby("sentido"):
        dm = g.groupby("id").izq.median().values
        if len(dm) < 20:
            continue
        lo, hi = np.quantile(dm, [0.03, 0.97])
        k = int(np.clip(round((hi - lo) / carril_m) + 1, 1, 6))
        c = np.linspace(lo, hi, k)
        for _ in range(50):
            lab = np.argmin(np.abs(dm[:, None] - c[None]), 1)
            c = np.array([dm[lab == j].mean() if (lab == j).any() else c[j] for j in range(k)])
        # descartar carriles casi vacíos (arcén, ruido) y fusionar centros muy juntos
        cnt = np.bincount(np.argmin(np.abs(dm[:, None] - c[None]), 1), minlength=k)
        c = np.sort(c[cnt >= max(5, 0.03 * len(dm))])
        c = np.array([x for i, x in enumerate(c) if i == 0 or x - c[i - 1] > 0.6 * carril_m])
        centros[sen] = c[::-1] if mano == "izquierda" else c
    return centros


def asignar_carril(df, centros, fps, histeresis=0.6, persistencia=0.5):
    """Carril por fotograma con histéresis; cambios = (id, frame del cruce, de, a)."""
    df["carril"] = 0
    cambios = []
    n_pers = int(persistencia * fps)
    for vid, g in df.groupby("id", sort=False):
        c = centros.get(g.sentido.iloc[0])
        if c is None:
            continue
        d = g.izq.values
        cand = np.argmin(np.abs(d[:, None] - c[None]), 1)
        lab = np.empty(len(d), int)
        cur = cand[0]
        for j in range(len(d)):
            if (cand[j] != cur and abs(d[j] - c[cur]) > abs(d[j] - c[cand[j]]) + histeresis
                    and j + n_pers <= len(d) and (cand[j:j + n_pers] == cand[j]).all()
                    and abs(int(cand[j]) - int(cur)) == 1):
                fr = (c[cur] + c[cand[j]]) / 2
                k0 = max(0, j - 2 * n_pers)
                kc = k0 + int(np.argmin(np.abs(d[k0:j + 1] - fr)))
                cambios.append((vid, int(g.frame.values[kc]), int(cur) + 1, int(cand[j]) + 1))
                lab[kc:j] = cand[j]
                cur = cand[j]
            lab[j] = cur
        df.loc[g.index, "carril"] = lab + 1
    return df, cambios


# ---------------------------------------------------------------- régimen
def regimen(df, fps, ventana=5.0):
    filas = []
    w = int(ventana * fps)
    for sen, g in df[df.carril > 0].groupby("sentido"):
        largo = (g.x.quantile(0.98) - g.x.quantile(0.02)) / 1000
        n_carr = g.carril.nunique()
        for f0 in range(int(g.frame.min()), int(g.frame.max()), w):
            h = g[(g.frame >= f0) & (g.frame < f0 + w)]
            if h.frame.nunique() < w // 2:
                continue
            k = h.groupby("frame").size().mean() / largo / n_carr
            v = 3.6 / np.mean(1 / np.maximum(h.v.values, 0.5))
            filas.append((sen, f0 / fps, k, v, k * v))
    return pd.DataFrame(filas, columns=["sentido", "t", "densidad", "vel", "flujo"])


# ---------------------------------------------------------------- pedal (IDM)
def parejas(df):
    out = []
    for (f, sen, c), g in df[df.carril > 0].groupby(["frame", "sentido", "carril"]):
        if len(g) < 2:
            continue
        g = g.sort_values("x")
        xs, vs, ids, ls = g.x.values, g.v.values, g.id.values, g.largo.values
        for i in range(len(g) - 1):
            out.append((f, ids[i], ids[i + 1], xs[i + 1] - xs[i] - ls[i + 1], vs[i + 1]))
    p = pd.DataFrame(out, columns=["frame", "id", "lider", "hueco", "v_lider"])
    return df.merge(p, on=["frame", "id"], how="left")


def idm_acc(v, vl, s, v0, T, a, b, s0, delta=4):
    s = np.maximum(s, 0.1)
    s_star = s0 + np.maximum(0, v * T + v * (v - vl) / (2 * np.sqrt(a * b)))
    return a * (1 - (v / v0) ** delta - (s_star / s) ** 2)


MANUAL = dict(v0=33.3, T=1.5, a=1.0, b=1.5, s0=2.0)  # Treiber, autovía
NOMBRES = ["v0", "T", "a", "b", "s0"]


def tramos(df, muestras, fps, h):
    pos = {(r.id, r.frame): (r.x, r.v) for r in df[["id", "frame", "x", "v"]].itertuples(index=False)}
    n = int(h * fps)
    X0, V0, LX, LV, FIN = [], [], [], [], []
    for r in muestras.itertuples(index=False):
        fin = pos.get((r.id, r.frame + n))
        lid = [pos.get((r.lider, r.frame + k)) for k in range(n + 1)]
        if fin is None or any(q is None for q in lid):
            continue
        x0, v0 = pos[(r.id, r.frame)]
        off = lid[0][0] - x0 - r.hueco
        X0.append(x0); V0.append(v0); FIN.append(fin[0])
        LX.append([q[0] - off for q in lid]); LV.append([q[1] for q in lid])
    return dict(x0=np.array(X0), v0=np.array(V0), lx=np.array(LX), lv=np.array(LV),
                fin=np.array(FIN), dt=1 / fps, n=n)


def simular(T_, p):
    x, v = T_["x0"].copy(), T_["v0"].copy()
    for k in range(T_["n"]):
        acc = np.clip(idm_acc(v, T_["lv"][:, k], T_["lx"][:, k] - x, **p), -9, 4)
        v = np.maximum(v + acc * T_["dt"], 0.0)
        x = x + v * T_["dt"]
    return x


def calibrar_idm(T_):
    """Ajuste sobre la trayectoria (la aceleración derivada es casi todo ruido)."""
    r = least_squares(lambda q: simular(T_, dict(zip(NOMBRES, q))) - T_["fin"],
                      [MANUAL[k] for k in NOMBRES], loss="soft_l1", f_scale=0.5,
                      diff_step=1e-3, bounds=([5, 0.3, 0.2, 0.5, 0.5], [45, 3.5, 4.0, 6.0, 6.0]))
    return dict(zip(NOMBRES, r.x))


def errores(T_, params):
    cv = np.abs(T_["x0"] + T_["v0"] * T_["n"] * T_["dt"] - T_["fin"])
    return np.column_stack([cv, np.abs(simular(T_, MANUAL) - T_["fin"]),
                            np.abs(simular(T_, params) - T_["fin"])])


def pedal(df, fps, entreno, rng):
    """IDM por sentido: calibra con la primera parte, evalúa en el resto."""
    corte = df.frame.min() + entreno * (df.frame.max() - df.frame.min())
    m = df.dropna(subset=["hueco"])
    m = m[(m.hueco > 1) & (m.hueco < 80) & (m.v > 1)]
    sub = lambda x: x.iloc[rng.permutation(len(x))[:4000]]
    m = m.assign(grupo=np.where(m.clase.isin(["camion", "bus"]), "pesado", "ligero"))
    # régimen local: densidad (veh/km/carril) del sentido en ese fotograma, media de 5 s
    dens = {}
    for sen, g in df[df.carril > 0].groupby("sentido"):
        largo = (g.x.quantile(0.98) - g.x.quantile(0.02)) / 1000
        k = g.groupby("frame").size() / largo / max(g.carril.nunique(), 1)
        k = k.reindex(range(int(k.index.min()), int(k.index.max()) + 1)).rolling(
            int(5 * fps), min_periods=1, center=True).mean()
        dens[sen] = k
    kd = np.array([dens[s].get(f, np.nan) for s, f in zip(m.sentido, m.frame)])
    m = m.assign(reg=np.where(kd >= 18, "denso", "libre"))
    params, res = {}, {}
    grupos = [(sen, None, g) for sen, g in m.groupby("sentido")] + \
             [(sen, gr, g) for (sen, gr), g in m.groupby(["sentido", "grupo"])] + \
             [(sen, "@" + r, g) for (sen, r), g in m.groupby(["sentido", "reg"])]
    for sen, gr, g in grupos:
        clave = sen if gr is None else (f"{sen}{gr}" if gr.startswith("@") else f"{sen}/{gr}")
        # horizonte: lo más largo (≤2 s) con suficientes parejas que sigan a la vista
        for h in (2.0, 1.5, 1.0):
            Ttr = tramos(df, sub(g[g.frame < corte]), fps, h)
            Tte = tramos(df, sub(g[g.frame >= corte]), fps, h)
            if len(Ttr["x0"]) >= (100 if gr is None else 60) and len(Tte["x0"]) >= (100 if gr is None else 60):
                break
        else:
            continue
        p = calibrar_idm(Ttr)
        e = errores(Tte, p)
        dv = np.abs(Tte["lv"][:, -1] - Tte["lv"][:, 0])
        q = dv > np.percentile(dv, 75)
        params[clave] = p
        firmado = Tte["fin"] - simular(Tte, p)   # >0: llegó más lejos de lo previsto
        res[clave] = dict(h=h, n=len(e), q10=np.percentile(firmado, 10), q90=np.percentile(firmado, 90), cv=np.median(e[:, 0]), manual=np.median(e[:, 1]),
                        calibrado=np.median(e[:, 2]), cv_p90=np.percentile(e[:, 0], 90),
                        calibrado_p90=np.percentile(e[:, 2], 90),
                        cv_cambia=np.median(e[q, 0]), calibrado_cambia=np.median(e[q, 2]))
    return params, res


def perfil_clases(df):
    """Para el simulador: velocidad libre (sin nadie delante a <60 m) y uso de carriles,
    por sentido y grupo (ligero/pesado)."""
    df = df.assign(grupo=np.where(df.clase.isin(["camion", "bus"]), "pesado", "ligero"))
    out = {}
    for (sen, gr), g in df[df.carril > 0].groupby(["sentido", "grupo"]):
        libre = g[(g.hueco.isna() | (g.hueco > 60)) & (g.v > 5)]
        vlib = libre.groupby("id").v.median().values * 3.6
        uso = g.groupby("carril").size()
        # distancia temporal en seguimiento estable (casi misma velocidad que el líder)
        est = g[g.hueco.notna() & (g.hueco > 0) & (g.hueco < 60) & (g.v > 3)
                & ((g.v - g.v_lider).abs() < 1.0)]
        t_eff = ((est.hueco - 2.5) / est.v).clip(0.2, 4.0)
        out[f"{sen}/{gr}"] = dict(
            T_estable={f"p{q}": float(np.percentile(t_eff, q)) for q in (5, 25, 50, 75, 95)}
            if len(t_eff) >= 50 else None,
            n_vehiculos=int(g.id.nunique()),
            v_libre_kmh={f"p{q}": float(np.percentile(vlib, q)) for q in (5, 25, 50, 75, 95)}
            if len(vlib) >= 10 else None,
            uso_carriles={int(k): float(v / uso.sum()) for k, v in uso.items()},
            cuota=float(g.id.nunique() / df[df.sentido == sen].id.nunique()))
    return out


# ---------------------------------------------------------------- volante (MOBIL + contexto)
def rasgos(ids, X, V, C, L, VL, cen, p, elegidos=None, cortesia=0.3):
    """Variables de la decisión de cambiar al carril vecino, para los vehículos de un
    sentido en un instante. Se usa igual al entrenar y en directo (trayectorias.py).
    X: avance en el sentido de marcha; C: carril (1 = arcén); VL: velocidad lateral
    hacia la izquierda del conductor; cen: centros de carril (coordenada 'izq')."""
    nc = len(cen)
    v_carril = {c: V[C == c].mean() for c in np.unique(C)}

    def entorno(c, x, i):
        m = (C == c) & (np.arange(len(X)) != i) & (np.abs(X - x) < ALCANCE)
        dl = np.where(m & (X > x))[0]; dt = np.where(m & (X <= x))[0]
        return (dl[np.argmin(X[dl])] if len(dl) else None,
                dt[np.argmax(X[dt])] if len(dt) else None)

    def acc(v, x, lider):
        if lider is None:
            return idm_acc(v, v, 1e3, **p), 1e3, 0.0
        return idm_acc(v, V[lider], X[lider] - x - L[lider], **p), X[lider] - x - L[lider], V[lider] - v

    filas = []
    for i in (range(len(X)) if elegidos is None else elegidos):
        c, x, v = int(C[i]), X[i], V[i]
        if c < 1 or c > nc:   # carril fuera de los centros conocidos
            continue
        dl, _ = entorno(c, x, i)
        a_yo, hueco_yo, dv_yo = acc(v, x, dl)
        for nuevo in (c - 1, c + 1):
            if nuevo < 1 or nuevo > nc:
                continue
            ndl, ndt = entorno(nuevo, x, i)
            a_nuevo, hueco_del, dv_del = acc(v, x, ndl)
            coste, hueco_det, dv_det = 0.0, 1e3, 0.0
            if ndt is not None:
                hueco_det = x - X[ndt] - L[i]
                dv_det = V[ndt] - v                      # >0: el de detrás viene más rápido
                a_t_ahora, _, _ = acc(V[ndt], X[ndt], ndl)
                coste = idm_acc(V[ndt], v, max(hueco_det, 0.1), **p) - a_t_ahora
            izq = np.sign(cen[nuevo - 1] - cen[c - 1])  # +1: el carril destino está a la izquierda
            filas.append(dict(
                i=i, de=c, a=nuevo, a_la_izquierda=bool(izq > 0), largo=float(L[i]),
                hacia_mediana=1.0 if nuevo > c else -1.0, v=v,
                ganancia=a_nuevo - a_yo, coste_detras=coste,
                incentivo=(a_nuevo - a_yo) + cortesia * coste,
                hueco_yo=min(hueco_yo, 100), dv_yo=dv_yo,
                hueco_del=min(hueco_del, 100), dv_del=dv_del,
                hueco_det=min(hueco_det, 100), dv_det=dv_det,
                dv_carriles=v_carril.get(nuevo, v) - v_carril.get(c, v),
                carril_extremo=float(c in (1, nc)),
                v_lat_hacia=VL[i] * izq))             # >0: ya se desplaza hacia allí
    return filas


def muestras_volante(df, cambios, fps, params, centros):
    """Muestras cada 0,5 s con la etiqueta 'cruza a ese carril en los próximos 3 s'."""
    futuros = {}
    for vid, f, de, a in cambios:
        futuros.setdefault(vid, []).append((f, a))
    paso, ventana = int(PASO_MUESTRA * fps), int(VENTANA_CAMBIO * fps)
    filas = []
    todos = df[df.carril > 0]
    for (f, sen), g in todos[todos.frame.isin(range(0, int(todos.frame.max()) + 1, paso))] \
            .groupby(["frame", "sentido"]):
        ids = g.id.values
        F = rasgos(ids, g.x.values, g.v.values, g.carril.values, g.largo.values,
                   g.v_lat.values, centros[sen], params.get(sen, MANUAL))
        for r in F:
            vid = ids[r.pop("i")]
            r.update(id=vid, frame=f, sentido=sen,
                     y=int(any(f < fc <= f + ventana and a == r["a"] for fc, a in futuros.get(vid, []))))
            filas.append(r)
    return pd.DataFrame(filas)


CONTEXTO = ["v", "ganancia", "coste_detras", "hueco_yo", "dv_yo", "hueco_del", "dv_del",
            "hueco_det", "dv_det", "dv_carriles", "hacia_mediana", "carril_extremo", "largo", "de"]


def evaluar_volante(mb):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    def cv_pred(cols, grupos, k=5):
        pred = np.zeros(len(mb))
        for tr, te in GroupKFold(n_splits=k).split(mb, mb.y, grupos):
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                 max_leaf_nodes=15, l2_regularization=1.0,
                                                 class_weight="balanced", random_state=0)
            clf.fit(mb.iloc[tr][cols], mb.y.iloc[tr])
            pred[te] = clf.predict_proba(mb.iloc[te][cols])[:, 1]
        return pred

    out = {}
    base = mb.y.mean()
    print(f"  {len(mb)} muestras, {mb.y.sum()} positivas (cambia en ≤{VENTANA_CAMBIO:.0f} s), "
          f"tasa base {100 * base:.2f}%")
    modelos = {"MOBIL (incentivo solo)": None,
               "contexto (por qué cambiaría)": CONTEXTO,
               "contexto + deriva lateral": CONTEXTO + ["v_lat_hacia"]}
    for nom, cols in modelos.items():
        p = mb.incentivo.values if cols is None else cv_pred(cols, mb.id)
        mb[f"p_{nom}"] = p
        auc, ap = roc_auc_score(mb.y, p), average_precision_score(mb.y, p)
        out[nom] = dict(auc=auc, ap=ap)
        print(f"    {nom:30s} AUC {auc:.2f}   precisión media {ap:.2f}  (azar = {base:.3f})")
    # generalización entre países: entrenar con unas cámaras, probar en otra
    if mb.camara.nunique() > 1:
        print("  Dejando una cámara fuera (entrena con las demás):")
        for cam in mb.camara.unique():
            tr, te = mb[mb.camara != cam], mb[mb.camara == cam]
            if te.y.sum() < 5:
                continue
            for nom, cols in [("contexto", CONTEXTO), ("contexto + deriva", CONTEXTO + ["v_lat_hacia"])]:
                clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                     max_leaf_nodes=15, l2_regularization=1.0,
                                                     class_weight="balanced", random_state=0)
                clf.fit(tr[cols], tr.y)
                p = clf.predict_proba(te[cols])[:, 1]
                print(f"    probar en {cam:16s} {nom:18s} AUC {roc_auc_score(te.y, p):.2f}  "
                      f"precisión media {average_precision_score(te.y, p):.2f} (azar {te.y.mean():.3f})")
    # importancia: qué mira el conductor (permutación sobre el modelo de contexto)
    from sklearn.inspection import permutation_importance
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         l2_regularization=1.0, class_weight="balanced",
                                         random_state=0).fit(mb[CONTEXTO], mb.y)
    imp = permutation_importance(clf, mb[CONTEXTO], mb.y, scoring="average_precision",
                                 n_repeats=3, random_state=0)
    orden = np.argsort(imp.importances_mean)[::-1]
    print("  Lo que más pesa en la decisión (modelo de contexto): " +
          ", ".join(f"{CONTEXTO[i]} ({imp.importances_mean[i]:.3f})" for i in orden[:6]))
    out["importancia"] = {CONTEXTO[i]: float(imp.importances_mean[i]) for i in orden}
    # modelo final (contexto + deriva) con todos los datos, para dibujar en directo
    final = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                           l2_regularization=1.0, class_weight="balanced",
                                           random_state=0).fit(mb[CONTEXTO + ["v_lat_hacia"]], mb.y)
    decision = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                              l2_regularization=1.0, class_weight="balanced",
                                              random_state=0).fit(mb[CONTEXTO], mb.y)
    return out, final, decision


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("camaras", nargs="+")
    ap.add_argument("--entreno", type=float, default=0.5, help="fracción del tiempo para calibrar")
    ap.add_argument("--salida", default="salidas/comportamiento.json")
    ap.add_argument("--excluir", nargs="*", default=[],
                    help="CAMARA:SENTIDO a ignorar (p. ej. 01_wqctLW0Hb_0:alejan, el atasco)")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    informe, volante = {}, []

    for cam in a.camaras:
        df, fps, cfg = cargar(cam)
        nom = Path(cam).name
        if df is None:
            print(f"\n##### {nom}: sin estados (¿calibración fallida?), se omite"); continue
        fuera = [e.split(":")[1] for e in a.excluir if e.split(":")[0] == nom]
        if fuera:
            print(f"\n  (se ignora el sentido {fuera} de {nom})")
            df = df[~df.sentido.isin(fuera)]
        print(f"\n##### {nom}  ({cfg.get('mano')}, {(df.frame.max() + 1) / fps / 60:.0f} min, "
              f"{df.id.nunique()} vehículos)")
        centros = carriles(df, cfg.get("carril", 3.65), cfg["mano"])
        df, cambios = asignar_carril(df, centros, fps)
        df = df[df.carril > 0]
        print("  CARRILES (1 = arcén):")
        for sen, c in centros.items():
            vel = df[df.sentido == sen].groupby("carril").v.median() * 3.6
            print(f"    {sen:8s} " + "  ".join(f"C{j + 1} {vel.get(j + 1, np.nan):4.0f} km/h"
                                             for j in range(len(c))))
        cam_df = pd.DataFrame(cambios, columns=["id", "frame", "de", "a"])
        cam_df["sentido"] = cam_df.id.map(df.groupby("id").sentido.first())
        horas = (df.frame.max() + 1) / fps / 3600
        print(f"  CAMBIOS DE CARRIL: {len(cam_df)} ({len(cam_df) / horas:.0f}/h)  " +
              "  ".join(f"{s}: " + ", ".join(f"{de}->{b} {n}" for (de, b), n in
                                             g.groupby(["de", "a"]).size().items())
                        for s, g in cam_df.groupby("sentido")))

        reg = regimen(df, fps)
        ref = reg.groupby("sentido").vel.median().max()
        print("  RÉGIMEN (ventanas de 5 s; mediana [p10-p90]):")
        for sen, g in reg.groupby("sentido"):
            q = lambda c: f"{g[c].median():5.0f} [{g[c].quantile(.1):4.0f}-{g[c].quantile(.9):4.0f}]"
            print(f"    {sen:8s} densidad {q('densidad')} veh/km/carril | vel {q('vel')} km/h | "
                  f"flujo {q('flujo')} veh/h/carril | delta vs fluido {g.vel.median() - ref:+4.0f} km/h")

        df = parejas(df)
        params, res = pedal(df, fps, a.entreno, rng)
        fmt = lambda p: " ".join(f"{k}={p[k]:.2f}" for k in NOMBRES)
        print(f"  PEDAL (IDM; manual: {fmt(MANUAL)})")
        for sen, p in params.items():
            r = res[sen]
            print(f"    {sen:15s} calibrado {fmt(p)}  (v0 {p['v0'] * 3.6:.0f} km/h)")
            print(f"             error a {r['h']:.1f} s (n={r['n']}): vel. constante {r['cv']:.2f} | "
                  f"IDM manual {r['manual']:.2f} | IDM calibrado {r['calibrado']:.2f} m"
                  f"   p90 {r['cv_p90']:.2f} -> {r['calibrado_p90']:.2f}"
                  f"   líder cambia de ritmo {r['cv_cambia']:.2f} -> {r['calibrado_cambia']:.2f}")

        if cfg.get("sin_volante"):   # posición lateral poco fiable: solo pedal y perfiles
            print("  VOLANTE: cámara excluida (sin_volante en config.txt)")
            mb = []
        else:
            mb = muestras_volante(df.drop(columns=["lider", "hueco", "v_lider"]), cambios, fps,
                                  params, centros)
        if len(mb):
            mb["camara"] = nom
            mb["id"] = mb.id.astype(str) + "@" + nom
            volante.append(mb)
        perfil = perfil_clases(df)
        print("  CLASES (velocidad libre mediana, uso de carriles 1..n):")
        for k, v_ in perfil.items():
            vl = f"{v_['v_libre_kmh']['p50']:.0f} km/h" if v_["v_libre_kmh"] else "  -   "
            print(f"    {k:15s} {v_['n_vehiculos']:5d} veh ({100 * v_['cuota']:.0f}%)  libre {vl}  carriles " +
                  " ".join(f"{100 * v_['uso_carriles'].get(c, 0):.0f}%" for c in range(1, len(centros[k.split('/')[0]]) + 1)))
        informe[nom] = dict(perfil=perfil, config=cfg, centros={k: list(map(float, v)) for k, v in centros.items()},
                            cambios=len(cambios), cambios_hora=len(cambios) / horas,
                            idm=params, error_idm=res,
                            regimen=reg.groupby("sentido")[["densidad", "vel", "flujo"]].median()
                            .to_dict("index"))

    if volante:
        mb = pd.concat(volante, ignore_index=True)
        print("\n##### VOLANTE: ¿cambia al carril vecino en los próximos 3 s?")
        informe["volante"], modelo, decision = evaluar_volante(mb)
        import joblib
        joblib.dump({"modelo": modelo, "columnas": CONTEXTO + ["v_lat_hacia"]},
                    Path(a.salida).with_suffix(".volante.joblib"))
        # sin deriva lateral: el "por qué" de la decisión, para el simulador
        joblib.dump({"modelo": decision, "columnas": CONTEXTO, "base": float(mb.y.mean())},
                    Path(a.salida).with_suffix(".decision.joblib"))
        mb.to_parquet(Path(a.salida).with_suffix(".volante.parquet")) if _parquet() else \
            mb.to_csv(Path(a.salida).with_suffix(".volante.csv"), index=False)
    json.dump(informe, open(a.salida, "w"), indent=1, default=float)
    print(f"\nGuardado en {a.salida}")


def _parquet():
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
