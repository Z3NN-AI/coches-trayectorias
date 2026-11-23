"""Who is on the road? Distribution of the drivers seen passing the cameras (real data)
and of the drivers the Average Real Driver (ARD) simulator creates.

Uso: .venv/bin/python distribucion_conductores.py
-> salidas/lista/distribucion_conductores.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from comportamiento import asignar_carril, cargar
from simulador import Poblacion

L = Path("salidas/lista")
CAMARAS = {"01_wqctLW0Hb_0": ("UK", ["acercan"]),     # sin el sentido del atasco
           "05_nt3D26lrkho": ("France", ["acercan", "alejan"]),
           "03_TW3EH4cnFZo": ("USA", ["acercan", "alejan"])}
AZUL, NARANJA, VERDE = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA2, REJILLA = "#1f1f1e", "#6b6a63", "#e4e3dd"


def datos_reales():
    info = json.load(open(L / "comportamiento.json"))
    filas = []
    for cam, (pais, sentidos) in CAMARAS.items():
        df, fps, _ = cargar(L / cam)
        c = {k: np.array(v) for k, v in info[cam]["centros"].items()}
        df, _ = asignar_carril(df, c, fps)
        df = df[df.sentido.isin(sentidos)]
        from comportamiento import parejas
        df = parejas(df)
        for vid, g in df.groupby("id"):
            if len(g) < fps:
                continue
            libre = g[(g.hueco.isna() | (g.hueco > 60))]
            sigue = g[(g.hueco > 0) & (g.hueco < 60) & ((g.v - g.v_lider).abs() < 1) & (g.v > 3)]
            filas.append(dict(pais=pais, clase=g.clase.iloc[0],
                              v_libre=libre.v.median() * 3.6 if len(libre) > fps / 2 else np.nan,
                              T=((sigue.hueco - 2.5) / sigue.v).median() if len(sigue) > fps / 2 else np.nan,
                              carril=int(g.carril.mode().iloc[0])))
    import pandas as pd
    return pd.DataFrame(filas)


def estilo(ax, titulo, xlabel):
    ax.set_title(titulo, loc="left", fontsize=12, color=TINTA, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=TINTA2, fontsize=10)
    ax.tick_params(colors=TINTA2, labelsize=9, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(REJILLA)
    ax.set_yticks([])
    ax.grid(axis="x", color=REJILLA, lw=0.8)
    ax.set_axisbelow(True)


def campana(ax, x, color, nombre, bins, dec=0):
    x = x[np.isfinite(x)]
    ax.hist(x, bins=bins, color=color, alpha=0.55, density=True, edgecolor="white", linewidth=1)
    m, s = x.mean(), x.std()
    xs = np.linspace(bins[0], bins[-1], 300)
    ax.plot(xs, np.exp(-0.5 * ((xs - m) / s) ** 2) / (s * np.sqrt(2 * np.pi)), color=color, lw=2)
    return f"{nombre}: mean {m:.{dec}f}, sd {s:.{dec}f} (n={len(x)})"


def main():
    d = datos_reales()
    coches = d[d.clase == "coche"]; pes = d[d.clase.isin(["camion", "bus"])]
    fig, axs = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Who is on the road?  Drivers seen by the cameras (real) and drivers created by the "
                 "Average Real Driver simulator (ARD)", x=0.01, ha="left", fontsize=15, color=TINTA,
                 fontweight="bold")

    # 1. velocidad deseada: coches vs camiones
    ax = axs[0, 0]; bins = np.arange(50, 151, 4)
    t1 = campana(ax, coches.v_libre.values, AZUL, "cars", bins)
    t2 = campana(ax, pes.v_libre.values, NARANJA, "trucks & buses", bins)
    estilo(ax, "Preferred speed (free driving, no one ahead)", "km/h")
    ax.legend([t1, t2], loc="upper left", frameon=False, fontsize=9, labelcolor=TINTA)

    # 2. coches por país
    ax = axs[0, 1]
    txt = []
    for pais, col in zip(("UK", "France", "USA"), (AZUL, NARANJA, VERDE)):
        x = coches[coches.pais == pais].v_libre.values
        x = x[np.isfinite(x)]
        if len(x) > 20:
            m, s = x.mean(), x.std()
            xs = np.linspace(50, 150, 300)
            ax.plot(xs, np.exp(-0.5 * ((xs - m) / s) ** 2) / (s * np.sqrt(2 * np.pi)), color=col, lw=2)
            txt.append(f"{pais}: mean {m:.0f}, sd {s:.0f} (n={len(x)})")
    estilo(ax, "Cars' preferred speed by country", "km/h")
    ax.legend(txt, loc="upper left", frameon=False, fontsize=9, labelcolor=TINTA)

    # 3. coches por carril
    ax = axs[0, 2]
    txt = []
    for c, col in zip((1, 2, 3), (AZUL, NARANJA, VERDE)):
        x = coches[coches.carril == c].v_libre.values; x = x[np.isfinite(x)]
        if len(x) > 20:
            ax.hist(x, bins=np.arange(50, 151, 4), color=col, alpha=0.45, density=True, edgecolor="white")
            txt.append(f"lane {c}: median {np.median(x):.0f} km/h")
    estilo(ax, "Cars' preferred speed by lane (lane 1 = nearside)", "km/h")
    ax.legend(txt, loc="upper left", frameon=False, fontsize=9, labelcolor=TINTA)

    # 4. distancia temporal
    ax = axs[1, 0]; bins = np.arange(0, 3.01, 0.1)
    t1 = campana(ax, coches["T"].values, AZUL, "cars", bins, 2)
    t2 = campana(ax, pes["T"].values, NARANJA, "trucks & buses", bins, 2)
    estilo(ax, "Time gap kept when following (real)", "seconds behind the vehicle ahead")
    ax.legend([t1, t2], loc="upper right", frameon=False, fontsize=9, labelcolor=TINTA)

    # 5-6. ARD: caracteres sorteados por el simulador
    info = json.load(open(L / "comportamiento.json"))["01_wqctLW0Hb_0"]
    import joblib
    mod = joblib.load(L / "comportamiento.decision.joblib")
    pob = Poblacion(info, "acercan", mod, np.random.default_rng(0), idm_de="acercan", perfil_de="acercan")
    P = [pob.personalidad(0, carril=int(np.random.default_rng(i).choice([1, 2, 3], p=[.35, .33, .32])))
         for i in range(3000)]
    ax = axs[1, 1]
    campana(ax, np.array([p["a"] for p in P]), AZUL, "acceleration", np.arange(0, 3.01, 0.1))
    campana(ax, np.array([p["b"] for p in P]), NARANJA, "comfortable braking", np.arange(0, 3.01, 0.1))
    estilo(ax, "ARD character: how lively each driver is", "m/s²")
    ax.legend(["acceleration (a)", "comfortable braking (b)"], loc="upper right", frameon=False, fontsize=9,
              labelcolor=TINTA)

    ax = axs[1, 2]
    v0 = np.array([p["v0_base"] * 3.6 for p in P]); cumple = np.array([p["vuelve"] for p in P])
    bins = np.arange(50, 151, 4)
    ax.hist([v0[cumple], v0[~cumple]], bins=bins, stacked=True, color=[AZUL, NARANJA], alpha=0.75,
            edgecolor="white", linewidth=1)
    estilo(ax, "ARD character: preferred speed and lane discipline", "km/h")
    ax.legend([f"keeps right when possible ({cumple.mean() * 100:.0f}%)",
               f"lane hogger ({(~cumple).mean() * 100:.0f}%)"], loc="upper left", frameon=False, fontsize=9,
              labelcolor=TINTA)

    fig.text(0.01, 0.01, "Real: cameras in the UK (free-flowing side only), France and USA; each vehicle "
             "counted once (median over the ~55 m it is visible).  ARD: 3,000 drivers drawn by the "
             "simulator. Curves = normal fit.", fontsize=9, color=TINTA2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out = L / "distribucion_conductores.png"
    fig.savefig(out, dpi=130)
    print(f"-> {out}")
    print(d.groupby(["pais", "clase"]).v_libre.describe()[["count", "mean", "std"]].round(1))


if __name__ == "__main__":
    main()
