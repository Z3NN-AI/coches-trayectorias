"""Comprobaciones de robustez del experimento del circuito (circuito.py).

1. Sensibilidad a la frenada cómoda b (la calibrada quedó en el límite inferior, 0.5):
   ¿sigue saliendo el atasco fantasma con b razonables?
2. Velocidad de la onda de parar-arrancar en el suelo (literatura: unos -15 a -20 km/h,
   hacia atrás; Sugiyama 2008 ~ -20 km/h).
3. Comparación justa: perfectos con NUESTRAS distancias (T, s0) y nuestro pedal; solo
   cambia el ruido humano / la reacción.

Uso: .venv/bin/python circuito_pruebas.py
"""
import numpy as np

from circuito import Circuito, params

N, LARGO, V_OBJ, DUR = 22, 230.0, 30 / 3.6, 900.0
SEMILLAS = (0, 1, 2)


def correr(p, T, humanos, semilla, dur=DUR, onda=False):
    c = Circuito(N, LARGO, p, T, humanos, V_OBJ, np.random.default_rng(semilla))
    vs, centro = [], []
    while c.t < dur:
        c.paso()
        if c.t > dur / 2:
            vs.append(c.v.copy())
            if onda and int(round(c.t * 10)) % 5 == 0:
                lento = c.v * 3.6 < 5
                if lento.sum() >= 3:   # centro del grupo parado (media circular)
                    ang = c.x[lento] / LARGO * 2 * np.pi
                    centro.append((c.t, np.angle(np.exp(1j * ang).mean())))
    vs = np.array(vs) * 3.6
    r = dict(v=vs.mean(), parados=(vs < 5).mean() * 100, desv=vs.std(axis=1).mean())
    if onda and len(centro) > 50:
        t, a = np.array(centro).T
        a = np.unwrap(a)
        r["onda_kmh"] = np.polyfit(t, a / (2 * np.pi) * LARGO, 1)[0] * 3.6
    return r


def fila(nombre, rs):
    v = np.mean([r["v"] for r in rs]); par = np.mean([r["parados"] for r in rs])
    atasco = "ATASCO" if par > 5 else ("inestable" if np.mean([r["desv"] for r in rs]) > 1 else "fluido")
    print(f"  {nombre:46s} {v:5.1f} km/h  {par:4.0f}% parados  -> {atasco}")


def main():
    p, T = params("salidas/lista/comportamiento.json", "01_wqctLW0Hb_0")
    print(f"Pedal calibrado: a={p['a']:.2f} b={p['b']:.2f} s0={p['s0']:.2f}; T real mediana {T['p50']:.2f} s")
    print(f"{N} coches, {LARGO:.0f} m, {DUR:.0f} s, semillas {SEMILLAS}\n")

    print("1) SENSIBILIDAD a la frenada cómoda b (humanos, con ruido)")
    for b in (0.5, 1.0, 1.5, 2.0):
        fila(f"b = {b}", [correr(dict(p, b=b), T, True, s) for s in SEMILLAS])
    print("   ...y a la aceleración a (b = 1.5)")
    for a_ in (0.52, 1.0, 1.5, 2.0):
        fila(f"a = {a_}", [correr(dict(p, a=a_, b=1.5), T, True, s) for s in SEMILLAS])

    print("\n2) VELOCIDAD DE LA ONDA (humanos, pedal calibrado)")
    ondas = [correr(p, T, True, s, dur=1200, onda=True).get("onda_kmh") for s in SEMILLAS]
    print("  " + "  ".join(f"{o:+.1f} km/h" if o is not None else "sin onda" for o in ondas)
          + "   (literatura: -15 a -20 km/h; negativo = hacia atrás)")

    print("\n3) COMPARACIÓN JUSTA: mismas distancias (T, s0) que nuestros conductores")
    fila("perfectos: nuestro pedal, sin ruido", [correr(p, T, False, s) for s in SEMILLAS])
    fila("humanos:   nuestro pedal, con ruido", [correr(p, T, True, s) for s in SEMILLAS])
    for a_ in (1.0, 1.5, 2.0):
        q = dict(p, a=a_, b=a_)
        fila(f"perfectos: reacción viva a=b={a_}, sin ruido", [correr(q, T, False, s) for s in SEMILLAS])
        fila(f"humanos:   reacción viva a=b={a_}, con ruido", [correr(q, T, True, s) for s in SEMILLAS])


if __name__ == "__main__":
    main()
