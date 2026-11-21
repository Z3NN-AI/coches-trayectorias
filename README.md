# coches-trayectorias

Experimento por curiosidad: a partir de un vídeo de tráfico grabado con cámara fija,
detectar los vehículos, medir su velocidad real en km/h y dibujar delante de cada uno
su **trayectoria prevista** con un **cono de posibilidades** que respeta la física.
Incluye una **vista cenital** reconstruida (la carretera vista desde arriba, en metros).

La idea de partida: un vehículo es una masa grande y no puede cambiar de trayectoria
a capricho. Cuanto más rápido va, menos puede girar (la aceleración lateral `a = v·ω`
está limitada), así que el cono se estrecha en autovía y se abre a baja velocidad.

## Cómo funciona

```
vídeo ─► detectar.py ─► pistas.json ─► calibrar.py ─► calibracion.json
                                   └──────────────► trayectorias.py ─► vídeo anotado + métricas
```

1. **`detectar.py`** — YOLOv8-seg + ByteTrack. Guarda por fotograma la caja, la clase
   y un punto de apoyo en el suelo sacado de la máscara de segmentación.
2. **`calibrar.py`** — calibración automática del plano de la carretera, sin marcar
   puntos a mano:
   - horizonte: el que hace que cada vehículo vaya a velocidad constante;
   - punto de fuga: donde se cortan las rectas que recorren los vehículos;
   - escala de distancia: el ciclo de las **líneas discontinuas** detectadas automáticamente;
   - escala lateral: la separación entre líneas vecinas (ancho de carril).
   Si no encuentra líneas, recurre al ancho típico de los coches.
3. **`trayectorias.py`** — un filtro de Kalman extendido por vehículo sobre el plano
   del suelo (modelo CTRV: velocidad y giro constantes), con:
   - **memoria**: un vehículo tapado sigue moviéndose con su física como "fantasma"
     (3 s si hay otro vehículo delante tapándolo, 0,5 s si simplemente se pierde) y,
     si el seguidor le cambia el número al reaparecer, hereda su pasado;
   - **medidas robustas**: se descartan las medidas de vehículos tapados, los saltos
     bruscos de tamaño de caja y los saltos aislados; solo un desvío persistente
     cuenta como maniobra real;
   - **masa**: camiones y buses giran menos y su límite es el de vuelco (~0,35 g),
     no el de adherencia (~0,8 g);
   - **interacción** (IDM): si alcanzas al de delante, frenas. La distancia cómoda se
     estima para cada conductor a partir de cómo va ahora;
   - **evaluación integrada**: compara cada predicción con la posición real 1 y 2 s
     después, el giro aparente (en recta debería ser 0) y el error al reaparecer un
     vehículo oculto.

## Uso

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# un minuto de vídeo de prueba (cámara fija sobre una autovía británica)
yt-dlp -f "bv*[height<=720]" --download-sections "*17:00-18:00" \
       -o videos/trafico.mp4 "https://youtu.be/wqctLW0Hb_0"

.venv/bin/python detectar.py videos/trafico.mp4 salidas/pistas.json
.venv/bin/python calibrar.py salidas/pistas.json --ciclo 9 --carril 3.65
.venv/bin/python trayectorias.py salidas/pistas.json salidas/trayectorias.mp4
```

- `--ciclo` y `--carril` dependen de la normativa del país:
  Reino Unido, autovía: `--ciclo 9 --carril 3.65` (línea de 2 m + hueco de 7 m);
  España, autovía: `--ciclo 12.5 --carril 3.5`.
- `YOLO_MODELO=/ruta/a/pesos.pt` para usar otros pesos (por defecto se descarga `yolov8m-seg.pt`).
- Opciones útiles de `trayectorias.py`: `--sin-video` (solo métricas), `--sin-prior`
  (rotondas o cruces: no asume una dirección de carretera), `--modelo cv|ctrv|ctra`,
  `--sin-idm`, `--basico` (sin memoria ni filtros robustos, para comparar).

## Resultados en el vídeo de prueba (2 minutos, ~500 vehículos)

| | |
|---|---|
| Velocidad vs. medida independiente por líneas del suelo | error mediano ±1–2 km/h |
| Error de posición predicha a 1 s / 2 s | ~0,4 m / ~0,75 m |
| Cono de 2 s: posiciones reales que caen dentro | ~95 % |
| Vehículo oculto 1–3 s: error al reaparecer (con física / si se quedara quieto) | ~1,5 m / ~12 m |

Cosas que se probaron y **no** funcionaron mejor (y por eso no están por defecto):
añadir la aceleración al estado (CTRA), el punto más bajo de la máscara como punto
de apoyo, y la corrección geométrica del punto de contacto en camiones.

## Limitaciones

- Pensado para **cámara fija**. La calibración automática supone una carretera recta y
  plana con líneas discontinuas.
- Los vehículos se tratan como puntos sobre el suelo: un camión grande visto muy en
  diagonal todavía genera algo de giro aparente, y un vehículo en el aire (accidente)
  rompe la suposición de suelo plano.
- El cono es físico más estadístico, no aprende el comportamiento de los conductores.
  El siguiente paso natural sería un IMM (varios modelos de maniobra unidos por una
  cadena de Markov) entrenado con datos reales, por ejemplo highD o rounD.

## Créditos

Vídeo de prueba: "Road traffic video for object recognition", de Andrey Nikishaev, en
YouTube. No se incluye en el repositorio. Detección y seguimiento con
[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) y ByteTrack.

## Licencia

MIT. Ojo: Ultralytics YOLOv8 tiene licencia AGPL-3.0, que aplica a su uso.
