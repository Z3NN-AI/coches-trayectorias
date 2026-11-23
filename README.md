# Traffic detection and modelling — the Average Real Driver (ARD)

From fixed-camera motorway video to a simulator populated by **average real drivers**:
not the ideal driver of textbook models, but the imperfect, varied one you meet on the road.

```
video ─► detectar.py ─► tracks ─► calibrar.py ─► ground plane (metres)
                                └─► trayectorias.py ─► states (EKF) + annotated video
states ─► comportamiento.py ─► ARD: car-following + learned lane-change model + driver profiles
ARD ─► simulador.py / video_sintetico.py ─► synthetic motorway, scenarios
ARD ─► circuito.py / circuito_tres.py   ─► phantom-jam ring experiments
```

## 1. Detection and trajectories

- **`detectar.py`**: YOLOv8-seg plus ByteTrack. For each vehicle it stores the box, the class and a ground-contact point taken from the segmentation mask.
- **`calibrar.py`**: automatic ground-plane calibration, with no hand-picked points.
  - Horizon: the one that makes each vehicle move at constant speed.
  - Vanishing point: where the vehicle paths cross.
  - Distance scale: the cycle of the dashed lane lines.
  - Lateral scale: the lane width.
  - Manual fall-back when the automatic method fails.
- **`trayectorias.py`**: an extended Kalman filter (CTRV) per vehicle on the ground plane.
  - Occlusion memory.
  - Physical limits by mass and class.
  - A 2 s prediction cone.
  - Per-carriageway panels: fluidity (FREE FLOW / DENSE / CONGESTED and a bottleneck forecast), vehicles passed by class, average speed and collision-risk indicators (rear-end TTC, cut-in risk).
  - A plausibility check, e.g. a truck faster than 90 km/h is flagged.
- **`lote.py`**: batch processing of a playlist, in resumable 5-minute chunks.

## 2. The Average Real Driver (ARD)

A hybrid agent built in **`comportamiento.py`**, `conductor.py` and `simulador.py`. It has three parts.

| Part | What it is |
|---|---|
| Pedal | IDM car-following whose parameters are **fitted to real trajectories** (least squares). |
| Steering | **Learned** lane-change probability: gradient-boosted trees. Inputs: gaps, relative speeds, lane, vehicle length and the MOBIL incentive. Validated with GroupKFold by vehicle and leave-one-camera-out. |
| Character | Each driver is drawn from distributions: desired speed (real per-lane distributions), headway (real), acceleration, comfortable braking, reaction time, lane-change eagerness, a slowly drifting speed "mood", pedal noise, and keep-right compliance (80 % comply, 20 % lane hoggers). |

Traffic rules are written explicitly and marked as rules, not learned behaviour:
- no undertaking above 60 km/h;
- trucks return to lane 1 after overtaking;
- a lane change must not force the follower to brake hard;
- UK: heavy goods vehicles may not use lane 3.

Lanes are numbered from the nearside in every country (lane 1 = left in the UK, right on the continent).

## 3. Experiments

- **Real vs simulated** (`video_simulacion.py`): vehicles enter the scene as they did in reality, and from then on the ARD drives them. On a free-flowing US camera, the lane speeds match within about 1–2 km/h.
- **Synthetic motorway** (`video_sintetico.py`): 6 km, 3 lanes, only ARD agents. A dense scenario with a truck "elephant race", and a light scenario.
- **Phantom traffic jam ring** (`circuito_tres.py`). 22 cars on a 230 m ring, compared side by side:

| | Speed | Flow | Result |
|---|---|---|---|
| Textbook IDM (Treiber 2000) | 9.4 km/h | 901 veh/h | smooth |
| **ARD** | ~7 km/h | ~700 veh/h | **stop-and-go wave emerges**, as in Sugiyama et al. 2008 |
| ARD + 1 damping robot car | ~9.9 km/h | ~940 veh/h | **jam dissolved**, as in Stern et al. 2018 (5 of 6 seeds) |

The jam was never programmed: it emerges from the measured spacing and the per-driver reaction. With 11 cars (half the density) none of the three jams.

## Honest limitations

- **Short view.** Each camera covers about 55 m (2–3 s per vehicle), so per-driver character (lane hogging, keep-right compliance, speed mood) cannot be measured. Those distributions are expert assumptions.
- **Little data.** Two and a half usable cameras. The learned lane-change model saw only 186 changes and generalises poorly across cameras (AUC around 0.6).
- **Weakly identified parameters.** The calibrated comfortable braking sits at its lower bound.
- **Ring mismatches.** The simulated jam wave travels at about −10 km/h; the literature reports −15 to −20 km/h. The robot controller's averaging window (120 s) was chosen by testing several values.
- **Rough edges.** The on-ramp merge model is unrealistic. On the synthetic motorway, lane speeds are about 10 km/h below the real ones.

## Usage

```bash
uv venv && uv pip install -r requirements.txt
python detectar.py video.mp4 out/pistas.json          # YOLO weights: env YOLO_MODELO or modelo_local.txt
python calibrar.py out/pistas.json out/calibracion.json
python trayectorias.py out/pistas.json --calib out/calibracion.json --estados out/estados.json
python comportamiento.py out/ --salida out/comportamiento.json
python circuito_tres.py --duracion 1200               # ring experiment video
python circuito_pruebas.py                            # robustness checks
```

Code comments are in Spanish; outputs and video overlays are in English.

## Credits

- Test videos: public YouTube traffic-camera footage. The videos are not included in this repository.
- Detection: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and ByteTrack.
- Models and experiments: IDM (Treiber, Hennecke and Helbing 2000), MOBIL (Kesting, Treiber and Helbing 2007), Sugiyama et al. 2008, Stern et al. 2018.

## License

MIT. Note: Ultralytics YOLOv8 is AGPL-3.0, which applies to its use.
