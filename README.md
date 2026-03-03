# Spiderman Web Shooter (OpenCV + MediaPipe)

Real-time webcam demo that tracks one hand and shoots a web projectile from your wrist when a Spiderman hand gesture is detected.

## Features

- **Hand tracking** with MediaPipe (wrist `0`, index tip `8`, middle `12`, ring `16`, pinky `20`).
- **Gesture controls**:
  - **Spiderman gesture** (index+pinky up, middle+ring down) → shoot web.
  - **Open palm** (all fingers up) → reset all effects.
- **Projectile simulation**:
  - Direction from wrist (`0`) to middle MCP (`9`) (configurable in code).
  - Constant-speed projectile motion.
  - Line strand from wrist to projectile.
  - Small fading projectile trail.
- **Impact animation**:
  - Smooth radial web expansion at edge impact.
  - Alpha-blended overlays for softer visuals.
- **Optional polish included**:
  - Wrist recoil while firing.
  - Configurable speed and expansion parameters.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python web_shooter.py
```

## Controls

- **ESC**: exit the app.
- **Spiderman gesture**: fire a web projectile.
- **Open palm**: clear projectile/impact and reset.

## Notes

- The script tries loading `assets/web_tip.png` with alpha if present.
- If the PNG is missing, it automatically falls back to a procedural RGBA sprite.
- For best gesture detection, keep your hand centered and well-lit.
