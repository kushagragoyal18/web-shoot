import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


# ----------------------------- Data Containers ----------------------------- #


@dataclass
class Projectile:
    x: float
    y: float
    dx: float
    dy: float
    speed: float
    active: bool = True
    impacted: bool = False
    impact_point: Optional[Tuple[int, int]] = None
    trail: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class ExpansionState:
    active: bool = False
    center: Optional[Tuple[int, int]] = None
    radius: float = 0.0
    max_radius: float = 180.0
    growth_rate: float = 7.0
    alpha: float = 0.8


# ----------------------------- Gesture Detection --------------------------- #


def _finger_is_up(landmarks, tip_id: int, pip_id: int) -> bool:
    return landmarks[tip_id].y < landmarks[pip_id].y


def _thumb_is_up(landmarks, handedness_label: str) -> bool:
    # For mirrored webcam input we still use a handedness-aware X-axis check.
    thumb_tip_x = landmarks[4].x
    thumb_ip_x = landmarks[3].x
    if handedness_label == "Right":
        return thumb_tip_x < thumb_ip_x
    return thumb_tip_x > thumb_ip_x


def detect_spiderman_gesture(landmarks, handedness_label: str) -> bool:
    index_up = _finger_is_up(landmarks, 8, 6)
    pinky_up = _finger_is_up(landmarks, 20, 18)
    middle_down = not _finger_is_up(landmarks, 12, 10)
    ring_down = not _finger_is_up(landmarks, 16, 14)
    thumb_any = True  # thumb can be either state for this gesture
    return index_up and pinky_up and middle_down and ring_down and thumb_any


def detect_open_palm(landmarks, handedness_label: str) -> bool:
    index_up = _finger_is_up(landmarks, 8, 6)
    middle_up = _finger_is_up(landmarks, 12, 10)
    ring_up = _finger_is_up(landmarks, 16, 14)
    pinky_up = _finger_is_up(landmarks, 20, 18)
    thumb_up = _thumb_is_up(landmarks, handedness_label)
    return index_up and middle_up and ring_up and pinky_up and thumb_up


# ----------------------------- Motion / Physics ---------------------------- #


def calculate_direction_vector(
    landmarks,
    frame_width: int,
    frame_height: int,
    use_middle_mcp: bool = True,
) -> Tuple[float, float]:
    wrist = landmarks[0]
    target = landmarks[9] if use_middle_mcp else landmarks[8]

    wx, wy = wrist.x * frame_width, wrist.y * frame_height
    tx, ty = target.x * frame_width, target.y * frame_height

    vx, vy = tx - wx, ty - wy
    length = math.hypot(vx, vy)
    if length < 1e-5:
        return 0.0, -1.0
    return vx / length, vy / length


def _is_outside_frame(x: float, y: float, w: int, h: int) -> bool:
    return x < 0 or y < 0 or x >= w or y >= h


def update_projectile(
    projectile: Projectile,
    frame_shape: Tuple[int, int, int],
    max_trail_points: int = 8,
) -> Projectile:
    if not projectile.active:
        return projectile

    frame_h, frame_w = frame_shape[:2]

    projectile.x += projectile.dx * projectile.speed
    projectile.y += projectile.dy * projectile.speed

    projectile.trail.append((int(projectile.x), int(projectile.y)))
    if len(projectile.trail) > max_trail_points:
        projectile.trail.pop(0)

    if _is_outside_frame(projectile.x, projectile.y, frame_w, frame_h):
        projectile.x = float(np.clip(projectile.x, 0, frame_w - 1))
        projectile.y = float(np.clip(projectile.y, 0, frame_h - 1))
        projectile.active = False
        projectile.impacted = True
        projectile.impact_point = (int(projectile.x), int(projectile.y))

    return projectile


# ----------------------------- Rendering Helpers --------------------------- #


def _blend_rgba_on_bgr(
    frame: np.ndarray,
    overlay_rgba: np.ndarray,
    center: Tuple[int, int],
    scale: float = 1.0,
) -> None:
    if scale <= 0:
        return

    h, w = overlay_rgba.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    sprite = cv2.resize(overlay_rgba, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    x, y = center
    x1 = x - new_w // 2
    y1 = y - new_h // 2
    x2 = x1 + new_w
    y2 = y1 + new_h

    frame_h, frame_w = frame.shape[:2]
    if x2 <= 0 or y2 <= 0 or x1 >= frame_w or y1 >= frame_h:
        return

    clip_x1, clip_y1 = max(0, x1), max(0, y1)
    clip_x2, clip_y2 = min(frame_w, x2), min(frame_h, y2)

    sprite_x1 = clip_x1 - x1
    sprite_y1 = clip_y1 - y1
    sprite_x2 = sprite_x1 + (clip_x2 - clip_x1)
    sprite_y2 = sprite_y1 + (clip_y2 - clip_y1)

    sprite_crop = sprite[sprite_y1:sprite_y2, sprite_x1:sprite_x2]
    alpha = (sprite_crop[:, :, 3:4] / 255.0).astype(np.float32)
    color = sprite_crop[:, :, :3].astype(np.float32)

    roi = frame[clip_y1:clip_y2, clip_x1:clip_x2].astype(np.float32)
    blended = (alpha * color) + ((1.0 - alpha) * roi)
    frame[clip_y1:clip_y2, clip_x1:clip_x2] = blended.astype(np.uint8)


def draw_projectile(
    frame: np.ndarray,
    wrist_px: Tuple[int, int],
    projectile: Projectile,
    web_tip_sprite_rgba: np.ndarray,
) -> None:
    if projectile.active:
        tip = (int(projectile.x), int(projectile.y))

        # Main web strand.
        cv2.line(frame, wrist_px, tip, (245, 245, 245), 2, cv2.LINE_AA)

        # Fading trail.
        for i in range(1, len(projectile.trail)):
            p0 = projectile.trail[i - 1]
            p1 = projectile.trail[i]
            alpha = i / max(1, len(projectile.trail) - 1)
            color = int(150 + 100 * alpha)
            thickness = 1 if i < len(projectile.trail) - 2 else 2
            cv2.line(frame, p0, p1, (color, color, color), thickness, cv2.LINE_AA)

        # Slight growth as it travels.
        travel_scale = 1.0 + min(0.8, len(projectile.trail) * 0.05)
        _blend_rgba_on_bgr(frame, web_tip_sprite_rgba, tip, scale=travel_scale)


def animate_web_expansion(
    frame: np.ndarray,
    expansion: ExpansionState,
    web_tip_sprite_rgba: np.ndarray,
) -> None:
    if not expansion.active or expansion.center is None:
        return

    overlay = frame.copy()
    cx, cy = expansion.center
    radius = int(expansion.radius)

    # Radial rings for web impact feel.
    for offset in range(0, 45, 15):
        ring_r = max(1, radius - offset)
        cv2.circle(overlay, (cx, cy), ring_r, (255, 255, 255), 1, cv2.LINE_AA)

    # Spokes.
    spokes = 8
    for i in range(spokes):
        angle = (2 * math.pi / spokes) * i
        ex = int(cx + radius * math.cos(angle))
        ey = int(cy + radius * math.sin(angle))
        cv2.line(overlay, (cx, cy), (ex, ey), (220, 220, 220), 1, cv2.LINE_AA)

    blend_alpha = max(0.0, min(1.0, expansion.alpha))
    cv2.addWeighted(overlay, blend_alpha, frame, 1 - blend_alpha, 0, dst=frame)

    # PNG sprite at center with growth.
    center_scale = 1.0 + (expansion.radius / max(1.0, expansion.max_radius)) * 2.0
    _blend_rgba_on_bgr(frame, web_tip_sprite_rgba, (cx, cy), scale=center_scale)

    expansion.radius += expansion.growth_rate
    expansion.alpha *= 0.985

    if expansion.radius >= expansion.max_radius:
        expansion.active = False


# ----------------------------- Asset Loading ------------------------------- #


def load_web_tip_sprite(path: Path) -> np.ndarray:
    sprite = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if sprite is None or sprite.shape[2] != 4:
        # Safe fallback: procedural white disc with alpha.
        fallback = np.zeros((64, 64, 4), dtype=np.uint8)
        cv2.circle(fallback, (32, 32), 26, (255, 255, 255, 220), -1, cv2.LINE_AA)
        cv2.circle(fallback, (32, 32), 12, (255, 255, 255, 255), -1, cv2.LINE_AA)
        return fallback
    return sprite


# ----------------------------- Main Loop ----------------------------------- #


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to access webcam. Make sure camera permissions are enabled.")

    assets_dir = Path(__file__).parent / "assets"
    web_tip_sprite = load_web_tip_sprite(assets_dir / "web_tip.png")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    projectile: Optional[Projectile] = None
    expansion = ExpansionState()

    shoot_cooldown_frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            wrist_px: Optional[Tuple[int, int]] = None

            if result.multi_hand_landmarks and result.multi_handedness:
                hand_landmarks = result.multi_hand_landmarks[0].landmark
                handedness = result.multi_handedness[0].classification[0].label

                wrist = hand_landmarks[0]
                wrist_px = (int(wrist.x * frame_w), int(wrist.y * frame_h))

                spiderman = detect_spiderman_gesture(hand_landmarks, handedness)
                open_palm = detect_open_palm(hand_landmarks, handedness)

                if open_palm:
                    projectile = None
                    expansion = ExpansionState()
                    shoot_cooldown_frames = 0
                elif spiderman and projectile is None and not expansion.active and shoot_cooldown_frames == 0:
                    dx, dy = calculate_direction_vector(
                        hand_landmarks,
                        frame_width=frame_w,
                        frame_height=frame_h,
                        use_middle_mcp=True,
                    )
                    projectile = Projectile(
                        x=float(wrist_px[0]),
                        y=float(wrist_px[1]),
                        dx=dx,
                        dy=dy,
                        speed=20.0,
                    )
                    shoot_cooldown_frames = 6

                # Recoil effect while active.
                if projectile and projectile.active:
                    recoil_x = int(wrist_px[0] - projectile.dx * 6)
                    recoil_y = int(wrist_px[1] - projectile.dy * 6)
                    cv2.circle(frame, (recoil_x, recoil_y), 8, (235, 235, 235), -1, cv2.LINE_AA)

            if shoot_cooldown_frames > 0:
                shoot_cooldown_frames -= 1

            if projectile is not None:
                projectile = update_projectile(projectile, frame.shape)
                if wrist_px is not None:
                    draw_projectile(frame, wrist_px, projectile, web_tip_sprite)

                if projectile.impacted and projectile.impact_point and not expansion.active:
                    expansion = ExpansionState(
                        active=True,
                        center=projectile.impact_point,
                        radius=10.0,
                        max_radius=180.0,
                        growth_rate=7.5,
                        alpha=0.75,
                    )
                    projectile = None

            animate_web_expansion(frame, expansion, web_tip_sprite)

            cv2.putText(
                frame,
                "Gesture: Spiderman (shoot) | Open palm (reset) | ESC to exit",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Spiderman Web Shooter", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
