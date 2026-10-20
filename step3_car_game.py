"""
AeroDrive 2D - Top-Down Hand-Gesture & Optical-Flow Highway Traffic Racer
Part of AeroTouch 3D (Unit 5: Pose Estimation, Pinch, Fist Brake & Lucas-Kanade Optical Flow)

Architecture (Zero-Lag 60 FPS Dual-Window Design):
  1. Dedicated Main Game Window (`pygame`, locked 60 FPS):
     - Crisp top-down 2D 4-lane highway graphics with textured asphalt, shoulders,
       roadside trees & streetlamps, tire brake marks, exhaust smoke, and HUD gauges.
     - Custom-designed top-down Player Supercar (headlight beams, glowing red brake
       lights, blinking turn signals, speed tilt, and supersonic Nitro flames).
     - Unpredictable Multi-Class Traffic AI (Sedans, Yellow Taxis, Sport Coupes,
       SUVs, and Heavy Semi-Trucks) that brake for slower cars and signal lane
       changes in advance using flashing AMBER TURN INDICATORS!
  2. Asynchronous Background Vision Thread + Companion Camera Monitor (`cv2`):
     - Runs MediaPipe Hand Pose + Scale-Invariant Fist/Pinch Classification +
       Pyramidal Lucas-Kanade Optical Flow (`cv2.calcOpticalFlowPyrLK`) off-thread
       so camera capture NEVER slows down the 60 FPS game loop!

Hand Gesture Controls:
  * MOVE HAND LEFT / RIGHT : Smoothly steer your Supercar across all 4 lanes
  * OPEN HAND              : Smooth highway cruise throttle
  * CLOSE HAND (FIST)      : BRAKE HARD (lights up red brake lamps + tire marks!)
  * PINCH (Thumb + Index)  : NITRO TURBO BOOST (max speed + 2x score multiplier!)
  * (Game Over) PINCH      : Restart race hands-free (or press [R])

Keyboard & Menu Controls (Always Active Alongside Gestures):
  [1 / 2 / 3] Easy / Medium / Hard   |  [M] Open Mode Selection Menu
  [Left/Right or A/D] Steer          |  [Down / S / Space] Brake  |  [Up / W / N] Nitro
  [C] Toggle Camera Monitor          |  [R] Restart Race          |  [Q / ESC] Quit
"""

from collections import deque
from dataclasses import dataclass
import math
import os
import random
import threading
import time
from typing import Deque, Dict, List, Optional, Tuple

# Position the main 2D game window neatly on the left so the companion camera
# monitor window sits right beside it on screen without overlapping!
os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "50,45")

import cv2
import numpy as np
import pygame

from step1_hand_tracking import HandGestureData, HandPoseTracker, open_webcam

# ---------------------------------------------------------------------------
# Display & Highway Layout Constants
# ---------------------------------------------------------------------------
GAME_W = 860
GAME_H = 760
TARGET_FPS = 60

CAM_CAPTURE_W = 480
CAM_CAPTURE_H = 360

ROAD_LEFT = 150
ROAD_RIGHT = 710
ROAD_W = ROAD_RIGHT - ROAD_LEFT  # 560 px across 4 wide highway lanes
NUM_LANES = 4
LANE_W = ROAD_W // NUM_LANES     # 140 px per lane
LANE_CENTERS = tuple(ROAD_LEFT + LANE_W * i + LANE_W // 2 for i in range(NUM_LANES))

PLAYER_Y = GAME_H - 145
MIN_SPEED_KMH = 75.0
KMH_TO_PX_PER_SEC = 3.15

LK_LANDMARK_IDS = (0, 4, 5, 8, 9, 12, 13, 16, 17, 20)
CAM_WINDOW_TITLE = "AeroDrive - Gesture & Optical Flow Monitor"


@dataclass(frozen=True)
class DifficultyConfig:
    """Per-mode traffic density, indicator/lane-switch rules & speed scaling."""

    key: str
    hotkey_label: str
    theme_rgb: Tuple[int, int, int]
    summary_line1: str
    summary_line2: str
    base_spawn_s: float
    min_spawn_s: float
    max_blocked_lanes: int
    spawn_clearance_y: float
    use_indicators: bool
    aggression_mult: float
    signal_lead_range: Tuple[float, float]
    merge_speed_mult: float
    base_cruise_kmh: float
    max_score_speed_bonus: float
    nitro_kmh: float
    score_mult: float


DIFFICULTY_ORDER = ("EASY", "MEDIUM", "HARD")

DIFFICULTY_CONFIGS: Dict[str, DifficultyConfig] = {
    "EASY": DifficultyConfig(
        key="EASY",
        hotkey_label="[1] EASY",
        theme_rgb=(72, 235, 148),
        summary_line1="Light Traffic",
        summary_line2="No Blinker Merges",
        base_spawn_s=1.42,
        min_spawn_s=0.72,
        max_blocked_lanes=2,
        spawn_clearance_y=290.0,
        use_indicators=False,
        aggression_mult=0.0,
        signal_lead_range=(1.0, 1.2),
        merge_speed_mult=1.0,
        base_cruise_kmh=165.0,
        max_score_speed_bonus=55.0,
        nitro_kmh=270.0,
        score_mult=1.0,
    ),
    "MEDIUM": DifficultyConfig(
        key="MEDIUM",
        hotkey_label="[2] MEDIUM",
        theme_rgb=(88, 215, 255),
        summary_line1="Balanced Traffic",
        summary_line2="Amber Indicators ON",
        base_spawn_s=0.92,
        min_spawn_s=0.46,
        max_blocked_lanes=3,
        spawn_clearance_y=235.0,
        use_indicators=True,
        aggression_mult=1.0,
        signal_lead_range=(0.82, 1.12),
        merge_speed_mult=1.0,
        base_cruise_kmh=195.0,
        max_score_speed_bonus=68.0,
        nitro_kmh=305.0,
        score_mult=1.25,
    ),
    "HARD": DifficultyConfig(
        key="HARD",
        hotkey_label="[3] HARD",
        theme_rgb=(255, 92, 118),
        summary_line1="Heavy Rush Traffic",
        summary_line2="Fast Blinker Cuts!",
        base_spawn_s=0.62,
        min_spawn_s=0.34,
        max_blocked_lanes=3,
        spawn_clearance_y=190.0,
        use_indicators=True,
        aggression_mult=1.80,
        signal_lead_range=(0.56, 0.80),
        merge_speed_mult=1.35,
        base_cruise_kmh=220.0,
        max_score_speed_bonus=82.0,
        nitro_kmh=335.0,
        score_mult=1.60,
    ),
}


# ---------------------------------------------------------------------------
# Scale-Invariant Hand Gesture & Lucas-Kanade Optical Flow Worker Thread
# ---------------------------------------------------------------------------
@dataclass
class GestureControlSnapshot:
    """Thread-safe snapshot passed from Vision Thread -> 60 FPS Game Loop."""

    hand_detected: bool = False
    steer_target: float = 0.0   # Normalized [-1.0 .. +1.0] across highway width
    is_braking: bool = False    # Closed Fist -> Brake!
    is_nitro: bool = False      # Thumb + Index Pinch (fingers open) -> Nitro!
    flow_vx: float = 0.0        # Lucas-Kanade horizontal velocity (px/frame)
    flow_speed: float = 0.0     # Lucas-Kanade total speed magnitude (px/frame)
    gesture_name: str = "NO HAND"
    cam_fps: float = 0.0


def classify_driving_gesture(
    hand: HandGestureData, frame_w: int
) -> Tuple[bool, bool, float, str]:
    """
    Scale-invariant classification of:
      1. FIST  -> BRAKE (3+ non-thumb fingers curled into palm)
      2. PINCH -> NITRO BOOST (Thumb [4] + Index [8] close while other fingers extended)
      3. OPEN  -> CRUISE THROTTLE
    Also maps palm center smoothly to normalized road steering `[-1.0, +1.0]`.
    """
    lms = hand.landmarks_px
    wrist_x, wrist_y = lms[0]
    mid_mcp_x, mid_mcp_y = lms[9]
    palm_size = max(25.0, math.hypot(mid_mcp_x - wrist_x, mid_mcp_y - wrist_y))

    def dist_to_wrist(idx: int) -> float:
        return math.hypot(lms[idx][0] - wrist_x, lms[idx][1] - wrist_y)

    # Check finger curl on Index (8 vs 6), Middle (12 vs 10), Ring (16 vs 14), Pinky (20 vs 18)
    finger_pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
    curled_flags = [
        dist_to_wrist(tip_i) < 1.12 * dist_to_wrist(pip_i)
        for tip_i, pip_i in finger_pairs
    ]
    total_curled = sum(curled_flags)
    non_index_curled = sum(curled_flags[1:])  # Middle, Ring, Pinky

    # Priority 1: Closed Fist -> Reliable Distance-Independent Brake!
    if total_curled >= 3:
        is_braking = True
        is_nitro = False
        label = "BRAKE (FIST)"
    # Priority 2: Thumb + Index Pinch with outer fingers extended -> Nitro Boost!
    elif hand.distance_px < 0.52 * palm_size and non_index_curled <= 1:
        is_braking = False
        is_nitro = True
        label = "NITRO (PINCH)"
    else:
        is_braking = False
        is_nitro = False
        label = "CRUISE (OPEN)"

    # Use stable palm anchor (average of Wrist [0], Index MCP [5], Pinky MCP [17])
    # so closing fist or pinching NEVER shifts steering sideways!
    palm_cx = (lms[0][0] + lms[5][0] + lms[9][0] + lms[17][0]) * 0.25
    raw_norm = (palm_cx / float(frame_w) - 0.5) / 0.30
    steer = max(-1.0, min(1.0, raw_norm))
    return is_braking, is_nitro, steer, label


class AsyncVisionWorker:
    """Runs Webcam + MediaPipe + Lucas-Kanade Optical Flow on a dedicated daemon thread."""

    def __init__(self, camera_index: int = 0) -> None:
        self._camera_index = camera_index
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._snapshot = GestureControlSnapshot()
        self._latest_cam_bgr: Optional[np.ndarray] = None
        self._frame_seq: int = 0
        self.init_error: Optional[str] = None
        self._ready = threading.Event()

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=8.0)

    def get_latest(self) -> Tuple[GestureControlSnapshot, Optional[np.ndarray], int]:
        with self._lock:
            return self._snapshot, self._latest_cam_bgr, self._frame_seq

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.5)

    def _run_loop(self) -> None:
        try:
            cap = open_webcam(self._camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_CAPTURE_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_CAPTURE_H)
            tracker = HandPoseTracker(
                max_num_hands=1,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.55,
            )
        except Exception as exc:  # pragma: no cover - hardware fallback safety
            self.init_error = str(exc)
            self._ready.set()
            return

        self._ready.set()

        lk_params = dict(
            winSize=(19, 19),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03),
        )
        prev_gray: Optional[np.ndarray] = None
        prev_pts: Optional[np.ndarray] = None
        trails: List[Deque[Tuple[int, int]]] = [
            deque(maxlen=10) for _ in LK_LANDMARK_IDS
        ]

        smoothed_steer = 0.0
        cam_fps = 30.0
        prev_t = time.perf_counter()

        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.005)
                    continue

                now = time.perf_counter()
                dt = max(now - prev_t, 1e-4)
                cam_fps = 0.85 * cam_fps + 0.15 * (1.0 / dt)
                prev_t = now

                frame = cv2.flip(frame, 1)
                fh, fw = frame.shape[:2]
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                hands = tracker.process(frame, draw_skeleton=False)
                vx, vy = 0.0, 0.0

                if hands:
                    hand = hands[0]
                    anchor_pts = np.array(
                        [hand.landmarks_px[idx] for idx in LK_LANDMARK_IDS],
                        dtype=np.float32,
                    ).reshape(-1, 1, 2)

                    if (
                        prev_gray is not None
                        and prev_pts is not None
                        and prev_gray.shape == curr_gray.shape
                    ):
                        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                            prev_gray, curr_gray, prev_pts, None, **lk_params
                        )
                        if next_pts is not None and status is not None:
                            valid = status.reshape(-1) == 1
                            if np.any(valid):
                                diff = (
                                    next_pts[valid].reshape(-1, 2)
                                    - prev_pts[valid].reshape(-1, 2)
                                )
                                vx = float(np.mean(diff[:, 0]))
                                vy = float(np.mean(diff[:, 1]))
                                anchor_pts[valid] = (
                                    0.65 * next_pts[valid] + 0.35 * anchor_pts[valid]
                                )

                    for t_i, pt in enumerate(anchor_pts.reshape(-1, 2)):
                        trails[t_i].append((int(round(float(pt[0]))), int(round(float(pt[1])))))
                    prev_pts = anchor_pts

                    is_brake, is_nitro, raw_steer, label = classify_driving_gesture(
                        hand, fw
                    )
                    # Fuse optical-flow velocity lead with palm position for crisp zero-lag feel
                    lead_steer = max(-1.0, min(1.0, raw_steer + vx * 0.014))
                    smoothed_steer += (lead_steer - smoothed_steer) * 0.45

                    snap = GestureControlSnapshot(
                        hand_detected=True,
                        steer_target=smoothed_steer,
                        is_braking=is_brake,
                        is_nitro=is_nitro,
                        flow_vx=vx,
                        flow_speed=math.hypot(vx, vy),
                        gesture_name=label,
                        cam_fps=cam_fps,
                    )
                else:
                    prev_pts = None
                    for tr in trails:
                        if tr:
                            tr.popleft()
                    snap = GestureControlSnapshot(
                        hand_detected=False,
                        steer_target=smoothed_steer,
                        gesture_name="NO HAND (Keyboard Ready)",
                        cam_fps=cam_fps,
                    )

                prev_gray = curr_gray

                # Build clean annotated monitor image on worker thread
                vis = frame.copy()
                draw_camera_monitor_overlay(vis, tracker, hands, trails, snap)

                with self._lock:
                    self._snapshot = snap
                    self._latest_cam_bgr = vis
                    self._frame_seq += 1
        finally:
            tracker.close()
            cap.release()


def draw_camera_monitor_overlay(
    vis: np.ndarray,
    tracker: HandPoseTracker,
    hands: List[HandGestureData],
    trails: List[Deque[Tuple[int, int]]],
    snap: GestureControlSnapshot,
) -> None:
    """Annotate the separate OpenCV Camera Monitor window clearly."""
    h, w = vis.shape[:2]

    # Subtle vertical lane guide lines showing active steering zone
    left_guide = int(w * 0.20)
    right_guide = int(w * 0.80)
    cv2.line(vis, (left_guide, 54), (left_guide, h - 30), (70, 90, 110), 1)
    cv2.line(vis, (right_guide, 54), (right_guide, h - 30), (70, 90, 110), 1)
    cv2.line(vis, (w // 2, 54), (w // 2, h - 30), (55, 70, 85), 1)

    # Lucas-Kanade Optical Flow trails
    for tr in trails:
        pts = list(tr)
        for i in range(1, len(pts)):
            ratio = i / float(len(pts))
            col = (int(255 * (1.0 - 0.5 * ratio)), int(140 + 115 * ratio), int(255 * ratio))
            cv2.line(vis, pts[i - 1], pts[i], col, max(1, int(round(2.5 * ratio))), cv2.LINE_AA)

    tracker.draw_landmarks_on_frame(vis)

    # Header status banner
    cv2.rectangle(vis, (0, 0), (w, 50), (18, 22, 30), cv2.FILLED)
    badge_col = (
        (60, 80, 255)
        if snap.is_braking
        else ((0, 255, 190) if snap.is_nitro else (90, 220, 255))
    )
    cv2.putText(
        vis,
        f"Mode: {snap.gesture_name}",
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        badge_col,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        f"LK Flow: {snap.flow_speed:4.1f} px/f  |  Cam FPS: {snap.cam_fps:3.0f}",
        (10, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.41,
        (210, 225, 240),
        1,
        cv2.LINE_AA,
    )

    # Bottom steering bar
    cv2.rectangle(vis, (0, h - 26), (w, h), (18, 22, 30), cv2.FILLED)
    bx1, bx2, by = 60, w - 60, h - 13
    cv2.line(vis, (bx1, by), (bx2, by), (80, 100, 130), 3)
    kx = int(round(bx1 + (snap.steer_target + 1.0) * 0.5 * (bx2 - bx1)))
    cv2.circle(vis, (kx, by), 7, badge_col, cv2.FILLED)


# ---------------------------------------------------------------------------
# Procedural High-Detail Top-Down 2D Vehicle Sprite Factory
# ---------------------------------------------------------------------------
class VehicleSpriteBank:
    """Pre-renders crisp, high-detail top-down 2D cars & trucks once at startup."""

    def __init__(self) -> None:
        self.player_base = self._build_supercar_surface()
        self.rotation_cache: Dict[int, pygame.Surface] = {}
        self.traffic_surfaces: Dict[Tuple[str, Tuple[int, int, int]], pygame.Surface] = {}

        palettes = {
            "SEDAN": [
                (225, 65, 75),
                (65, 135, 235),
                (230, 235, 242),
                (52, 185, 125),
            ],
            "TAXI": [(248, 202, 38)],
            "COUPE": [(255, 115, 42), (168, 82, 245), (38, 212, 225)],
            "SUV": [(78, 92, 112), (212, 218, 226), (165, 58, 68)],
            "TRUCK": [(215, 222, 232), (82, 130, 188)],
        }
        for kind, colors in palettes.items():
            for col in colors:
                self.traffic_surfaces[(kind, col)] = self._build_traffic_surface(
                    kind, col
                )
        self.palettes = palettes

    def get_rotated_player(self, angle_deg: float) -> pygame.Surface:
        quant = int(round(max(-16.0, min(16.0, angle_deg)) * 2.0))
        surf = self.rotation_cache.get(quant)
        if surf is None:
            surf = pygame.transform.rotozoom(self.player_base, -quant * 0.5, 1.0)
            self.rotation_cache[quant] = surf
        return surf

    @staticmethod
    def _build_supercar_surface() -> pygame.Surface:
        """Design a sleek top-down Hero Supercar with racing stripes & aero details."""
        w, h = 68, 122
        s = pygame.Surface((w, h), pygame.SRCALPHA)

        # Soft drop shadow
        pygame.draw.rect(s, (0, 0, 0, 85), (6, 8, w - 8, h - 8), border_radius=18)

        # Four dark sport tires protruding subtly from wheel arches
        for wx, wy in ((3, 18), (w - 11, 18), (3, h - 38), (w - 11, h - 38)):
            pygame.draw.rect(s, (22, 25, 30), (wx, wy, 8, 22), border_radius=3)
            pygame.draw.line(s, (90, 100, 115), (wx + 2, wy + 4), (wx + 2, wy + 18), 1)

        # Sculpted main body shell (Pearl Electric Cyan)
        body_rect = pygame.Rect(8, 4, w - 16, h - 10)
        pygame.draw.rect(s, (18, 188, 242), body_rect, border_radius=17)

        # Metallic side contour highlights
        pygame.draw.rect(s, (125, 235, 255), (11, 10, w - 22, h - 22), width=2, border_radius=14)

        # Twin carbon-navy center racing stripes
        pygame.draw.rect(s, (18, 28, 46), (w // 2 - 8, 5, 6, h - 12))
        pygame.draw.rect(s, (18, 28, 46), (w // 2 + 2, 5, 6, h - 12))

        # Front carbon splitter lip & hood vents
        pygame.draw.rect(s, (25, 30, 38), (14, 3, w - 28, 5), border_radius=3)
        pygame.draw.polygon(s, (20, 32, 48), [(20, 24), (w - 20, 24), (w - 24, 31), (24, 31)])

        # Tinted wraparound windshield + cockpit glass roof
        pygame.draw.polygon(
            s,
            (24, 38, 58),
            [(14, 42), (w - 14, 42), (w - 18, 29), (18, 29)],
        )
        pygame.draw.line(s, (165, 230, 255), (19, 31), (w - 19, 31), 2)
        pygame.draw.rect(s, (16, 24, 38), (16, 42, w - 32, 34), border_radius=6)

        # Rear engine louvers & carbon rear spoiler wing
        for ly in range(82, 98, 5):
            pygame.draw.line(s, (24, 34, 48), (20, ly), (w - 20, ly), 2)
        pygame.draw.rect(s, (22, 26, 36), (9, h - 15, w - 18, 7), border_radius=3)
        pygame.draw.rect(s, (255, 65, 125), (10, h - 14, w - 20, 2))

        # Side mirrors
        pygame.draw.polygon(s, (18, 188, 242), [(8, 36), (2, 39), (8, 42)])
        pygame.draw.polygon(s, (18, 188, 242), [(w - 8, 36), (w - 2, 39), (w - 8, 42)])

        # Crisp front xenon headlamp lenses & rear LED tail strip
        pygame.draw.ellipse(s, (235, 252, 255), (12, 6, 11, 7))
        pygame.draw.ellipse(s, (235, 252, 255), (w - 23, 6, 11, 7))
        pygame.draw.rect(s, (185, 30, 48), (12, h - 9, 14, 4), border_radius=2)
        pygame.draw.rect(s, (185, 30, 48), (w - 26, h - 9, 14, 4), border_radius=2)
        return s

    @staticmethod
    def _build_traffic_surface(
        kind: str, body_col: Tuple[int, int, int]
    ) -> pygame.Surface:
        """Design distinct top-down traffic vehicles: Sedan, Taxi, Coupe, SUV, Semi-Truck."""
        dims = {
            "SEDAN": (62, 112),
            "TAXI": (62, 112),
            "COUPE": (60, 108),
            "SUV": (68, 124),
            "TRUCK": (76, 198),
        }
        w, h = dims[kind]
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(s, (0, 0, 0, 80), (5, 7, w - 6, h - 6), border_radius=12)

        dark_col = tuple(max(0, int(c * 0.68)) for c in body_col)
        light_col = tuple(min(255, int(c * 1.18) + 18) for c in body_col)

        if kind == "TRUCK":
            # Wheels along cab + double rear axle
            for wy in (20, 62, h - 54, h - 28):
                pygame.draw.rect(s, (24, 26, 30), (3, wy, 7, 19), border_radius=2)
                pygame.draw.rect(s, (24, 26, 30), (w - 10, wy, 7, 19), border_radius=2)
            # Tractor cab at front
            pygame.draw.rect(s, body_col, (9, 5, w - 18, 52), border_radius=10)
            pygame.draw.rect(s, (34, 48, 66), (13, 14, w - 26, 15), border_radius=4)
            # Hitch bar + long ribbed cargo box trailer
            pygame.draw.rect(s, (45, 50, 58), (w // 2 - 8, 55, 16, 12))
            pygame.draw.rect(s, (236, 240, 246), (7, 64, w - 14, h - 70), border_radius=6)
            pygame.draw.rect(s, body_col, (10, 68, w - 20, h - 78), width=3, border_radius=4)
            for ry in range(80, h - 18, 18):
                pygame.draw.line(s, (205, 212, 222), (10, ry), (w - 10, ry), 2)
        else:
            for wx, wy in ((3, 17), (w - 10, 17), (3, h - 35), (w - 10, h - 35)):
                pygame.draw.rect(s, (24, 26, 30), (wx, wy, 7, 20), border_radius=3)

            rad = 16 if kind == "COUPE" else 13
            pygame.draw.rect(s, body_col, (8, 4, w - 16, h - 8), border_radius=rad)
            pygame.draw.rect(s, light_col, (11, 9, w - 22, h - 18), width=2, border_radius=rad - 2)

            if kind == "COUPE":
                pygame.draw.rect(s, (245, 248, 255), (w // 2 - 3, 5, 6, h - 10))

            # Front & rear windshields + roof panel
            pygame.draw.polygon(
                s, (30, 44, 62), [(13, 40), (w - 13, 40), (w - 16, 26), (16, 26)]
            )
            pygame.draw.rect(s, dark_col, (14, 40, w - 28, h - 74), border_radius=5)
            pygame.draw.polygon(
                s,
                (30, 44, 62),
                [(15, h - 34), (w - 15, h - 34), (w - 13, h - 22), (13, h - 22)],
            )

            if kind == "TAXI":
                # Iconic black/white taxi sign on roof
                pygame.draw.rect(s, (22, 24, 28), (w // 2 - 13, h // 2 - 6, 26, 11), border_radius=3)
                pygame.draw.rect(s, (255, 245, 190), (w // 2 - 9, h // 2 - 4, 18, 7), border_radius=2)
            elif kind == "SUV":
                # Roof rails + panoramic sunroof
                pygame.draw.line(s, (215, 222, 232), (15, 34), (15, h - 26), 2)
                pygame.draw.line(s, (215, 222, 232), (w - 15, 34), (w - 15, h - 26), 2)
                pygame.draw.rect(s, (25, 35, 48), (19, 45, w - 38, 26), border_radius=4)

        # Headlights & dim tail lamps
        pygame.draw.rect(s, (245, 250, 235), (12, 5, 10, 4), border_radius=2)
        pygame.draw.rect(s, (245, 250, 235), (w - 22, 5, 10, 4), border_radius=2)
        pygame.draw.rect(s, (165, 28, 38), (11, h - 8, 12, 4), border_radius=2)
        pygame.draw.rect(s, (165, 28, 38), (w - 23, h - 8, 12, 4), border_radius=2)
        return s


# ---------------------------------------------------------------------------
# Traffic Vehicle AI & Visual FX Entities
# ---------------------------------------------------------------------------
@dataclass
class TrafficCar:
    kind: str
    color: Tuple[int, int, int]
    surface: pygame.Surface
    lane: int
    target_lane: int
    x: float
    y: float
    base_speed_kmh: float
    speed_kmh: float
    aggression: float
    decision_timer: float
    blinker_dir: int = 0          # -1 = left indicator, +1 = right indicator, 0 = off
    signal_lead_timer: float = 0.0
    is_merging: bool = False
    is_braking: bool = False
    heading_deg: float = 0.0
    passed_awarded: bool = False

    @property
    def width(self) -> int:
        return self.surface.get_width()

    @property
    def height(self) -> int:
        return self.surface.get_height()

    def hitbox(self) -> pygame.Rect:
        pad_x, pad_y = 9, 8
        return pygame.Rect(
            int(self.x - self.width * 0.5) + pad_x,
            int(self.y - self.height * 0.5) + pad_y,
            self.width - 2 * pad_x,
            self.height - 2 * pad_y,
        )


@dataclass
class FloatingPopup:
    text: str
    x: float
    y: float
    color: Tuple[int, int, int]
    timer: float = 0.95


@dataclass
class SkidSegment:
    x: float
    y: float


@dataclass
class SmokePuff:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    color: Tuple[int, int, int]
    life: float = 0.42


# ---------------------------------------------------------------------------
# Main Top-Down 2D Highway Traffic Game State
# ---------------------------------------------------------------------------
class TopDownTrafficGame:
    """Manages road scrolling, unpredictable blinking traffic AI, collisions & HUD."""

    def __init__(self, sprites: VehicleSpriteBank) -> None:
        self.sprites = sprites
        self.tile_h = 240
        self.road_tile = self._bake_road_tile(self.tile_h)
        self.headlight_cone = self._bake_headlight_cone()

        self.font_hud = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_big = pygame.font.SysFont("consolas", 36, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 15, bold=True)

        self.difficulty_key: str = "MEDIUM"
        self.menu_cursor_idx: int = 1
        self.in_menu: bool = True
        self.pinch_latch: bool = False
        self.best_scores: Dict[str, int] = {m: 0 for m in DIFFICULTY_ORDER}

        self.scenery_items: List[Tuple[float, float, str]] = []
        for sy in range(0, GAME_H + 120, 95):
            self._spawn_scenery_row(float(sy))

        self.reset()

    @property
    def cfg(self) -> DifficultyConfig:
        return DIFFICULTY_CONFIGS[self.difficulty_key]

    @property
    def best_score(self) -> int:
        return self.best_scores[self.difficulty_key]

    @property
    def score_ramp(self) -> float:
        """Smooth [0.0 .. 1.0) progression factor driven directly by current score."""
        return 1.0 - math.exp(-max(0.0, self.score) / 5000.0)

    def select_difficulty(self, mode_key: str) -> None:
        """Select EASY, MEDIUM, or HARD mode and start a fresh race immediately."""
        self.difficulty_key = mode_key
        self.menu_cursor_idx = DIFFICULTY_ORDER.index(mode_key)
        self.in_menu = False
        self.pinch_latch = True
        self.reset()

    def open_menu(self) -> None:
        """Open the interactive Easy / Medium / Hard mode selector overlay."""
        self.in_menu = True
        self.pinch_latch = True

    @staticmethod
    def mode_card_rects() -> List[pygame.Rect]:
        card_w, card_h, gap = 228, 225, 22
        total_w = 3 * card_w + 2 * gap
        start_x = (GAME_W - total_w) // 2
        y = GAME_H // 2 - 90
        return [
            pygame.Rect(start_x + i * (card_w + gap), y, card_w, card_h)
            for i in range(3)
        ]

    @staticmethod
    def _bake_road_tile(tile_h: int) -> pygame.Surface:
        """Pre-bake crisp textured asphalt, rumble strips & lush grass shoulders."""
        surf = pygame.Surface((GAME_W, tile_h))
        surf.fill((34, 68, 42))  # Rich roadside turf

        # Subtle grass texture stripes
        for gy in range(0, tile_h, 24):
            pygame.draw.rect(surf, (30, 62, 38), (0, gy, ROAD_LEFT, 12))
            pygame.draw.rect(surf, (30, 62, 38), (ROAD_RIGHT, gy, GAME_W - ROAD_RIGHT, 12))

        # Concrete shoulder kerbs
        pygame.draw.rect(surf, (62, 66, 76), (ROAD_LEFT - 20, 0, ROAD_W + 40, tile_h))

        # Main 4-lane asphalt surface
        pygame.draw.rect(surf, (40, 44, 52), (ROAD_LEFT, 0, ROAD_W, tile_h))

        # Crisp asphalt micro-grain (deterministic RNG so tile seams match cleanly)
        rng = random.Random(42)
        for _ in range(420):
            gx = rng.randint(ROAD_LEFT + 4, ROAD_RIGHT - 5)
            gy = rng.randint(0, tile_h - 1)
            shade = rng.choice((34, 48, 55))
            surf.set_at((gx, gy), (shade, shade + 3, shade + 9))

        # Red-and-white racing kerb rumble strips on left & right shoulders
        stripe_h = 30
        for idx, sy in enumerate(range(0, tile_h, stripe_h)):
            k_col = (218, 58, 62) if idx % 2 == 0 else (232, 236, 242)
            pygame.draw.rect(surf, k_col, (ROAD_LEFT - 12, sy, 10, stripe_h))
            pygame.draw.rect(surf, k_col, (ROAD_RIGHT + 2, sy, 10, stripe_h))

        # Solid yellow/white outer edge lines
        pygame.draw.line(surf, (240, 200, 70), (ROAD_LEFT + 3, 0), (ROAD_LEFT + 3, tile_h), 3)
        pygame.draw.line(surf, (240, 200, 70), (ROAD_RIGHT - 4, 0), (ROAD_RIGHT - 4, tile_h), 3)

        # Dashed white lane dividers between the 4 lanes
        for lane_i in range(1, NUM_LANES):
            lx = ROAD_LEFT + LANE_W * lane_i
            for dy in range(12, tile_h, 60):
                pygame.draw.rect(surf, (226, 232, 240), (lx - 3, dy, 6, 32), border_radius=2)

        return surf.convert()

    @staticmethod
    def _bake_headlight_cone() -> pygame.Surface:
        cone = pygame.Surface((120, 175), pygame.SRCALPHA)
        for step in range(5):
            alpha = 30 - step * 5
            spread = step * 7
            pygame.draw.polygon(
                cone,
                (235, 250, 255, alpha),
                [
                    (40, 172),
                    (18 - spread // 2, 10),
                    (58 + spread // 2, 10),
                ],
            )
            pygame.draw.polygon(
                cone,
                (235, 250, 255, alpha),
                [
                    (80, 172),
                    (62 - spread // 2, 10),
                    (102 + spread // 2, 10),
                ],
            )
        return cone

    def _spawn_scenery_row(self, y_pos: float) -> None:
        left_x = random.uniform(35.0, ROAD_LEFT - 48.0)
        right_x = random.uniform(ROAD_RIGHT + 48.0, GAME_W - 35.0)
        self.scenery_items.append((left_x, y_pos, random.choice(("TREE", "LAMP", "BUSH"))))
        self.scenery_items.append((right_x, y_pos, random.choice(("TREE", "LAMP", "BUSH"))))

    def reset(self) -> None:
        cfg = self.cfg
        self.player_x: float = float((LANE_CENTERS[1] + LANE_CENTERS[2]) * 0.5)
        self.player_vx: float = 0.0
        self.player_heading: float = 0.0
        self.speed_kmh: float = cfg.base_cruise_kmh * 0.85
        self.is_braking: bool = False
        self.is_nitro: bool = False
        self.player_blinker: int = 0

        self.shields: int = 3
        self.invuln_timer: float = 0.0
        self.shake_timer: float = 0.0
        self.game_over: bool = False
        self.score: float = 0.0
        self.stage: int = 1
        self.combo: int = 1
        self.combo_decay_timer: float = 0.0
        self.distance_m: float = 0.0

        self.scroll_y: float = 0.0
        self.scenery_spawn_acc: float = 0.0
        self.spawn_timer: float = 0.35

        self.traffic: List[TrafficCar] = []
        self.popups: List[FloatingPopup] = [
            FloatingPopup(
                f"{cfg.key} MODE - {cfg.summary_line2.upper()}",
                GAME_W * 0.5,
                PLAYER_Y - 125,
                cfg.theme_rgb,
                timer=1.5,
            )
        ]
        self.skids: Deque[SkidSegment] = deque(maxlen=90)
        self.particles: List[SmokePuff] = []

    def player_hitbox(self) -> pygame.Rect:
        return pygame.Rect(int(self.player_x) - 23, int(PLAYER_Y) - 48, 46, 96)

    def _try_spawn_traffic(self) -> None:
        """Spawn vehicles according to active difficulty mode & score progression."""
        cfg = self.cfg
        ramp = self.score_ramp

        candidate_lanes = list(range(NUM_LANES))
        random.shuffle(candidate_lanes)

        # Enforce mode-specific max blocked lanes (Easy leaves >= 2 open lanes!)
        blocked_top = set()
        for c in self.traffic:
            if c.y < 195:
                blocked_top.add(c.lane)
                blocked_top.add(c.target_lane)
        if len(blocked_top) >= cfg.max_blocked_lanes:
            return

        # Slight clearance tightening as score rises while staying completely fair
        clearance_y = max(175.0, cfg.spawn_clearance_y - 35.0 * ramp)

        for lane in candidate_lanes:
            if any(
                (c.lane == lane or c.target_lane == lane) and c.y < clearance_y
                for c in self.traffic
            ):
                continue

            # Vehicle mix shifts toward faster sport coupes in harder modes & higher scores
            fast_weight = 10 + int(18 * ramp) + (14 if cfg.key == "HARD" else 0)
            if lane in (0, NUM_LANES - 1):
                kind = random.choices(
                    ["SEDAN", "TAXI", "SUV", "TRUCK", "COUPE"],
                    weights=[30, 18, 22, 18, fast_weight],
                )[0]
            else:
                kind = random.choices(
                    ["SEDAN", "TAXI", "COUPE", "SUV"],
                    weights=[30, 20, 18 + fast_weight, 18],
                )[0]

            speed_ranges = {
                "TRUCK": (84.0, 106.0),
                "SUV": (102.0, 130.0),
                "SEDAN": (110.0, 144.0),
                "TAXI": (114.0, 150.0),
                "COUPE": (132.0, 170.0),
            }
            lo, hi = speed_ranges[kind]
            # Traffic pace rises steadily as player score climbs
            spd = random.uniform(lo, hi) + 24.0 * ramp
            col = random.choice(self.sprites.palettes[kind])
            surf = self.sprites.traffic_surfaces[(kind, col)]
            aggr = (
                random.uniform(0.55, 0.95)
                if kind in ("COUPE", "TAXI")
                else random.uniform(0.20, 0.55)
            )
            if kind == "TRUCK":
                aggr = random.uniform(0.08, 0.22)

            self.traffic.append(
                TrafficCar(
                    kind=kind,
                    color=col,
                    surface=surf,
                    lane=lane,
                    target_lane=lane,
                    x=float(LANE_CENTERS[lane]),
                    y=-surf.get_height(),
                    base_speed_kmh=spd,
                    speed_kmh=spd,
                    aggression=aggr,
                    decision_timer=random.uniform(0.45, 1.9),
                )
            )
            break

    def _lane_is_clear_for_merge(self, mover: TrafficCar, dest_lane: int) -> bool:
        """Ensure target lane has room among AI cars & doesn't sideswipe beside player."""
        for other in self.traffic:
            if other is mover:
                continue
            if other.lane == dest_lane or other.target_lane == dest_lane:
                min_gap = (mover.height + other.height) * 0.5 + 48.0
                if abs(other.y - mover.y) < min_gap:
                    return False

        # Never initiate a sudden sideways slam when already directly alongside player;
        # DO initiate ahead of the player so the player sees the blinker and reacts!
        dest_x = LANE_CENTERS[dest_lane]
        if abs(mover.y - PLAYER_Y) < 115.0 and abs(dest_x - self.player_x) < 95.0:
            return False
        return True

    def _update_traffic_ai(self, dt: float) -> None:
        """Update traffic following speeds, mode-dependent blinkers & lane changes."""
        cfg = self.cfg
        ramp = self.score_ramp

        for car in self.traffic:
            car.is_braking = False
            ahead_car: Optional[TrafficCar] = None
            min_dy = 9999.0
            for other in self.traffic:
                if other is car:
                    continue
                same_corridor = (
                    other.lane == car.lane or other.target_lane == car.target_lane
                )
                dy = car.y - other.y  # `other` is further ahead when `other.y < car.y`
                if same_corridor and 0.0 < dy < min_dy:
                    min_dy = dy
                    ahead_car = other

            desired_spd = car.base_speed_kmh
            blocked_ahead = False
            safe_follow_dist = (
                (car.height + ahead_car.height) * 0.5 + 55.0
                if ahead_car is not None
                else 0.0
            )
            if ahead_car is not None and min_dy < safe_follow_dist + 75.0:
                blocked_ahead = True
                desired_spd = min(desired_spd, max(70.0, ahead_car.speed_kmh - 8.0))
                if min_dy < safe_follow_dist + 25.0:
                    car.is_braking = True

            car.speed_kmh += (desired_spd - car.speed_kmh) * min(1.0, 4.5 * dt)

            # EASY Mode strictly keeps vehicles in their own lane without indicators!
            if not cfg.use_indicators:
                car.blinker_dir = 0
                car.is_merging = False
                car.heading_deg = 0.0
            else:
                # MEDIUM & HARD Modes: Use amber indicators & switch lanes more as score grows!
                eff_aggr = (
                    car.aggression * cfg.aggression_mult * (1.0 + 0.85 * ramp)
                )

                # State 1: Cruising -> decide unpredictably (or when blocked!) to switch lanes
                if car.blinker_dir == 0 and not car.is_merging:
                    car.decision_timer -= dt
                    player_approaching = 140.0 < (PLAYER_Y - car.y) < 560.0
                    want_check = car.decision_timer <= 0.0 or blocked_ahead
                    if want_check and -40.0 < car.y < PLAYER_Y - 135.0:
                        car.decision_timer = random.uniform(
                            max(0.65, 1.15 - 0.35 * ramp),
                            max(1.40, 2.90 - 0.80 * ramp),
                        )
                        chance = eff_aggr * (
                            1.45 if (blocked_ahead or player_approaching) else 0.65
                        )
                        if random.random() < chance:
                            neighbors = [
                                nl
                                for nl in (car.lane - 1, car.lane + 1)
                                if 0 <= nl < NUM_LANES
                            ]
                            random.shuffle(neighbors)
                            for nl in neighbors:
                                if self._lane_is_clear_for_merge(car, nl):
                                    car.target_lane = nl
                                    car.blinker_dir = -1 if nl < car.lane else 1
                                    lo_t, hi_t = cfg.signal_lead_range
                                    car.signal_lead_timer = random.uniform(lo_t, hi_t)
                                    break

                # State 2: Flashing Amber Turn Signal before lane change
                elif car.blinker_dir != 0 and not car.is_merging:
                    car.signal_lead_timer -= dt
                    if not self._lane_is_clear_for_merge(car, car.target_lane):
                        car.target_lane = car.lane
                        car.blinker_dir = 0
                    elif car.signal_lead_timer <= 0.0:
                        car.is_merging = True

                # State 3: Smoothly steering across into `target_lane` with blinker flashing
                if car.is_merging:
                    dest_x = float(LANE_CENTERS[car.target_lane])
                    diff_x = dest_x - car.x
                    lateral_speed = (
                        (115.0 + 75.0 * car.aggression) * cfg.merge_speed_mult * dt
                    )
                    if abs(diff_x) <= lateral_speed:
                        car.x = dest_x
                        car.lane = car.target_lane
                        car.is_merging = False
                        car.blinker_dir = 0
                        car.heading_deg = 0.0
                        car.decision_timer = random.uniform(1.4, 3.2)
                    else:
                        step_x = math.copysign(lateral_speed, diff_x)
                        car.x += step_x
                        car.heading_deg += (
                            math.copysign(11.5, diff_x) - car.heading_deg
                        ) * min(1.0, 9.0 * dt)
                else:
                    car.heading_deg += (0.0 - car.heading_deg) * min(1.0, 8.0 * dt)

            # Vertical motion relative to scrolling camera (`player speed - traffic speed`)
            rel_vy = (self.speed_kmh - car.speed_kmh) * KMH_TO_PX_PER_SEC
            car.y += rel_vy * dt

    def update(
        self,
        dt: float,
        ctrl: GestureControlSnapshot,
        keys_steer: float,
        key_brake: bool,
        key_nitro: bool,
    ) -> None:
        if not ctrl.is_nitro:
            self.pinch_latch = False

        if self.in_menu:
            # Gentle animated background highway scroll behind Mode Menu
            self.scroll_y = (self.scroll_y + 180.0 * dt) % self.tile_h
            if ctrl.hand_detected:
                if ctrl.steer_target < -0.33:
                    self.menu_cursor_idx = 0
                elif ctrl.steer_target > 0.33:
                    self.menu_cursor_idx = 2
                else:
                    self.menu_cursor_idx = 1
                if ctrl.is_nitro and not self.pinch_latch:
                    self.select_difficulty(DIFFICULTY_ORDER[self.menu_cursor_idx])
            return

        if self.game_over:
            if ctrl.is_nitro and not self.pinch_latch:  # Pinch to restart hands-free!
                self.pinch_latch = True
                self.reset()
            return

        cfg = self.cfg
        ramp = self.score_ramp

        # 1. Combine Hand Gestures + Optional Keyboard Controls seamlessly
        self.is_braking = ctrl.is_braking or key_brake
        self.is_nitro = (not self.is_braking) and (ctrl.is_nitro or key_nitro)

        if self.is_braking:
            target_speed = MIN_SPEED_KMH
            accel_rate = 3.8
        elif self.is_nitro:
            target_speed = cfg.nitro_kmh + 30.0 * ramp
            accel_rate = 2.8
        else:
            target_speed = cfg.base_cruise_kmh + cfg.max_score_speed_bonus * ramp
            accel_rate = 1.9

        self.speed_kmh += (target_speed - self.speed_kmh) * min(1.0, accel_rate * dt)

        # Lateral steering
        min_car_x = float(ROAD_LEFT + 38)
        max_car_x = float(ROAD_RIGHT - 38)
        prev_px = self.player_x

        if ctrl.hand_detected and abs(keys_steer) < 0.01:
            road_mid = (min_car_x + max_car_x) * 0.5
            road_span = (max_car_x - min_car_x) * 0.5
            desired_x = road_mid + ctrl.steer_target * road_span
            self.player_x += (desired_x - self.player_x) * min(1.0, 11.5 * dt)
        else:
            self.player_x += keys_steer * 420.0 * dt

        self.player_x = max(min_car_x, min(max_car_x, self.player_x))
        self.player_vx = (self.player_x - prev_px) / max(dt, 1e-4)
        self.player_heading += (
            max(-14.0, min(14.0, self.player_vx * 0.032)) - self.player_heading
        ) * min(1.0, 12.0 * dt)
        self.player_blinker = (
            -1 if self.player_vx < -95.0 else (1 if self.player_vx > 95.0 else 0)
        )

        # Tire skid marks & smoke when braking hard, flames smoke when Nitro active
        world_px_s = self.speed_kmh * KMH_TO_PX_PER_SEC
        for sk in self.skids:
            sk.y += world_px_s * dt
        if self.is_braking and self.speed_kmh > 95.0:
            self.skids.append(SkidSegment(self.player_x - 19, PLAYER_Y + 42))
            self.skids.append(SkidSegment(self.player_x + 19, PLAYER_Y + 42))
            if random.random() < 0.55:
                self.particles.append(
                    SmokePuff(
                        self.player_x + random.choice((-20.0, 20.0)),
                        PLAYER_Y + 48.0,
                        random.uniform(-18.0, 18.0),
                        random.uniform(70.0, 130.0),
                        random.uniform(5.0, 9.0),
                        (185, 190, 200),
                    )
                )
        elif self.is_nitro and random.random() < 0.75:
            self.particles.append(
                SmokePuff(
                    self.player_x + random.choice((-13.0, 13.0)),
                    PLAYER_Y + 62.0,
                    random.uniform(-12.0, 12.0),
                    random.uniform(140.0, 220.0),
                    random.uniform(4.0, 7.5),
                    (80, 225, 255),
                )
            )

        # Scroll highway & roadside scenery
        self.scroll_y = (self.scroll_y + world_px_s * dt) % self.tile_h
        self.scenery_spawn_acc += world_px_s * dt
        while self.scenery_spawn_acc >= 95.0:
            self.scenery_spawn_acc -= 95.0
            self._spawn_scenery_row(-40.0)
        self.scenery_items = [
            (sx, sy + world_px_s * dt, kind)
            for sx, sy, kind in self.scenery_items
            if sy + world_px_s * dt < GAME_H + 80
        ]

        # Distance, Score & Dynamic Score-Based Stage Escalation
        meters_step = (self.speed_kmh / 3.6) * dt
        self.distance_m += meters_step
        mult = (2.0 if self.is_nitro else 1.0) * cfg.score_mult
        self.score += meters_step * 1.45 * mult
        self.best_scores[self.difficulty_key] = max(
            self.best_scores[self.difficulty_key], int(self.score)
        )

        new_stage = 1 + int(self.score // 1500)
        if new_stage > self.stage:
            self.stage = new_stage
            self.popups.append(
                FloatingPopup(
                    f"STAGE {self.stage}! SPEED & TRAFFIC UP!",
                    GAME_W * 0.5,
                    PLAYER_Y - 120,
                    (255, 220, 75),
                    timer=1.45,
                )
            )

        if self.combo_decay_timer > 0.0:
            self.combo_decay_timer -= dt
            if self.combo_decay_timer <= 0.0:
                self.combo = 1
        if self.invuln_timer > 0.0:
            self.invuln_timer = max(0.0, self.invuln_timer - dt)
        if self.shake_timer > 0.0:
            self.shake_timer = max(0.0, self.shake_timer - dt)

        # Spawn interval tightens steadily as score grows!
        self.spawn_timer -= dt
        if self.spawn_timer <= 0.0:
            self._try_spawn_traffic()
            cur_interval = cfg.base_spawn_s - self.score_ramp * (
                cfg.base_spawn_s - cfg.min_spawn_s
            )
            self.spawn_timer = random.uniform(cur_interval * 0.82, cur_interval * 1.18)

        self._update_traffic_ai(dt)

        # Near-miss overtake rewards & collision handling
        p_box = self.player_hitbox()
        for car in self.traffic:
            c_box = car.hitbox()
            if p_box.colliderect(c_box):
                if self.invuln_timer <= 0.0:
                    self.shields -= 1
                    self.combo = 1
                    self.invuln_timer = 1.5
                    self.shake_timer = 0.30
                    self.speed_kmh = max(MIN_SPEED_KMH, self.speed_kmh * 0.62)
                    # Gently push colliding traffic car forward out of overlap
                    car.y -= 75.0
                    for _ in range(14):
                        self.particles.append(
                            SmokePuff(
                                (self.player_x + car.x) * 0.5,
                                (PLAYER_Y + car.y) * 0.5,
                                random.uniform(-150.0, 150.0),
                                random.uniform(-120.0, 120.0),
                                random.uniform(4.0, 8.0),
                                (255, 195, 65),
                            )
                        )
                    if self.shields <= 0:
                        self.game_over = True
                    else:
                        self.popups.append(
                            FloatingPopup(
                                "CRASH! SHIELD -1",
                                self.player_x,
                                PLAYER_Y - 70,
                                (255, 95, 95),
                            )
                        )
            elif (
                not car.passed_awarded
                and car.y > PLAYER_Y + 35
                and abs(car.x - self.player_x) < 108.0
                and self.speed_kmh > 135.0
            ):
                car.passed_awarded = True
                pts = int(round(100 * self.combo * (2 if self.is_nitro else 1) * cfg.score_mult))
                self.score += pts
                self.best_scores[self.difficulty_key] = max(
                    self.best_scores[self.difficulty_key], int(self.score)
                )
                self.combo = min(10, self.combo + 1)
                self.combo_decay_timer = 3.2
                self.popups.append(
                    FloatingPopup(
                        f"CLOSE OVERTAKE +{pts} (x{self.combo})",
                        self.player_x,
                        PLAYER_Y - 68,
                        (95, 255, 190),
                    )
                )

        self.traffic = [
            c for c in self.traffic if -260.0 < c.y < GAME_H + 240.0
        ]

        for pop in self.popups:
            pop.y -= 42.0 * dt
            pop.timer -= dt
        self.popups = [p for p in self.popups if p.timer > 0.0]

        for pf in self.particles:
            pf.x += pf.vx * dt
            pf.y += pf.vy * dt
            pf.life -= dt
        self.particles = [p for p in self.particles if p.life > 0.0]

    # -----------------------------------------------------------------------
    # Pure 2D Top-Down Rendering Pipeline
    # -----------------------------------------------------------------------
    def render(self, screen: pygame.Surface, ctrl: GestureControlSnapshot, fps: float) -> None:
        ox = int(random.uniform(-5, 5)) if self.shake_timer > 0.0 else 0
        oy = int(random.uniform(-5, 5)) if self.shake_timer > 0.0 else 0

        # 1. Seamless scrolling road tiles
        start_y = int(self.scroll_y) - self.tile_h
        for ty in range(start_y, GAME_H, self.tile_h):
            screen.blit(self.road_tile, (ox, ty + oy))

        # 2. Roadside trees, bushes & streetlamps
        for sx, sy, kind in self.scenery_items:
            ix, iy = int(sx) + ox, int(sy) + oy
            if kind == "TREE":
                pygame.draw.circle(screen, (20, 42, 26), (ix + 4, iy + 5), 22)
                pygame.draw.circle(screen, (42, 112, 56), (ix, iy), 21)
                pygame.draw.circle(screen, (68, 148, 76), (ix - 5, iy - 5), 12)
            elif kind == "BUSH":
                pygame.draw.circle(screen, (36, 96, 48), (ix, iy), 13)
                pygame.draw.circle(screen, (56, 128, 66), (ix + 6, iy - 2), 9)
            else:
                pole_dir = 1 if sx < ROAD_LEFT else -1
                pygame.draw.circle(screen, (75, 82, 95), (ix, iy), 6)
                pygame.draw.line(
                    screen, (125, 135, 150), (ix, iy), (ix + pole_dir * 26, iy), 4
                )
                pygame.draw.circle(
                    screen, (255, 240, 185), (ix + pole_dir * 26, iy), 6
                )

        # 3. Tire brake skid segments on asphalt
        for sk in self.skids:
            pygame.draw.rect(
                screen, (24, 26, 30), (int(sk.x) - 3 + ox, int(sk.y) + oy, 6, 14)
            )

        blink_on = (int(time.perf_counter() * 7.5) % 2) == 0

        # 4. AI Traffic Cars + Flashing Amber Turn Indicators + Brake Lamps
        for car in self.traffic:
            surf = (
                pygame.transform.rotozoom(car.surface, -car.heading_deg, 1.0)
                if abs(car.heading_deg) > 0.6
                else car.surface
            )
            rect = surf.get_rect(center=(int(car.x) + ox, int(car.y) + oy))
            screen.blit(surf, rect)

            half_w = car.width // 2 - 6
            half_h = car.height // 2 - 5
            cx, cy = int(car.x) + ox, int(car.y) + oy

            # Red brake lights when slowing down behind another vehicle
            if car.is_braking:
                for side in (-1, 1):
                    pygame.draw.circle(
                        screen, (255, 42, 55), (cx + side * (half_w - 6), cy + half_h), 6
                    )

            # High-visibility flashing AMBER TURN SIGNAL INDICATORS + lane-merge arrow!
            if car.blinker_dir != 0 and blink_on:
                bx = cx + car.blinker_dir * half_w
                for by in (cy - half_h + 3, cy + half_h - 3):
                    pygame.draw.circle(screen, (255, 196, 25), (bx, by), 8)
                    pygame.draw.circle(screen, (255, 250, 200), (bx, by), 4)
                # Crisp amber chevron beside the vehicle so lane switches read instantly
                tip_x = bx + car.blinker_dir * 18
                pygame.draw.polygon(
                    screen,
                    (255, 185, 20),
                    [(tip_x, cy), (bx + car.blinker_dir * 6, cy - 9), (bx + car.blinker_dir * 6, cy + 9)],
                )

        # 5. Smoke & spark particles
        for pf in self.particles:
            pygame.draw.circle(
                screen,
                pf.color,
                (int(pf.x) + ox, int(pf.y) + oy),
                max(2, int(pf.radius * (pf.life / 0.42))),
            )

        # 6. Player Hero Supercar + Headlights + Brake Glow + Nitro Flames
        px, py = int(self.player_x) + ox, int(PLAYER_Y) + oy
        screen.blit(self.headlight_cone, (px - 60, py - 215))

        if self.is_nitro:
            flick = random.randint(24, 42)
            for fx in (-14, 14):
                pygame.draw.polygon(
                    screen,
                    (40, 215, 255),
                    [(px + fx - 7, py + 54), (px + fx + 7, py + 54), (px + fx, py + 54 + flick)],
                )
                pygame.draw.polygon(
                    screen,
                    (230, 255, 255),
                    [(px + fx - 4, py + 54), (px + fx + 4, py + 54), (px + fx, py + 54 + flick // 2)],
                )

        blink_hide = self.invuln_timer > 0.0 and (int(self.invuln_timer * 15) % 2 == 1)
        if not blink_hide:
            p_surf = self.sprites.get_rotated_player(self.player_heading)
            screen.blit(p_surf, p_surf.get_rect(center=(px, py)))

            if self.is_braking:
                for sx_off in (-18, 18):
                    pygame.draw.circle(screen, (255, 38, 52), (px + sx_off, py + 53), 9)
                    pygame.draw.circle(screen, (255, 205, 205), (px + sx_off, py + 53), 4)

            if self.player_blinker != 0 and blink_on:
                bx = px + self.player_blinker * 26
                pygame.draw.circle(screen, (255, 195, 25), (bx, py - 50), 6)
                pygame.draw.circle(screen, (255, 195, 25), (bx, py + 50), 6)

        # 7. Floating Near-Miss / Crash popups
        for pop in self.popups:
            lbl = self.font_small.render(pop.text, True, pop.color)
            bg = lbl.get_rect(center=(int(pop.x), int(pop.y))).inflate(12, 6)
            pygame.draw.rect(screen, (15, 20, 30), bg, border_radius=6)
            screen.blit(lbl, lbl.get_rect(center=bg.center))

        # 8. Crisp Arcade Top Dashboard & Bottom Gesture Telemetry Bar
        self._draw_dashboard(screen, ctrl, fps)

    def _draw_dashboard(
        self, screen: pygame.Surface, ctrl: GestureControlSnapshot, fps: float
    ) -> None:
        cfg = self.cfg
        header = pygame.Surface((GAME_W, 74), pygame.SRCALPHA)
        header.fill((14, 18, 28, 224))
        screen.blit(header, (0, 0))
        pygame.draw.line(screen, cfg.theme_rgb, (0, 74), (GAME_W, 74), 2)

        # Speedometer + Pedal Badge
        spd_col = (
            (255, 95, 105)
            if self.is_braking
            else ((90, 255, 205) if self.is_nitro else (255, 255, 255))
        )
        spd_txt = self.font_hud.render(f"SPEED: {int(self.speed_kmh):3d} km/h", True, spd_col)
        screen.blit(spd_txt, (16, 12))

        pedal_str = (
            "BRAKING [FIST]"
            if self.is_braking
            else ("NITRO [PINCH]" if self.is_nitro else "CRUISE [OPEN]")
        )
        screen.blit(
            self.font_small.render(
                f"{pedal_str}  |  {cfg.key} STG {self.stage}", True, cfg.theme_rgb
            ),
            (16, 43),
        )

        # Score, Distance & Combo
        score_txt = self.font_hud.render(
            f"SCORE: {int(self.score):06d}  (x{self.combo})", True, (110, 230, 255)
        )
        screen.blit(score_txt, (320, 12))
        dist_txt = self.font_small.render(
            f"BEST: {self.best_score:06d}   |   DIST: {self.distance_m / 1000.0:.2f} km   |   {int(fps):2d} FPS",
            True,
            (190, 210, 230),
        )
        screen.blit(dist_txt, (320, 43))

        # Shield integrity bars on top-right
        for i in range(3):
            col = (45, 235, 155) if i < self.shields else (55, 62, 75)
            pygame.draw.rect(screen, col, (GAME_W - 148 + i * 44, 18, 36, 16), border_radius=4)
        screen.blit(
            self.font_small.render("ARMOR", True, (185, 205, 225)),
            (GAME_W - 148, 42),
        )

        # Footer controls bar
        footer = pygame.Surface((GAME_W, 30), pygame.SRCALPHA)
        footer.fill((14, 18, 28, 222))
        screen.blit(footer, (0, GAME_H - 30))
        ctrl_line = (
            "Hand: Steer | Fist: Brake | Pinch: Nitro | "
            "[1-3] Mode [M] Menu [C] Cam [R] Retry [Q] Quit"
        )
        screen.blit(self.font_small.render(ctrl_line, True, (175, 215, 245)), (20, GAME_H - 23))

        # Interactive Easy / Medium / Hard Mode Selection Overlay
        if self.in_menu:
            veil = pygame.Surface((GAME_W, GAME_H - 104), pygame.SRCALPHA)
            veil.fill((10, 14, 24, 205))
            screen.blit(veil, (0, 74))

            title = self.font_big.render("SELECT TRAFFIC DIFFICULTY", True, (240, 248, 255))
            sub = self.font_small.render(
                "Click a Card, press [1] / [2] / [3], or Move Hand + PINCH to Start!",
                True,
                (140, 225, 255),
            )
            screen.blit(title, title.get_rect(center=(GAME_W // 2, 162)))
            screen.blit(sub, sub.get_rect(center=(GAME_W // 2, 205)))

            rects = self.mode_card_rects()
            for idx, m_key in enumerate(DIFFICULTY_ORDER):
                mc = DIFFICULTY_CONFIGS[m_key]
                r = rects[idx]
                selected = idx == self.menu_cursor_idx

                card_surf = pygame.Surface((r.width, r.height), pygame.SRCALPHA)
                card_surf.fill((22, 30, 48, 242) if selected else (16, 20, 32, 218))
                screen.blit(card_surf, r.topleft)
                pygame.draw.rect(
                    screen,
                    mc.theme_rgb if selected else (70, 88, 115),
                    r,
                    4 if selected else 2,
                    border_radius=14,
                )

                h_lbl = self.font_hud.render(mc.hotkey_label, True, mc.theme_rgb)
                l1 = self.font_small.render(mc.summary_line1, True, (235, 242, 250))
                l2 = self.font_small.render(mc.summary_line2, True, mc.theme_rgb)
                l3 = self.font_small.render(
                    f"Cruise: {int(mc.base_cruise_kmh)}+ km/h", True, (180, 198, 220)
                )
                l4 = self.font_small.render(
                    "Ramps up with Score!", True, (255, 220, 110)
                )
                best_lbl = self.font_small.render(
                    f"Best: {self.best_scores[m_key]:06d}", True, (150, 210, 235)
                )

                screen.blit(h_lbl, h_lbl.get_rect(center=(r.centerx, r.top + 34)))
                screen.blit(l1, l1.get_rect(center=(r.centerx, r.top + 78)))
                screen.blit(l2, l2.get_rect(center=(r.centerx, r.top + 106)))
                screen.blit(l3, l3.get_rect(center=(r.centerx, r.top + 142)))
                screen.blit(l4, l4.get_rect(center=(r.centerx, r.top + 168)))
                screen.blit(best_lbl, best_lbl.get_rect(center=(r.centerx, r.top + 198)))
            return

        # Game Over overlay card
        if self.game_over:
            card = pygame.Surface((600, 195), pygame.SRCALPHA)
            card.fill((14, 18, 30, 240))
            cx, cy = GAME_W // 2 - 300, GAME_H // 2 - 98
            screen.blit(card, (cx, cy))
            pygame.draw.rect(screen, (255, 85, 105), (cx, cy, 600, 195), 3, border_radius=12)

            t1 = self.font_big.render(f"CRASHED! ({cfg.key} STG {self.stage})", True, (255, 95, 110))
            t2 = self.font_hud.render(
                f"Score: {int(self.score):06d}   |   {cfg.key} Best: {self.best_score:06d}",
                True,
                (240, 245, 255),
            )
            t3 = self.font_small.render(
                "PINCH or [R] Retry  |  [1] Easy  [2] Med  [3] Hard  [M] Menu",
                True,
                (95, 255, 195),
            )
            screen.blit(t1, t1.get_rect(center=(GAME_W // 2, cy + 48)))
            screen.blit(t2, t2.get_rect(center=(GAME_W // 2, cy + 102)))
            screen.blit(t3, t3.get_rect(center=(GAME_W // 2, cy + 150)))


# ---------------------------------------------------------------------------
# Main Application Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    print("[AeroDrive 2D] Launching 60 FPS Top-Down Traffic Racer + Async Gesture Worker...")
    pygame.init()
    pygame.display.set_caption(
        "AeroDrive 2D - Top-Down Hand-Gesture Highway Traffic Racer"
    )
    screen = pygame.display.set_mode((GAME_W, GAME_H))
    clock = pygame.time.Clock()

    sprites = VehicleSpriteBank()
    game = TopDownTrafficGame(sprites)
    vision = AsyncVisionWorker(camera_index=0)

    if vision.init_error:
        print(f"[Warning] Webcam unavailable ({vision.init_error}). Running in Keyboard Mode.")

    show_cam_window = vision.init_error is None
    cam_window_positioned = False
    last_cam_seq = -1

    running = True
    try:
        while running:
            dt = min(clock.tick(TARGET_FPS) / 1000.0, 0.05)
            fps = clock.get_fps() or 60.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEMOTION and game.in_menu:
                    for idx, rect in enumerate(game.mode_card_rects()):
                        if rect.collidepoint(event.pos):
                            game.menu_cursor_idx = idx
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if game.in_menu:
                        for idx, rect in enumerate(game.mode_card_rects()):
                            if rect.collidepoint(event.pos):
                                game.select_difficulty(DIFFICULTY_ORDER[idx])
                                break
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key in (pygame.K_1, pygame.K_KP1):
                        game.select_difficulty("EASY")
                    elif event.key in (pygame.K_2, pygame.K_KP2):
                        game.select_difficulty("MEDIUM")
                    elif event.key in (pygame.K_3, pygame.K_KP3):
                        game.select_difficulty("HARD")
                    elif event.key == pygame.K_m:
                        game.open_menu()
                    elif game.in_menu and event.key in (pygame.K_LEFT, pygame.K_a):
                        game.menu_cursor_idx = (game.menu_cursor_idx - 1) % 3
                    elif game.in_menu and event.key in (pygame.K_RIGHT, pygame.K_d):
                        game.menu_cursor_idx = (game.menu_cursor_idx + 1) % 3
                    elif game.in_menu and event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        game.select_difficulty(DIFFICULTY_ORDER[game.menu_cursor_idx])
                    elif event.key == pygame.K_r:
                        game.reset()
                    elif event.key == pygame.K_c and vision.init_error is None:
                        show_cam_window = not show_cam_window
                        if not show_cam_window:
                            cv2.destroyWindow(CAM_WINDOW_TITLE)
                            cam_window_positioned = False

            pressed = pygame.key.get_pressed()
            keys_steer = float(pressed[pygame.K_RIGHT] or pressed[pygame.K_d]) - float(
                pressed[pygame.K_LEFT] or pressed[pygame.K_a]
            )
            key_brake = bool(
                pressed[pygame.K_DOWN] or pressed[pygame.K_s] or pressed[pygame.K_SPACE]
            )
            key_nitro = bool(
                pressed[pygame.K_UP] or pressed[pygame.K_w] or pressed[pygame.K_n]
            )

            ctrl_snap, cam_frame, seq = vision.get_latest()

            # Update companion OpenCV gesture monitor window only when a new camera frame arrives
            if show_cam_window and cam_frame is not None and seq != last_cam_seq:
                last_cam_seq = seq
                cv2.imshow(CAM_WINDOW_TITLE, cam_frame)
                if not cam_window_positioned:
                    cv2.moveWindow(CAM_WINDOW_TITLE, 930, 65)
                    cam_window_positioned = True
                key_cv = cv2.waitKey(1) & 0xFF
                if key_cv in (ord("q"), ord("Q"), 27):
                    running = False

            game.update(dt, ctrl_snap, keys_steer, key_brake, key_nitro)
            game.render(screen, ctrl_snap, fps)
            pygame.display.flip()
    finally:
        vision.stop()
        cv2.destroyAllWindows()
        pygame.quit()
        print("[AeroDrive 2D] Cleanly shut down.")


if __name__ == "__main__":
    main()
