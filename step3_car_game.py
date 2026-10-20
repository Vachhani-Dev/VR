"""
AeroTouch 3D - Step 3: AeroDrive 3D (AR Hand-Gesture & Optical-Flow Car Game)
Unit 5: Vision-Based AR - Pose Estimation, Pinch Scaling/Boost, Pyramidal
        Lucas-Kanade Optical Flow (`cv2.calcOpticalFlowPyrLK`), and 3D OpenGL Rendering

Gameplay & Gesture Controls:
  * MOVE HAND LEFT / RIGHT : Smoothly steer & bank your 3D OpenGL Sports Car
                             across the 3-lane AR cyber-highway!
  * PINCH (Thumb + Index)  : Activate NITRO TURBO BOOST (2x speed, 2x score
                             multiplier & twin 3D exhaust flames)!
                             (Also Pinch on Game Over to restart hands-free!)
  * QUICK UPWARD SWIPE     : Tracked via Pyramidal Lucas-Kanade Optical Flow
                             (`cv2.calcOpticalFlowPyrLK`) -> launches your 3D car
                             into a 360-degree STUNT FLIP JUMP over road barriers!
  * [G] GARAGE INSPECT MODE: Toggle close-up 3D Showroom mode where finger spread
                             distance directly SCALES & ROTATES the 3D Sports Car!

Unit 5 Rendering & Keyboard Controls:
  [W] Wireframe Rendering  |  [P] Solid Polygon Rendering  |  [L] Cycle 4 Lights
  [G] Garage Inspect Mode  |  [F] Toggle Optical Flow Trails
  [R] Restart Race         |  [Arrows / Space / N] Optional Keyboard Drive
  [Q / ESC] Quit cleanly
"""

from collections import deque
from dataclasses import dataclass
import math
import random
import time
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
import pygame
from OpenGL.GL import (
    GL_AMBIENT,
    GL_AMBIENT_AND_DIFFUSE,
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_COLOR_MATERIAL,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DIFFUSE,
    GL_FILL,
    GL_FRONT_AND_BACK,
    GL_LEQUAL,
    GL_LIGHT0,
    GL_LIGHTING,
    GL_LINE,
    GL_LINE_SMOOTH,
    GL_LINE_SMOOTH_HINT,
    GL_LINES,
    GL_LINEAR,
    GL_MODELVIEW,
    GL_NICEST,
    GL_NORMALIZE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POSITION,
    GL_PROJECTION,
    GL_QUADS,
    GL_RGB,
    GL_SHININESS,
    GL_SMOOTH,
    GL_SPECULAR,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TRIANGLES,
    GL_UNPACK_ALIGNMENT,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glBlendFunc,
    glClear,
    glClearDepth,
    glColor3f,
    glColor3fv,
    glColor4f,
    glColorMaterial,
    glDeleteTextures,
    glDepthFunc,
    glDisable,
    glEnable,
    glEnd,
    glGenTextures,
    glHint,
    glLightfv,
    glLineWidth,
    glLoadIdentity,
    glMaterialf,
    glMaterialfv,
    glMatrixMode,
    glNormal3fv,
    glOrtho,
    glPixelStorei,
    glPolygonMode,
    glPopMatrix,
    glPushMatrix,
    glRotatef,
    glScalef,
    glShadeModel,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glTexSubImage2D,
    glTranslatef,
    glVertex2f,
    glVertex3f,
    glVertex3fv,
    glViewport,
)
from OpenGL.GLU import gluPerspective
from pygame.locals import DOUBLEBUF, KEYDOWN, OPENGL, QUIT

from step1_hand_tracking import (
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_PINCH_THRESHOLD_PX,
    HandGestureData,
    HandPoseTracker,
    draw_pinch_overlay,
    open_webcam,
)
from step2_ar_renderer import LIGHTING_PRESETS, LightingPreset

# ---------------------------------------------------------------------------
# Constants & Highway Geometry Configuration
# ---------------------------------------------------------------------------
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720
FOV_DEGREES = 48.0

LANE_X = (-1.65, 0.0, 1.65)
ROAD_HALF_WIDTH = 2.65
CAR_Z = -4.6
ROAD_Y = -1.55
SPAWN_Z = -28.0
DESPAWN_Z = -1.5

# Landmarks tracked by Lucas-Kanade Optical Flow (wrist, palm & 5 fingertips)
LK_LANDMARK_IDS = (0, 4, 5, 8, 9, 12, 13, 16, 17, 20)
JUMP_FLOW_VY_THRESHOLD = -13.5  # Upward optical-flow velocity (px/frame) to trigger jump


# ---------------------------------------------------------------------------
# Pyramidal Lucas-Kanade Optical Flow Tracker (`cv2.calcOpticalFlowPyrLK`)
# ---------------------------------------------------------------------------
class LucasKanadeFlowTracker:
    """
    Computes sparse Pyramidal Lucas-Kanade Optical Flow (`cv2.calcOpticalFlowPyrLK`)
    across tracked hand landmarks and draws glowing motion vector trails.
    """

    def __init__(self, trail_length: int = 14) -> None:
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03),
        )
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.trails: List[Deque[Tuple[int, int]]] = [
            deque(maxlen=trail_length) for _ in LK_LANDMARK_IDS
        ]
        self.velocity_xy: Tuple[float, float] = (0.0, 0.0)
        self.speed_px: float = 0.0

    def update(
        self, raw_frame_bgr: np.ndarray, hands: List[HandGestureData]
    ) -> Tuple[float, float]:
        """Run `cv2.calcOpticalFlowPyrLK` between consecutive grayscale frames."""
        curr_gray = cv2.cvtColor(raw_frame_bgr, cv2.COLOR_BGR2GRAY)

        if not hands:
            self.prev_gray = curr_gray
            self.prev_pts = None
            for trail in self.trails:
                if trail:
                    trail.popleft()
            self.velocity_xy = (0.0, 0.0)
            self.speed_px = 0.0
            return self.velocity_xy

        hand = hands[0]
        anchor_pts = np.array(
            [hand.landmarks_px[idx] for idx in LK_LANDMARK_IDS], dtype=np.float32
        ).reshape(-1, 1, 2)

        vx, vy = 0.0, 0.0
        if self.prev_gray is not None and self.prev_pts is not None:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, curr_gray, self.prev_pts, None, **self.lk_params
            )
            if next_pts is not None and status is not None:
                valid_mask = status.reshape(-1) == 1
                if np.any(valid_mask):
                    deltas = (
                        next_pts[valid_mask].reshape(-1, 2)
                        - self.prev_pts[valid_mask].reshape(-1, 2)
                    )
                    vx = float(np.mean(deltas[:, 0]))
                    vy = float(np.mean(deltas[:, 1]))

                    # Blend LK-propagated points with fresh MediaPipe anchors (70/30)
                    blended = anchor_pts.copy()
                    blended[valid_mask] = (
                        0.65 * next_pts[valid_mask] + 0.35 * anchor_pts[valid_mask]
                    )
                    anchor_pts = blended

        for idx, pt in enumerate(anchor_pts.reshape(-1, 2)):
            px = int(round(float(pt[0])))
            py = int(round(float(pt[1])))
            self.trails[idx].append((px, py))

        self.prev_gray = curr_gray
        self.prev_pts = anchor_pts
        self.velocity_xy = (vx, vy)
        self.speed_px = math.hypot(vx, vy)
        return self.velocity_xy

    def draw_trails(self, frame_bgr: np.ndarray) -> None:
        """Draw glowing multi-segment motion trails & live velocity vector arrow."""
        for trail_idx, trail in enumerate(self.trails):
            pts = list(trail)
            if len(pts) < 2:
                continue
            is_fingertip = trail_idx in (1, 3)  # Thumb tip & Index tip
            for i in range(1, len(pts)):
                ratio = i / float(len(pts))
                thickness = max(1, int(round(ratio * (4 if is_fingertip else 2.5))))
                color = (
                    int(255 * (1.0 - 0.5 * ratio)),
                    int(120 + 135 * ratio),
                    int(255 * ratio),
                )
                cv2.line(frame_bgr, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

        # Draw resultant optical-flow velocity vector arrow from palm center (trail 4)
        palm_trail = self.trails[4]
        if palm_trail and self.speed_px > 2.0:
            ox, oy = palm_trail[-1]
            vx, vy = self.velocity_xy
            ex = int(round(ox + vx * 4.0))
            ey = int(round(oy + vy * 4.0))
            cv2.arrowedLine(
                frame_bgr,
                (ox, oy),
                (ex, ey),
                (0, 255, 180),
                3,
                cv2.LINE_AA,
                tipLength=0.28,
            )


# ---------------------------------------------------------------------------
# Low-Level 3D OpenGL Primitive Helpers (Sports Car, Barriers, Gems, Sparks)
# ---------------------------------------------------------------------------
def draw_unit_box(
    sx: float, sy: float, sz: float, color: Tuple[float, float, float]
) -> None:
    """Draw a lit 3D rectangular prism of half-extents `(sx, sy, sz)` with normals."""
    v = (
        (-sx, -sy,  sz),
        ( sx, -sy,  sz),
        ( sx,  sy,  sz),
        (-sx,  sy,  sz),
        (-sx, -sy, -sz),
        ( sx, -sy, -sz),
        ( sx,  sy, -sz),
        (-sx,  sy, -sz),
    )
    faces = (
        ((0, 1, 2, 3), ( 0.0,  0.0,  1.0), color),
        ((5, 4, 7, 6), ( 0.0,  0.0, -1.0), color),
        ((3, 2, 6, 7), ( 0.0,  1.0,  0.0), tuple(min(1.0, c * 1.15) for c in color)),
        ((4, 5, 1, 0), ( 0.0, -1.0,  0.0), tuple(c * 0.65 for c in color)),
        ((1, 5, 6, 2), ( 1.0,  0.0,  0.0), tuple(c * 0.88 for c in color)),
        ((4, 0, 3, 7), (-1.0,  0.0,  0.0), tuple(c * 0.88 for c in color)),
    )
    glBegin(GL_QUADS)
    for idxs, normal, face_col in faces:
        glNormal3fv(normal)
        glColor3fv(face_col)
        for idx in idxs:
            glVertex3fv(v[idx])
    glEnd()


def draw_sports_car(wheel_spin_deg: float, nitro_active: bool) -> None:
    """Render a multi-part 3D OpenGL Cyber Sports Car with spinning wheels & exhaust."""
    body_color = (0.08, 0.72, 1.00) if not nitro_active else (0.15, 0.95, 0.85)
    trim_color = (1.00, 0.25, 0.62)

    # 1. Lower aerodynamic main chassis
    glPushMatrix()
    glTranslatef(0.0, 0.20, 0.0)
    draw_unit_box(0.52, 0.14, 0.98, body_color)
    glPopMatrix()

    # 2. Neon side skirts
    for side in (-0.54, 0.54):
        glPushMatrix()
        glTranslatef(side, 0.10, 0.0)
        draw_unit_box(0.04, 0.05, 0.88, trim_color)
        glPopMatrix()

    # 3. Sleek cockpit cabin
    glPushMatrix()
    glTranslatef(0.0, 0.42, -0.05)
    draw_unit_box(0.38, 0.13, 0.48, (0.14, 0.20, 0.34))
    glPopMatrix()

    # 4. Rear racing spoiler wing + struts
    for strut_x in (-0.36, 0.36):
        glPushMatrix()
        glTranslatef(strut_x, 0.40, 0.82)
        draw_unit_box(0.04, 0.10, 0.06, trim_color)
        glPopMatrix()
    glPushMatrix()
    glTranslatef(0.0, 0.50, 0.84)
    draw_unit_box(0.50, 0.035, 0.14, trim_color)
    glPopMatrix()

    # 5. Twin glowing rear taillights & front headlights
    for lx in (-0.34, 0.34):
        glPushMatrix()
        glTranslatef(lx, 0.22, 0.99)
        draw_unit_box(0.11, 0.045, 0.02, (1.0, 0.18, 0.25))
        glPopMatrix()

        glPushMatrix()
        glTranslatef(lx, 0.21, -0.99)
        draw_unit_box(0.11, 0.045, 0.02, (0.85, 0.98, 1.0))
        glPopMatrix()

    # 6. Four rotating 3D wheels
    wheel_offsets = (
        (-0.56, 0.14, -0.62),
        ( 0.56, 0.14, -0.62),
        (-0.56, 0.14,  0.62),
        ( 0.56, 0.14,  0.62),
    )
    for wx, wy, wz in wheel_offsets:
        glPushMatrix()
        glTranslatef(wx, wy, wz)
        glRotatef(wheel_spin_deg, 1.0, 0.0, 0.0)
        draw_unit_box(0.09, 0.14, 0.14, (0.16, 0.18, 0.22))
        draw_unit_box(0.10, 0.07, 0.07, (0.85, 0.92, 1.00))
        glPopMatrix()

    # 7. Twin 3D Nitro Turbo Exhaust Flames when Pinching!
    if nitro_active:
        flicker = 0.38 + 0.16 * abs(math.sin(time.perf_counter() * 28.0))
        for fx in (-0.24, 0.24):
            glPushMatrix()
            glTranslatef(fx, 0.16, 1.02 + flicker * 0.5)
            glBegin(GL_TRIANGLES)
            glNormal3fv((0.0, 1.0, 0.0))
            glColor3f(0.20, 1.00, 1.00)
            glVertex3f(-0.11, 0.0, -flicker * 0.5)
            glVertex3f( 0.11, 0.0, -flicker * 0.5)
            glColor3f(1.00, 0.30, 0.90)
            glVertex3f( 0.00, 0.02, flicker)
            glEnd()
            glPopMatrix()


def draw_energy_crystal(spin_deg: float) -> None:
    """Render a glowing rotating 3D octahedral crystal pickup."""
    glPushMatrix()
    glRotatef(spin_deg, 0.0, 1.0, 0.0)
    glRotatef(25.0, 0.0, 0.0, 1.0)
    top = (0.0, 0.42, 0.0)
    bot = (0.0, -0.42, 0.0)
    ring = (
        (-0.30, 0.0,  0.30),
        ( 0.30, 0.0,  0.30),
        ( 0.30, 0.0, -0.30),
        (-0.30, 0.0, -0.30),
    )
    glBegin(GL_TRIANGLES)
    for i in range(4):
        a = ring[i]
        b = ring[(i + 1) % 4]
        glNormal3fv((0.0, 0.6, 0.8))
        glColor3f(0.15, 0.98, 0.72)
        glVertex3fv(top)
        glVertex3fv(a)
        glVertex3fv(b)

        glNormal3fv((0.0, -0.6, 0.8))
        glColor3f(0.10, 0.72, 1.00)
        glVertex3fv(bot)
        glVertex3fv(b)
        glVertex3fv(a)
    glEnd()
    glPopMatrix()


def draw_hazard_barrier() -> None:
    """Render a high-contrast 3D neon-crimson highway barrier block + warning stripe."""
    glPushMatrix()
    glTranslatef(0.0, 0.30, 0.0)
    draw_unit_box(0.56, 0.30, 0.30, (0.98, 0.20, 0.28))
    glTranslatef(0.0, 0.05, 0.31)
    draw_unit_box(0.46, 0.09, 0.02, (1.00, 0.86, 0.18))
    glPopMatrix()


# ---------------------------------------------------------------------------
# Game Entities & State Machine
# ---------------------------------------------------------------------------
@dataclass
class RoadEntity:
    kind: str  # "BARRIER" or "CRYSTAL"
    lane: int
    z: float
    spin: float = 0.0
    active: bool = True


@dataclass
class SparkParticle:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    color: Tuple[float, float, float]
    life: float = 0.55


class AeroDriveEngine:
    """Full 3D OpenGL Highway Racer + Garage Rubric Inspector."""

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Unit 5 Rendering states
        self.wireframe_mode: bool = False
        self.lighting_index: int = 0
        self.garage_mode: bool = False  # [G] toggles close-up 3D Scale/Rotate mode

        # Garage inspect state
        self.garage_scale: float = 1.0
        self.garage_rot_x: float = 18.0
        self.garage_rot_y: float = -32.0

        # OpenGL initialization
        glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        glClearDepth(1.0)
        glDepthFunc(GL_LEQUAL)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_NORMALIZE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(2.2)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        self.bg_texture_id = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self.bg_texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        empty_rgb = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            frame_width,
            frame_height,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            empty_rgb,
        )
        glBindTexture(GL_TEXTURE_2D, 0)

        self.best_score: int = 0
        self.reset_race()

    def reset_race(self) -> None:
        """Reset highway race state for a fresh run."""
        self.score: int = 0
        self.combo: int = 1
        self.shields: int = 3
        self.game_over: bool = False
        self.invuln_timer: float = 0.0
        self.banner_text: str = "STEER WITH YOUR HAND! PINCH = NITRO!"
        self.banner_timer: float = 2.6

        # Car kinematics
        self.car_x: float = 0.0
        self.target_car_x: float = 0.0
        self.car_roll: float = 0.0
        self.car_y_offset: float = 0.0
        self.car_vy: float = 0.0
        self.jump_flip_deg: float = 0.0
        self.is_jumping: bool = False
        self.nitro_active: bool = False
        self.speed_world: float = 11.5
        self.wheel_spin_deg: float = 0.0
        self.road_scroll: float = 0.0

        # Entities
        self.entities: List[RoadEntity] = []
        self.particles: List[SparkParticle] = []
        self.spawn_timer: float = 0.55

    @property
    def current_lighting(self) -> LightingPreset:
        return LIGHTING_PRESETS[self.lighting_index]

    @property
    def render_mode_label(self) -> str:
        return "WIREFRAME [W]" if self.wireframe_mode else "POLYGON [P]"

    def trigger_jump(self) -> None:
        """Launch 3D car into an aerial 360-degree flip over road barriers."""
        if not self.is_jumping and not self.game_over and not self.garage_mode:
            self.is_jumping = True
            self.car_vy = 6.6
            self.jump_flip_deg = 0.0
            self.banner_text = "OPTICAL FLOW STUNT JUMP!"
            self.banner_timer = 1.1

    def spawn_burst(
        self, x: float, y: float, z: float, color: Tuple[float, float, float]
    ) -> None:
        """Spawn 3D OpenGL spark cubes on crystal pickup or barrier hit."""
        for _ in range(10):
            self.particles.append(
                SparkParticle(
                    x=x,
                    y=y,
                    z=z,
                    vx=random.uniform(-2.6, 2.6),
                    vy=random.uniform(1.0, 4.2),
                    vz=random.uniform(-2.0, 2.5),
                    color=color,
                )
            )

    def spawn_wave(self) -> None:
        """Spawn fair 3-lane highway patterns (always leaves >= 1 clear lane)."""
        lanes = [0, 1, 2]
        random.shuffle(lanes)
        num_barriers = 1 if self.score < 600 else random.choice((1, 2))
        barrier_lanes = lanes[:num_barriers]
        free_lanes = lanes[num_barriers:]

        for b_lane in barrier_lanes:
            self.entities.append(RoadEntity(kind="BARRIER", lane=b_lane, z=SPAWN_Z))

        if free_lanes and random.random() < 0.80:
            c_lane = random.choice(free_lanes)
            self.entities.append(RoadEntity(kind="CRYSTAL", lane=c_lane, z=SPAWN_Z))

    def update(
        self,
        dt: float,
        hands: List[HandGestureData],
        flow_vel: Tuple[float, float],
        key_steer: float = 0.0,
        key_nitro: bool = False,
    ) -> None:
        """Update gesture mapping, car physics, collisions, and particle bursts."""
        vx, vy = flow_vel

        if self.garage_mode:
            # Close-up 3D Showroom: Hand moves rotation, finger distance scales car!
            if hands:
                hand = hands[0]
                norm_x = (hand.midpoint_px[0] / float(self.frame_width)) * 2.0 - 1.0
                norm_y = (hand.midpoint_px[1] / float(self.frame_height)) * 2.0 - 1.0
                self.garage_rot_y += (norm_x * 95.0 + vx * 3.2) * dt
                self.garage_rot_x = max(-45.0, min(55.0, 18.0 + norm_y * 35.0))
                target_scale = max(0.50, min(1.55, hand.distance_px / 95.0))
                self.garage_scale += (target_scale - self.garage_scale) * min(
                    1.0, 8.0 * dt
                )
            else:
                self.garage_rot_y = (self.garage_rot_y + 38.0 * dt) % 360.0
            self.wheel_spin_deg = (self.wheel_spin_deg + 180.0 * dt) % 360.0
            return

        # Restart race hands-free by pinching when Game Over!
        if self.game_over:
            if hands and hands[0].is_pinching:
                self.reset_race()
            return

        # 1. Gesture & Optical-Flow Control Mapping
        if hands:
            hand = hands[0]
            # Comfortable central 56% of webcam width maps across full 3-lane road
            norm_hand = (hand.midpoint_px[0] / float(self.frame_width) - 0.5) / 0.28
            norm_hand = max(-1.0, min(1.0, norm_hand))
            # Fuse MediaPipe pose with Lucas-Kanade horizontal optical flow velocity
            fused_norm = max(-1.0, min(1.0, norm_hand + vx * 0.012))
            self.target_car_x = fused_norm * 1.95
            self.nitro_active = hand.is_pinching or key_nitro

            # Upward hand flick detected by Lucas-Kanade Optical Flow -> Stunt Jump!
            if vy < JUMP_FLOW_VY_THRESHOLD:
                self.trigger_jump()
        else:
            self.target_car_x = max(
                -1.95, min(1.95, self.target_car_x + key_steer * 4.2 * dt)
            )
            self.nitro_active = key_nitro

        if abs(key_steer) > 0.01 and hands:
            self.target_car_x = max(
                -1.95, min(1.95, self.target_car_x + key_steer * 4.2 * dt)
            )

        # Smooth steering response & realistic body roll banking
        prev_x = self.car_x
        self.car_x += (self.target_car_x - self.car_x) * min(1.0, 10.5 * dt)
        lat_vel = (self.car_x - prev_x) / max(dt, 1e-4)
        target_roll = max(-22.0, min(22.0, -lat_vel * 2.6))
        self.car_roll += (target_roll - self.car_roll) * min(1.0, 12.0 * dt)

        # Jump arc & 360-degree aerial barrel flip
        if self.is_jumping:
            self.car_y_offset += self.car_vy * dt
            self.car_vy -= 16.5 * dt
            self.jump_flip_deg = min(360.0, self.jump_flip_deg + 460.0 * dt)
            if self.car_y_offset <= 0.0:
                self.car_y_offset = 0.0
                self.car_vy = 0.0
                self.jump_flip_deg = 0.0
                self.is_jumping = False

        # Dynamic highway speed (ramps up with score + doubles with Pinch Nitro!)
        difficulty_bonus = min(6.5, self.score / 450.0)
        base_speed = 11.5 + difficulty_bonus
        target_speed = base_speed * (1.75 if self.nitro_active else 1.0)
        self.speed_world += (target_speed - self.speed_world) * min(1.0, 6.0 * dt)

        self.road_scroll = (self.road_scroll + self.speed_world * dt) % 2.4
        self.wheel_spin_deg = (
            self.wheel_spin_deg + self.speed_world * 38.0 * dt
        ) % 360.0

        # Distance points accumulation
        mult = 2 if self.nitro_active else 1
        self.score += int(round(self.speed_world * dt * 2.5 * mult))
        self.best_score = max(self.best_score, self.score)

        if self.invuln_timer > 0.0:
            self.invuln_timer = max(0.0, self.invuln_timer - dt)
        if self.banner_timer > 0.0:
            self.banner_timer = max(0.0, self.banner_timer - dt)

        # 2. Spawn & Move Highway Entities
        self.spawn_timer -= dt
        if self.spawn_timer <= 0.0:
            self.spawn_wave()
            self.spawn_timer = max(0.48, 1.15 - min(0.55, self.score / 2200.0))

        for ent in self.entities:
            ent.z += self.speed_world * dt
            ent.spin = (ent.spin + 150.0 * dt) % 360.0
            if not ent.active:
                continue

            ent_x = LANE_X[ent.lane]
            dz = abs(ent.z - CAR_Z)
            dx = abs(ent_x - self.car_x)

            if dz < 0.82 and dx < 0.82:
                if ent.kind == "CRYSTAL" and self.car_y_offset < 1.1:
                    ent.active = False
                    gained = 100 * self.combo * (2 if self.nitro_active else 1)
                    self.score += gained
                    self.best_score = max(self.best_score, self.score)
                    self.combo = min(8, self.combo + 1)
                    self.banner_text = f"+{gained} CRYSTAL! COMBO x{self.combo}"
                    self.banner_timer = 1.0
                    self.spawn_burst(ent_x, ROAD_Y + 0.5, CAR_Z, (0.15, 1.0, 0.78))
                elif ent.kind == "BARRIER":
                    if self.car_y_offset > 0.62:
                        # Cleared barrier mid-air! Award Optical Flow Stunt Bonus!
                        ent.active = False
                        self.score += 250
                        self.best_score = max(self.best_score, self.score)
                        self.banner_text = "+250 AIR STUNT CLEAR!"
                        self.banner_timer = 1.1
                        self.spawn_burst(ent_x, ROAD_Y + 0.8, CAR_Z, (0.2, 0.9, 1.0))
                    elif self.invuln_timer <= 0.0:
                        ent.active = False
                        self.shields -= 1
                        self.combo = 1
                        self.invuln_timer = 1.35
                        self.spawn_burst(ent_x, ROAD_Y + 0.5, CAR_Z, (1.0, 0.25, 0.3))
                        if self.shields <= 0:
                            self.game_over = True
                            self.banner_text = "GAME OVER! PINCH OR PRESS [R] TO RACE AGAIN"
                            self.banner_timer = 999.0
                        else:
                            self.banner_text = f"SHIELD HIT! {self.shields} LEFT"
                            self.banner_timer = 1.2

        self.entities = [e for e in self.entities if e.active and e.z < DESPAWN_Z]

        # 3. Update 3D Spark Particles
        for p in self.particles:
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.z += p.vz * dt
            p.vy -= 9.5 * dt
            p.life -= dt
        self.particles = [p for p in self.particles if p.life > 0.0]

    # -----------------------------------------------------------------------
    # OpenGL Texture Upload & 3D AR Highway Rendering
    # -----------------------------------------------------------------------
    def upload_webcam_frame(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        frame_rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        glBindTexture(GL_TEXTURE_2D, self.bg_texture_id)
        if (w, h) != (self.frame_width, self.frame_height):
            self.frame_width, self.frame_height = w, h
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb
            )
        else:
            glTexSubImage2D(
                GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb
            )
        glBindTexture(GL_TEXTURE_2D, 0)

    def draw_background_quad(self) -> None:
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.bg_texture_id)
        glColor3f(1.0, 1.0, 1.0)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0)
        glVertex2f(-1.0, -1.0)
        glTexCoord2f(1.0, 1.0)
        glVertex2f(1.0, -1.0)
        glTexCoord2f(1.0, 0.0)
        glVertex2f(1.0, 1.0)
        glTexCoord2f(0.0, 0.0)
        glVertex2f(-1.0, 1.0)
        glEnd()

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)

    def apply_lighting_state(self) -> None:
        preset = self.current_lighting
        if not preset.enabled:
            glDisable(GL_LIGHTING)
            glDisable(GL_LIGHT0)
            return

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        glLightfv(GL_LIGHT0, GL_POSITION, preset.position)
        glLightfv(GL_LIGHT0, GL_AMBIENT, preset.ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, preset.diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, preset.specular)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, preset.specular)
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, preset.shininess)

    def draw_cyber_highway(self) -> None:
        """Render semi-transparent AR 3D highway deck + animated glowing lane markers."""
        glDisable(GL_LIGHTING)

        # Translucent dark cyber asphalt deck so real webcam floor subtly shines through
        glBegin(GL_QUADS)
        glColor4f(0.06, 0.08, 0.14, 0.72)
        glVertex3f(-ROAD_HALF_WIDTH, ROAD_Y, DESPAWN_Z)
        glVertex3f( ROAD_HALF_WIDTH, ROAD_Y, DESPAWN_Z)
        glColor4f(0.04, 0.12, 0.22, 0.42)
        glVertex3f( ROAD_HALF_WIDTH, ROAD_Y, SPAWN_Z)
        glVertex3f(-ROAD_HALF_WIDTH, ROAD_Y, SPAWN_Z)
        glEnd()

        # Glowing neon cyan/magenta guardrails
        glLineWidth(3.0)
        glBegin(GL_LINES)
        rail_col = (0.15, 1.0, 0.90) if self.nitro_active else (0.20, 0.78, 1.0)
        glColor3fv(rail_col)
        for side_x in (-ROAD_HALF_WIDTH, ROAD_HALF_WIDTH):
            glVertex3f(side_x, ROAD_Y + 0.02, DESPAWN_Z)
            glVertex3f(side_x, ROAD_Y + 0.02, SPAWN_Z)
        glEnd()
        glLineWidth(2.2)

        # Scrolling lane divider dashes
        glBegin(GL_QUADS)
        glColor4f(0.85, 0.95, 1.0, 0.85)
        for divider_x in (-0.82, 0.82):
            z_cursor = SPAWN_Z + self.road_scroll
            while z_cursor < DESPAWN_Z:
                z_end = min(DESPAWN_Z, z_cursor + 1.1)
                glVertex3f(divider_x - 0.04, ROAD_Y + 0.015, z_end)
                glVertex3f(divider_x + 0.04, ROAD_Y + 0.015, z_end)
                glVertex3f(divider_x + 0.04, ROAD_Y + 0.015, z_cursor)
                glVertex3f(divider_x - 0.04, ROAD_Y + 0.015, z_cursor)
                z_cursor += 2.4
        glEnd()

        self.apply_lighting_state()

    def render_scene(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.draw_background_quad()

        glClear(GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(FOV_DEGREES, WINDOW_WIDTH / float(WINDOW_HEIGHT), 0.1, 60.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        self.apply_lighting_state()

        glPolygonMode(
            GL_FRONT_AND_BACK, GL_LINE if self.wireframe_mode else GL_FILL
        )

        if self.garage_mode:
            # Close-Up AR Garage Mode (Docked on Right Side away from user's face!)
            glPushMatrix()
            glTranslatef(1.45, -0.05, -5.4)
            glScalef(self.garage_scale, self.garage_scale, self.garage_scale)
            glRotatef(self.garage_rot_x, 1.0, 0.0, 0.0)
            glRotatef(self.garage_rot_y, 0.0, 1.0, 0.0)
            draw_sports_car(self.wheel_spin_deg, nitro_active=True)
            glPopMatrix()
        else:
            # 3D AR Highway Race Mode
            self.draw_cyber_highway()

            # Obstacles & Energy Crystals
            for ent in self.entities:
                glPushMatrix()
                glTranslatef(LANE_X[ent.lane], ROAD_Y, ent.z)
                if ent.kind == "CRYSTAL":
                    glTranslatef(0.0, 0.52, 0.0)
                    draw_energy_crystal(ent.spin)
                else:
                    draw_hazard_barrier()
                glPopMatrix()

            # Player 3D Sports Car (flashes briefly while invulnerable)
            blink_skip = (
                self.invuln_timer > 0.0
                and int(self.invuln_timer * 14.0) % 2 == 1
            )
            if not blink_skip:
                glPushMatrix()
                glTranslatef(self.car_x, ROAD_Y + self.car_y_offset, CAR_Z)
                if self.nitro_active:
                    glScalef(1.04, 0.94, 1.12)  # Aero speed stance stretch
                glRotatef(self.car_roll, 0.0, 0.0, 1.0)
                glRotatef(-self.car_roll * 0.45, 0.0, 1.0, 0.0)
                if self.is_jumping:
                    glRotatef(-self.jump_flip_deg, 1.0, 0.0, 0.0)
                draw_sports_car(self.wheel_spin_deg, self.nitro_active)
                glPopMatrix()

            # 3D Spark Bursts
            for p in self.particles:
                glPushMatrix()
                glTranslatef(p.x, p.y, p.z)
                draw_unit_box(0.06, 0.06, 0.06, p.color)
                glPopMatrix()

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def close(self) -> None:
        if self.bg_texture_id:
            glDeleteTextures([int(self.bg_texture_id)])
            self.bg_texture_id = 0


# ---------------------------------------------------------------------------
# HUD Telemetry Overlay
# ---------------------------------------------------------------------------
def draw_game_hud(
    frame_bgr: np.ndarray,
    engine: AeroDriveEngine,
    hands: List[HandGestureData],
    flow_tracker: LucasKanadeFlowTracker,
    fps: float,
) -> None:
    """Render arcade race telemetry, shield hearts, steering bar & Unit 5 status."""
    frame_h, frame_w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()

    cv2.rectangle(overlay, (0, 0), (frame_w, 66), (14, 18, 26), cv2.FILLED)
    cv2.rectangle(
        overlay, (0, frame_h - 30), (frame_w, frame_h), (14, 18, 26), cv2.FILLED
    )
    cv2.addWeighted(overlay, 0.80, frame_bgr, 0.20, 0, frame_bgr)
    cv2.line(frame_bgr, (0, 66), (frame_w, 66), (70, 205, 255), 1, cv2.LINE_AA)

    if engine.garage_mode:
        cv2.putText(
            frame_bgr,
            f"GARAGE INSPECT [G]  |  Scale: {engine.garage_scale:.2f}x  |  {engine.render_mode_label}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            (95, 230, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"Spread Fingers = Scale Car  |  Move Hand = Rotate  |  Light [L]: {engine.current_lighting.name}",
            (12, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (225, 235, 245),
            1,
            cv2.LINE_AA,
        )
    else:
        speed_kmh = int(round(engine.speed_world * 14.0))
        shields_str = "O " * engine.shields + "- " * (3 - engine.shields)
        nitro_tag = "NITRO ON!" if engine.nitro_active else "Pinch=Nitro"

        cv2.putText(
            frame_bgr,
            f"SCORE: {engine.score:05d} (x{engine.combo})  |  {speed_kmh} km/h  |  Shields: {shields_str}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            (100, 255, 195) if engine.nitro_active else (95, 225, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"{nitro_tag}  |  LK Flow: {flow_tracker.speed_px:4.1f} px/f (Flick Up=Jump)  |  FPS: {fps:4.0f}",
            (12, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (235, 240, 245),
            1,
            cv2.LINE_AA,
        )

        # Live compact Steering Gauge centered right above the bottom bar
        bar_x1, bar_x2, bar_y = frame_w // 2 - 85, frame_w // 2 + 85, frame_h - 44
        cv2.rectangle(
            frame_bgr, (bar_x1, bar_y), (bar_x2, bar_y + 8), (40, 50, 68), cv2.FILLED
        )
        cv2.rectangle(
            frame_bgr, (bar_x1, bar_y), (bar_x2, bar_y + 8), (90, 180, 240), 1
        )
        steer_norm = max(0.0, min(1.0, (engine.car_x / 1.95 + 1.0) * 0.5))
        knob_x = int(round(bar_x1 + steer_norm * (bar_x2 - bar_x1)))
        cv2.circle(frame_bgr, (knob_x, bar_y + 4), 6, (0, 255, 180), cv2.FILLED)

        # Center event callout banner (Combo / Stunt / Game Over)
        if engine.banner_timer > 0.0:
            (tw, th), _ = cv2.getTextSize(
                engine.banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2
            )
            bx = max(10, (frame_w - tw) // 2)
            by = 98
            cv2.rectangle(
                frame_bgr,
                (bx - 10, by - th - 7),
                (bx + tw + 10, by + 8),
                (18, 22, 32),
                cv2.FILLED,
            )
            border_c = (70, 90, 255) if engine.game_over else (0, 240, 175)
            cv2.rectangle(
                frame_bgr,
                (bx - 10, by - th - 7),
                (bx + tw + 10, by + 8),
                border_c,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame_bgr,
                engine.banner_text,
                (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # Hand detection hint badge if no hand is visible
    if not hands and not engine.game_over:
        cv2.putText(
            frame_bgr,
            "Raise 1 Hand to Steer (or use Left/Right Arrows)",
            (145, frame_h - 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (140, 230, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame_bgr,
        f"[W/P] {engine.render_mode_label}  [L] Light  [G] Garage Scale  [F] Flow  [R] Restart  [Q] Quit",
        (10, frame_h - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.37,
        (175, 210, 240),
        1,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Main Application Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    print("[AeroDrive 3D] Initializing AR Hand-Gesture & Optical-Flow Car Game...")
    cap = open_webcam(0)
    pose_tracker = HandPoseTracker(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    flow_tracker = LucasKanadeFlowTracker(trail_length=14)

    pygame.init()
    pygame.display.set_caption(
        "AeroDrive 3D - AR Hand-Gesture & Optical-Flow Car Game ([W] Wire | [P] Poly | [L] Light | [G] Garage)"
    )
    pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL)

    engine = AeroDriveEngine(DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT)
    show_flow_trails = True
    prev_time = time.perf_counter()
    fps = 0.0

    print("[AeroDrive 3D] Ready to Race!")
    print("  * Move Hand Left / Right : Steer 3D Sports Car")
    print("  * Pinch Thumb + Index    : NITRO TURBO BOOST (2x Speed & Score)!")
    print("  * Quick Hand Flick UP    : Lucas-Kanade Optical Flow STUNT JUMP!")
    print("  * [G] Garage Inspect     : Pinch/Spread to Scale & Rotate 3D Car")
    print("  * [W] Wireframe | [P] Polygon | [L] Lighting | [R] Restart | [Q] Quit")

    running = True
    try:
        while running:
            now = time.perf_counter()
            dt = min(now - prev_time, 0.08)
            inst_fps = 1.0 / max(now - prev_time, 1e-6)
            fps = 0.85 * fps + 0.15 * inst_fps if fps > 0 else inst_fps
            prev_time = now

            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_w:
                        engine.wireframe_mode = True
                    elif event.key == pygame.K_p:
                        engine.wireframe_mode = False
                    elif event.key == pygame.K_l:
                        engine.lighting_index = (engine.lighting_index + 1) % len(
                            LIGHTING_PRESETS
                        )
                    elif event.key == pygame.K_g:
                        engine.garage_mode = not engine.garage_mode
                    elif event.key == pygame.K_f:
                        show_flow_trails = not show_flow_trails
                    elif event.key == pygame.K_r:
                        engine.reset_race()
                    elif event.key in (pygame.K_SPACE, pygame.K_UP):
                        engine.trigger_jump()

            pressed = pygame.key.get_pressed()
            key_steer = float(pressed[pygame.K_RIGHT]) - float(pressed[pygame.K_LEFT])
            key_nitro = bool(pressed[pygame.K_n])

            success, raw_frame = cap.read()
            hands: List[HandGestureData] = []
            flow_vel = (0.0, 0.0)

            if success:
                raw_frame = cv2.flip(raw_frame, 1)
                # Run MediaPipe pose estimation & Lucas-Kanade optical flow on clean camera pixels
                hands = pose_tracker.process(
                    raw_frame,
                    pinch_threshold_px=DEFAULT_PINCH_THRESHOLD_PX,
                    draw_skeleton=False,
                )
                flow_vel = flow_tracker.update(raw_frame, hands)

            engine.update(dt, hands, flow_vel, key_steer=key_steer, key_nitro=key_nitro)

            if success:
                display_frame = raw_frame
                draw_game_hud(display_frame, engine, hands, flow_tracker, fps)
                if show_flow_trails:
                    flow_tracker.draw_trails(display_frame)
                pose_tracker.draw_landmarks_on_frame(display_frame)
                for hand in hands:
                    draw_pinch_overlay(display_frame, hand)
                engine.upload_webcam_frame(display_frame)

            engine.render_scene()
            pygame.display.flip()
    finally:
        engine.close()
        pose_tracker.close()
        cap.release()
        pygame.quit()
        print("[AeroDrive 3D] Cleanly shut down.")


if __name__ == "__main__":
    main()
