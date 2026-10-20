"""
AeroTouch 3D - Step 2: 3D Rendering Engine
Unit 5: Vision-Based AR - OpenGL Background Texture, 3D Geometry, Wireframe/Polygon
        Rasterization & Dynamic Phong/Gouraud Lighting

Features:
  1. Streams real-time OpenCV webcam frames (with live Step 1 MediaPipe hand tracking)
     directly into a GPU 2D OpenGL texture (`GL_TEXTURE_2D` + `glTexSubImage2D`).
  2. Renders a 3D rotating Cube or Pyramid floating in perspective space (`z = -5.0`)
     in the center of the AR viewport with accurate surface normals & depth testing.
  3. Interactive Unit 5 Rendering Toggles:
       [W]     : Switch to Wireframe Rendering (`glPolygonMode(..., GL_LINE)`)
       [P]     : Switch to Polygon / Solid Rendering (`glPolygonMode(..., GL_FILL)`)
       [L]     : Cycle OpenGL Lighting Presets (Studio White -> Cyber Cyan ->
                 Sunset Amber -> Unlit Flat Shading OFF)
       [S]     : Switch 3D Geometry between Cube and Pyramid
       [SPACE] : Pause / Resume 3D rotation
       [H]     : Show / Hide 2D hand landmark skeleton overlay
       [Q/ESC] : Quit cleanly
"""

import time
from dataclasses import dataclass
from typing import List, Tuple

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
    glShadeModel,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glTexSubImage2D,
    glTranslatef,
    glVertex2f,
    glVertex3fv,
    glViewport,
)
from OpenGL.GLU import gluPerspective
from pygame.locals import DOUBLEBUF, K_ESCAPE, K_SPACE, KEYDOWN, OPENGL, QUIT

from step1_hand_tracking import (
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_PINCH_THRESHOLD_PX,
    HandGestureData,
    HandPoseTracker,
    draw_pinch_overlay,
    open_webcam,
)

# ---------------------------------------------------------------------------
# Window & OpenGL Configuration
# ---------------------------------------------------------------------------
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720
FOV_DEGREES = 45.0


@dataclass(frozen=True)
class LightingPreset:
    """Configuration for a selectable OpenGL illumination mode (`L` key)."""

    name: str
    enabled: bool
    position: Tuple[float, float, float, float]
    ambient: Tuple[float, float, float, float]
    diffuse: Tuple[float, float, float, float]
    specular: Tuple[float, float, float, float]
    shininess: float


LIGHTING_PRESETS: List[LightingPreset] = [
    LightingPreset(
        name="Studio Key (Phong)",
        enabled=True,
        position=(3.5, 4.0, 5.0, 1.0),
        ambient=(0.28, 0.28, 0.32, 1.0),
        diffuse=(1.00, 0.98, 0.94, 1.0),
        specular=(1.00, 1.00, 1.00, 1.0),
        shininess=64.0,
    ),
    LightingPreset(
        name="Cyber Neon (Cyan)",
        enabled=True,
        position=(-4.0, 3.0, 4.5, 1.0),
        ambient=(0.12, 0.24, 0.34, 1.0),
        diffuse=(0.15, 0.95, 1.00, 1.0),
        specular=(0.80, 1.00, 1.00, 1.0),
        shininess=90.0,
    ),
    LightingPreset(
        name="Sunset Rim (Warm)",
        enabled=True,
        position=(5.0, 1.2, 2.0, 1.0),
        ambient=(0.22, 0.14, 0.10, 1.0),
        diffuse=(1.00, 0.55, 0.18, 1.0),
        specular=(1.00, 0.85, 0.55, 1.0),
        shininess=48.0,
    ),
    LightingPreset(
        name="Unlit Flat (OFF)",
        enabled=False,
        position=(0.0, 0.0, 1.0, 0.0),
        ambient=(1.0, 1.0, 1.0, 1.0),
        diffuse=(1.0, 1.0, 1.0, 1.0),
        specular=(0.0, 0.0, 0.0, 1.0),
        shininess=0.0,
    ),
]


# ---------------------------------------------------------------------------
# 3D Geometry Definitions (Cube & Square Pyramid with Outward Unit Normals)
# ---------------------------------------------------------------------------
CUBE_VERTICES = (
    (-1.0, -1.0,  1.0),  # 0: Front-Bottom-Left
    ( 1.0, -1.0,  1.0),  # 1: Front-Bottom-Right
    ( 1.0,  1.0,  1.0),  # 2: Front-Top-Right
    (-1.0,  1.0,  1.0),  # 3: Front-Top-Left
    (-1.0, -1.0, -1.0),  # 4: Back-Bottom-Left
    ( 1.0, -1.0, -1.0),  # 5: Back-Bottom-Right
    ( 1.0,  1.0, -1.0),  # 6: Back-Top-Right
    (-1.0,  1.0, -1.0),  # 7: Back-Top-Left
)

# Each tuple: (vertex_indices, unit_normal, rgb_face_color)
CUBE_FACES = (
    ((0, 1, 2, 3), ( 0.0,  0.0,  1.0), (0.15, 0.78, 1.00)),  # Front  - Cyan
    ((5, 4, 7, 6), ( 0.0,  0.0, -1.0), (1.00, 0.32, 0.65)),  # Back   - Magenta
    ((3, 2, 6, 7), ( 0.0,  1.0,  0.0), (0.30, 0.95, 0.55)),  # Top    - Mint Green
    ((4, 5, 1, 0), ( 0.0, -1.0,  0.0), (1.00, 0.72, 0.18)),  # Bottom - Amber
    ((1, 5, 6, 2), ( 1.0,  0.0,  0.0), (0.58, 0.42, 1.00)),  # Right  - Violet
    ((4, 0, 3, 7), (-1.0,  0.0,  0.0), (1.00, 0.45, 0.32)),  # Left   - Coral
)

PYRAMID_APEX = (0.0, 1.25, 0.0)
PYRAMID_BASE = (
    (-1.05, -0.95,  1.05),  # 0: Front-Left
    ( 1.05, -0.95,  1.05),  # 1: Front-Right
    ( 1.05, -0.95, -1.05),  # 2: Back-Right
    (-1.05, -0.95, -1.05),  # 3: Back-Left
)

PYRAMID_SIDES = (
    ((PYRAMID_APEX, PYRAMID_BASE[0], PYRAMID_BASE[1]), ( 0.000, 0.431,  0.902), (0.18, 0.82, 1.00)),
    ((PYRAMID_APEX, PYRAMID_BASE[1], PYRAMID_BASE[2]), ( 0.902, 0.431,  0.000), (0.62, 0.42, 1.00)),
    ((PYRAMID_APEX, PYRAMID_BASE[2], PYRAMID_BASE[3]), ( 0.000, 0.431, -0.902), (1.00, 0.35, 0.68)),
    ((PYRAMID_APEX, PYRAMID_BASE[3], PYRAMID_BASE[0]), (-0.902, 0.431,  0.000), (0.30, 0.95, 0.58)),
)


def draw_cube() -> None:
    """Render a 3D Cube centered at the origin with per-face normals & colors."""
    glBegin(GL_QUADS)
    for indices, normal, color in CUBE_FACES:
        glNormal3fv(normal)
        glColor3fv(color)
        for vertex_idx in indices:
            glVertex3fv(CUBE_VERTICES[vertex_idx])
    glEnd()


def draw_pyramid() -> None:
    """Render a 3D Square Pyramid centered at the origin with unit normals."""
    glBegin(GL_TRIANGLES)
    for vertices, normal, color in PYRAMID_SIDES:
        glNormal3fv(normal)
        glColor3fv(color)
        for vertex in vertices:
            glVertex3fv(vertex)
    glEnd()

    # Bottom square cap
    glBegin(GL_QUADS)
    glNormal3fv((0.0, -1.0, 0.0))
    glColor3fv((1.00, 0.72, 0.20))
    for vertex in reversed(PYRAMID_BASE):
        glVertex3fv(vertex)
    glEnd()


# ---------------------------------------------------------------------------
# OpenGL Renderer (2D Webcam Texture + 3D AR Scene)
# ---------------------------------------------------------------------------
class AeroTouchRenderer:
    """Manages OpenGL state, live webcam texture streaming, and 3D AR rendering."""

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Interactive rendering states (Unit 5 controls)
        self.wireframe_mode: bool = False          # [W] = True, [P] = False
        self.lighting_index: int = 0               # [L] cycles presets
        self.shape_mode: str = "CUBE"              # [S] toggles CUBE <-> PYRAMID
        self.auto_rotate: bool = True              # [SPACE] pauses/resumes

        # Current 3D orientation (degrees)
        self.rot_x: float = 22.0
        self.rot_y: float = 35.0
        self.rot_z: float = 0.0

        # Configure baseline OpenGL state
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

        # Allocate persistent GPU 2D texture for webcam background frames
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

    @property
    def current_lighting(self) -> LightingPreset:
        return LIGHTING_PRESETS[self.lighting_index]

    @property
    def render_mode_label(self) -> str:
        return "WIREFRAME [W]" if self.wireframe_mode else "POLYGON SOLID [P]"

    def update_rotation(self, dt: float) -> None:
        """Advance smooth frame-rate-independent rotation angles."""
        if self.auto_rotate:
            self.rot_x = (self.rot_x + 28.0 * dt) % 360.0
            self.rot_y = (self.rot_y + 46.0 * dt) % 360.0
            self.rot_z = (self.rot_z + 14.0 * dt) % 360.0

    def upload_webcam_frame(self, frame_bgr: np.ndarray) -> None:
        """Stream BGR OpenCV frame into the existing GPU 2D texture with zero reallocation."""
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
        """Render the 2D webcam texture across the full viewport behind 3D geometry."""
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

        # Map OpenCV top-left image origin (0,0) -> OpenGL top-left screen (-1,+1)
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
        """Configure OpenGL lighting parameters according to active `LightingPreset`."""
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

    def render_scene(self) -> None:
        """Execute full AR frame pass: 2D camera background -> 3D lit geometry."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Pass 1: 2D Webcam AR Background
        self.draw_background_quad()

        # Pass 2: 3D Perspective Object in Center of Screen
        glClear(GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(FOV_DEGREES, WINDOW_WIDTH / float(WINDOW_HEIGHT), 0.1, 50.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        self.apply_lighting_state()

        glTranslatef(0.0, 0.0, -5.2)
        glRotatef(self.rot_x, 1.0, 0.0, 0.0)
        glRotatef(self.rot_y, 0.0, 1.0, 0.0)
        glRotatef(self.rot_z, 0.0, 0.0, 1.0)

        glPolygonMode(
            GL_FRONT_AND_BACK, GL_LINE if self.wireframe_mode else GL_FILL
        )

        if self.shape_mode == "CUBE":
            draw_cube()
        else:
            draw_pyramid()

        # Always restore solid fill mode after 3D pass
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def close(self) -> None:
        """Delete allocated GPU texture cleanly."""
        if self.bg_texture_id:
            glDeleteTextures([int(self.bg_texture_id)])
            self.bg_texture_id = 0


# ---------------------------------------------------------------------------
# 2D Telemetry Overlay on Webcam Texture
# ---------------------------------------------------------------------------
def draw_step2_hud(
    frame_bgr: np.ndarray,
    renderer: AeroTouchRenderer,
    hands: List[HandGestureData],
    show_skeleton: bool,
    fps: float,
) -> None:
    """Draw AR status HUD onto webcam frame prior to GPU texture upload."""
    frame_h, frame_w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()

    cv2.rectangle(overlay, (0, 0), (frame_w, 66), (16, 20, 28), cv2.FILLED)
    cv2.rectangle(
        overlay, (0, frame_h - 28), (frame_w, frame_h), (16, 20, 28), cv2.FILLED
    )
    cv2.addWeighted(overlay, 0.78, frame_bgr, 0.22, 0, frame_bgr)
    cv2.line(frame_bgr, (0, 66), (frame_w, 66), (80, 195, 255), 1, cv2.LINE_AA)

    cv2.putText(
        frame_bgr,
        f"AeroTouch 3D | Step 2: {renderer.shape_mode}  |  {renderer.render_mode_label}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (90, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"FPS: {fps:4.1f}",
        (frame_w - 102, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (185, 235, 255),
        1,
        cv2.LINE_AA,
    )

    hand_status = "No Hand"
    if hands:
        hand_status = (
            f"PINCH ({hands[0].distance_px:.0f}px)"
            if hands[0].is_pinching
            else f"OPEN ({hands[0].distance_px:.0f}px)"
        )

    cv2.putText(
        frame_bgr,
        f"Light [L]: {renderer.current_lighting.name}   |   Hand: {hand_status}",
        (12, 51),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (235, 240, 245),
        1,
        cv2.LINE_AA,
    )

    skel_tag = "ON" if show_skeleton else "OFF"
    cv2.putText(
        frame_bgr,
        f"[W] Wireframe  [P] Polygon  [L] Light  [S] Shape  [H] Hand:{skel_tag}  [Space] Pause  [Q] Quit",
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
    print("[AeroTouch 3D] Initializing Step 2: OpenGL 3D Rendering Engine...")
    cap = open_webcam(0)
    tracker = HandPoseTracker(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    pygame.init()
    pygame.display.set_caption(
        "AeroTouch 3D - Step 2: OpenGL AR Rendering Engine ([W] Wireframe | [P] Polygon | [L] Light | [S] Shape)"
    )
    pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL)

    renderer = AeroTouchRenderer(DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT)
    show_skeleton = True
    prev_time = time.perf_counter()
    fps = 0.0

    print("[AeroTouch 3D] OpenGL AR Engine running!")
    print("  [W] Wireframe Mode   |  [P] Solid Polygon Mode")
    print("  [L] Cycle Lighting   |  [S] Switch Cube / Pyramid")
    print("  [SPACE] Pause Spin   |  [H] Toggle Hand Skeleton   |  [Q / ESC] Quit")

    running = True
    try:
        while running:
            now = time.perf_counter()
            dt = min(now - prev_time, 0.1)
            inst_fps = 1.0 / max(now - prev_time, 1e-6)
            fps = 0.85 * fps + 0.15 * inst_fps if fps > 0 else inst_fps
            prev_time = now

            # 1. Handle Keyboard & Window Events
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key in (K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_w:
                        renderer.wireframe_mode = True
                        print("[Render Mode] Switched to WIREFRAME rendering (GL_LINE)")
                    elif event.key == pygame.K_p:
                        renderer.wireframe_mode = False
                        print("[Render Mode] Switched to POLYGON SOLID rendering (GL_FILL)")
                    elif event.key == pygame.K_l:
                        renderer.lighting_index = (renderer.lighting_index + 1) % len(
                            LIGHTING_PRESETS
                        )
                        print(f"[Lighting] Switched to: {renderer.current_lighting.name}")
                    elif event.key == pygame.K_s:
                        renderer.shape_mode = (
                            "PYRAMID" if renderer.shape_mode == "CUBE" else "CUBE"
                        )
                        print(f"[3D Shape] Switched to: {renderer.shape_mode}")
                    elif event.key == pygame.K_h:
                        show_skeleton = not show_skeleton
                    elif event.key == K_SPACE:
                        renderer.auto_rotate = not renderer.auto_rotate

            # 2. Capture Webcam Frame + Track Hand Pose
            success, frame = cap.read()
            if success:
                frame = cv2.flip(frame, 1)
                hands = tracker.process(
                    frame,
                    pinch_threshold_px=DEFAULT_PINCH_THRESHOLD_PX,
                    draw_skeleton=False,
                )
                draw_step2_hud(frame, renderer, hands, show_skeleton, fps)
                if show_skeleton:
                    tracker.draw_landmarks_on_frame(frame)
                    for hand in hands:
                        draw_pinch_overlay(frame, hand)
                renderer.upload_webcam_frame(frame)

            # 3. Update & Render OpenGL 3D Scene Over Live Camera Background
            renderer.update_rotation(dt)
            renderer.render_scene()
            pygame.display.flip()
    finally:
        renderer.close()
        tracker.close()
        cap.release()
        pygame.quit()
        print("[AeroTouch 3D] Step 2 cleanly shut down.")


if __name__ == "__main__":
    main()
