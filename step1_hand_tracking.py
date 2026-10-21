"""
AeroTouch 3D - Step 1: Hand Tracking & Background Setup
Unit 5: Vision-Based AR - 2D/2.5D Hand Pose Estimation & Pinch Gesture Detection

Features:
  1. Opens the laptop webcam feed using OpenCV (cv2) at 640x480 resolution.
  2. Tracks all 21 hand landmarks using MediaPipe (compatible with both modern
     MediaPipe Tasks API v0.10.31+/v1.x and legacy mp.solutions.hands API).
  3. Extracts Thumb Tip (Landmark 4) and Index Finger Tip (Landmark 8) pixel
     coordinates and computes Euclidean pixel distance to detect a "pinch".
  4. Renders the mirrored AR camera feed with full landmark skeleton, highlighted
     fingertip targets, dynamic pinch line, distance telemetry, and live FPS.

Controls:
  [Q] or [ESC] : Quit application
  [+] / [=]    : Increase pinch distance threshold (+5 px)
  [-]          : Decrease pinch distance threshold (-5 px)
"""

import math
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import List, Tuple

import runtime_bootstrap  # noqa: F401
import cv2
import mediapipe as mp

# ---------------------------------------------------------------------------
# Constants (MediaPipe Hand Landmark Indices & AR Configuration)
# ---------------------------------------------------------------------------
THUMB_TIP_ID = 4
INDEX_FINGER_TIP_ID = 8

DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480
DEFAULT_PINCH_THRESHOLD_PX = 40.0

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass
class HandGestureData:
    """Structured output for a single tracked hand in pixel coordinates."""

    handedness: str
    thumb_px: Tuple[int, int]
    index_px: Tuple[int, int]
    midpoint_px: Tuple[int, int]
    distance_px: float
    is_pinching: bool
    landmarks_px: List[Tuple[int, int]]


def ensure_hand_model() -> str:
    """
    Locate or download the official MediaPipe `hand_landmarker.task` bundle
    required by MediaPipe Tasks API (mediapipe >= 0.10.31 / 1.x).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "models", "hand_landmarker.task"),
        os.path.join(script_dir, "hand_landmarker.task"),
    ]
    for path in candidates:
        if os.path.isfile(path) and os.path.getsize(path) > 1_000_000:
            return path

    target_path = candidates[0]
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    print(f"[AeroTouch 3D] Downloading hand_landmarker.task model to {target_path} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, target_path)
    except Exception as exc:
        raise RuntimeError(
            "Failed to download MediaPipe hand_landmarker.task model.\n"
            f"Please check your internet connection or manually download:\n  {MODEL_URL}\n"
            f"and save it to:\n  {target_path}"
        ) from exc
    print("[AeroTouch 3D] Model download complete.")
    return target_path


class HandPoseTracker:
    """
    Unified MediaPipe Hand Pose Estimator supporting:
      - MediaPipe Tasks Vision API (`mp.tasks.python.vision.HandLandmarker`, v1.x)
      - MediaPipe Legacy Solutions API (`mp.solutions.hands.Hands`, v0.10.x)
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        static_image_mode: bool = False,
    ) -> None:
        self._use_tasks_api = not hasattr(mp, "solutions")
        self._static_image_mode = static_image_mode
        self._last_timestamp_ms = 0
        self._start_time = time.perf_counter()
        self._latest_raw_landmarks: List = []

        if self._use_tasks_api:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            self._mp_vision = mp_vision
            model_path = ensure_hand_model()
            running_mode = (
                mp_vision.RunningMode.IMAGE
                if static_image_mode
                else mp_vision.RunningMode.VIDEO
            )
            options = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=running_mode,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        else:
            self._mp_hands = mp.solutions.hands
            self._mp_draw = mp.solutions.drawing_utils
            self._mp_styles = mp.solutions.drawing_styles
            self._hands = self._mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

    def process(
        self,
        frame_bgr,
        pinch_threshold_px: float = DEFAULT_PINCH_THRESHOLD_PX,
        draw_skeleton: bool = True,
    ) -> List[HandGestureData]:
        """
        Detect hand landmarks on a BGR frame, optionally draw the 21-point hand
        skeleton, and compute thumb-to-index pixel distance & pinch state.
        """
        frame_h, frame_w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detected_hands: List[HandGestureData] = []
        self._latest_raw_landmarks = []

        if self._use_tasks_api:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            if self._static_image_mode:
                result = self._landmarker.detect(mp_image)
            else:
                elapsed_ms = int((time.perf_counter() - self._start_time) * 1000)
                timestamp_ms = max(self._last_timestamp_ms + 1, elapsed_ms)
                self._last_timestamp_ms = timestamp_ms
                result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

            self._latest_raw_landmarks = list(result.hand_landmarks)

            for idx, hand_landmarks in enumerate(result.hand_landmarks):
                label = "Hand"
                if idx < len(result.handedness) and result.handedness[idx]:
                    label = result.handedness[idx][0].category_name

                detected_hands.append(
                    self._extract_gesture_data(
                        hand_landmarks, label, frame_w, frame_h, pinch_threshold_px
                    )
                )
        else:
            results = self._hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                self._latest_raw_landmarks = list(results.multi_hand_landmarks)
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    label = "Hand"
                    if results.multi_handedness and idx < len(results.multi_handedness):
                        label = results.multi_handedness[idx].classification[0].label

                    detected_hands.append(
                        self._extract_gesture_data(
                            hand_landmarks.landmark,
                            label,
                            frame_w,
                            frame_h,
                            pinch_threshold_px,
                        )
                    )

        if draw_skeleton:
            self.draw_landmarks_on_frame(frame_bgr)

        return detected_hands

    def draw_landmarks_on_frame(self, frame_bgr) -> None:
        """Draw the 21 MediaPipe hand landmarks and connections onto `frame_bgr`."""
        for hand_landmarks in self._latest_raw_landmarks:
            if self._use_tasks_api:
                self._mp_vision.drawing_utils.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    self._mp_vision.HandLandmarksConnections.HAND_CONNECTIONS,
                    self._mp_vision.drawing_styles.get_default_hand_landmarks_style(),
                    self._mp_vision.drawing_styles.get_default_hand_connections_style(),
                )
            else:
                self._mp_draw.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_styles.get_default_hand_landmarks_style(),
                    self._mp_styles.get_default_hand_connections_style(),
                )

    @staticmethod
    def _extract_gesture_data(
        landmarks,
        handedness: str,
        frame_w: int,
        frame_h: int,
        pinch_threshold_px: float,
    ) -> HandGestureData:
        """Convert normalized landmarks [0, 1] to pixel space & evaluate pinch."""
        landmarks_px = [
            (int(lm.x * frame_w), int(lm.y * frame_h)) for lm in landmarks
        ]

        thumb_x, thumb_y = landmarks_px[THUMB_TIP_ID]
        index_x, index_y = landmarks_px[INDEX_FINGER_TIP_ID]
        mid_x = (thumb_x + index_x) // 2
        mid_y = (thumb_y + index_y) // 2

        # Euclidean distance in pixel space between Thumb Tip (4) & Index Tip (8)
        distance_px = math.hypot(index_x - thumb_x, index_y - thumb_y)
        is_pinching = distance_px < pinch_threshold_px

        return HandGestureData(
            handedness=handedness,
            thumb_px=(thumb_x, thumb_y),
            index_px=(index_x, index_y),
            midpoint_px=(mid_x, mid_y),
            distance_px=distance_px,
            is_pinching=is_pinching,
            landmarks_px=landmarks_px,
        )

    def close(self) -> None:
        """Release underlying MediaPipe resources cleanly."""
        if self._use_tasks_api:
            self._landmarker.close()
        else:
            self._hands.close()


def draw_badge(
    frame_bgr,
    text: str,
    pos: Tuple[int, int],
    scale: float = 0.44,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    border_color: Tuple[int, int, int] = (80, 190, 255),
) -> Tuple[int, int, int, int]:
    """Draw a sharp single-pass label pill that never clips HUD bars or edges."""
    frame_h, frame_w = frame_bgr.shape[:2]
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
    )
    pad_x, pad_y = 6, 4
    box_w = tw + 2 * pad_x
    box_h = th + baseline + 2 * pad_y

    x1 = max(6, min(pos[0], frame_w - box_w - 6))
    y1 = max(70, min(pos[1] - th - pad_y, frame_h - 34 - box_h))
    x2, y2 = x1 + box_w, y1 + box_h

    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (18, 22, 28), cv2.FILLED)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)
    cv2.putText(
        frame_bgr,
        text,
        (x1 + pad_x, y1 + pad_y + th),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        text_color,
        1,
        cv2.LINE_AA,
    )
    return x1, y1, x2, y2


def draw_pinch_overlay(frame_bgr, hand: HandGestureData) -> None:
    """Highlight Thumb Tip (4), Index Finger Tip (8), and dynamic pinch vector."""
    tx, ty = hand.thumb_px
    ix, iy = hand.index_px
    mx, my = hand.midpoint_px

    active_color = (0, 255, 100) if hand.is_pinching else (255, 220, 0)

    # Line connecting Thumb Tip (4) and Index Finger Tip (8)
    cv2.line(frame_bgr, (tx, ty), (ix, iy), active_color, 3, cv2.LINE_AA)

    # Highlighted fingertip rings
    cv2.circle(frame_bgr, (tx, ty), 9, (255, 0, 255), cv2.FILLED, cv2.LINE_AA)
    cv2.circle(frame_bgr, (tx, ty), 12, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.circle(frame_bgr, (ix, iy), 9, (0, 200, 255), cv2.FILLED, cv2.LINE_AA)
    cv2.circle(frame_bgr, (ix, iy), 12, (255, 255, 255), 2, cv2.LINE_AA)

    # Midpoint indicator (solid green pulse on pinch, hollow ring otherwise)
    if hand.is_pinching:
        cv2.circle(frame_bgr, (mx, my), 11, (0, 255, 100), cv2.FILLED, cv2.LINE_AA)
        cv2.circle(frame_bgr, (mx, my), 16, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.circle(frame_bgr, (mx, my), 6, active_color, 2, cv2.LINE_AA)

    # Non-overlapping pill badges near Index Tip [8] & Thumb Tip [4]
    _, iy1, _, iy2 = draw_badge(
        frame_bgr,
        f"Index [8]: ({ix}, {iy})",
        (ix + 16, iy - 6),
        text_color=(160, 240, 255),
        border_color=(0, 200, 255),
    )
    thumb_y = ty + 18
    if abs(thumb_y - (iy1 + iy2) // 2) < 26:
        thumb_y = iy2 + 20
    draw_badge(
        frame_bgr,
        f"Thumb [4]: ({tx}, {ty})",
        (tx + 16, thumb_y),
        text_color=(255, 185, 255),
        border_color=(255, 0, 255),
    )


def draw_hud(
    frame_bgr,
    hands: List[HandGestureData],
    pinch_threshold_px: float,
    fps: float,
) -> None:
    """Draw sleek top & bottom AR telemetry bars without obstructing center view."""
    frame_h, frame_w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()

    # Sleek translucent top header bar + subtle bottom controls bar
    cv2.rectangle(overlay, (0, 0), (frame_w, 64), (18, 22, 28), cv2.FILLED)
    cv2.rectangle(
        overlay, (0, frame_h - 28), (frame_w, frame_h), (18, 22, 28), cv2.FILLED
    )
    cv2.addWeighted(overlay, 0.78, frame_bgr, 0.22, 0, frame_bgr)
    cv2.line(frame_bgr, (0, 64), (frame_w, 64), (80, 190, 255), 1, cv2.LINE_AA)

    cv2.putText(
        frame_bgr,
        "AeroTouch 3D  |  Step 1: Hand Pose & Pinch Tracking",
        (14, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (90, 215, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"FPS: {fps:4.1f}",
        (frame_w - 105, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (180, 230, 255),
        1,
        cv2.LINE_AA,
    )

    if hands:
        primary = hands[0]
        state_text = "PINCH DETECTED!" if primary.is_pinching else "OPEN HAND"
        state_color = (0, 255, 110) if primary.is_pinching else (255, 255, 255)

        cv2.putText(
            frame_bgr,
            f"Gesture: {state_text}   |   Pinch Distance: {primary.distance_px:5.1f} px   (Threshold: {pinch_threshold_px:.0f} px)",
            (14, 51),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            state_color,
            2 if primary.is_pinching else 1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame_bgr,
            f"Gesture: Waiting for hand...   |   Threshold: {pinch_threshold_px:.0f} px",
            (14, 51),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            (185, 195, 205),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame_bgr,
        "Controls:  [+/=] Increase Threshold    [-] Decrease Threshold    [Q / ESC] Quit",
        (14, frame_h - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (175, 205, 235),
        1,
        cv2.LINE_AA,
    )


def open_webcam(camera_index: int = 0) -> cv2.VideoCapture:
    """Open webcam cleanly across Windows, macOS, and Linux."""
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {camera_index}. "
            "Check camera permissions and ensure no other app is locking the webcam."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_FRAME_HEIGHT)
    return cap


def main() -> None:
    print("[AeroTouch 3D] Initializing Step 1: Hand Tracking & Pinch Detection...")
    tracker = HandPoseTracker(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    cap = open_webcam(0)

    pinch_threshold_px = DEFAULT_PINCH_THRESHOLD_PX
    prev_time = time.perf_counter()
    fps = 0.0

    print("[AeroTouch 3D] Webcam active! Show your hand to track landmarks.")
    print("  -> Bring Thumb Tip & Index Finger Tip closer than 40 px to trigger PINCH.")
    print("  -> Press 'Q' or 'ESC' inside the video window to exit.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("[Warning] Empty webcam frame skipped.")
                continue

            # Mirror frame horizontally for intuitive AR selfie-view interaction
            frame = cv2.flip(frame, 1)

            # Detect hand landmarks & evaluate pinch gesture (draw skeleton after HUD)
            hands = tracker.process(
                frame, pinch_threshold_px=pinch_threshold_px, draw_skeleton=False
            )

            # Smoothed FPS calculation
            now = time.perf_counter()
            inst_fps = 1.0 / max(now - prev_time, 1e-6)
            fps = 0.85 * fps + 0.15 * inst_fps if fps > 0 else inst_fps
            prev_time = now

            draw_hud(frame, hands, pinch_threshold_px, fps)

            # Re-draw full skeleton + highlighted pinch overlay cleanly above HUD bars
            tracker.draw_landmarks_on_frame(frame)
            for hand in hands:
                draw_pinch_overlay(frame, hand)
            cv2.imshow("AeroTouch 3D - Step 1: Hand Tracking & Pinch Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                break
            if key in (ord("+"), ord("=")):
                pinch_threshold_px = min(150.0, pinch_threshold_px + 5.0)
            elif key in (ord("-"), ord("_")):
                pinch_threshold_px = max(15.0, pinch_threshold_px - 5.0)
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
        print("[AeroTouch 3D] Cleanly shut down.")


if __name__ == "__main__":
    main()
