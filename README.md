# AeroTouch 3D

**Vision-Based Augmented Reality, Optical Flow Tracking, 3D OpenGL Rendering & Sensor Fusion**  
*University Coursework Project — Unit 5 (Vision-Based AR, Tracking, Optical Flow, Rendering) & Unit 6 (Sensor Fusion, Simulink)*

---

## Project Roadmap

- [x] **Step 1: Hand Tracking & Background Setup (Unit 5 — Pose Estimation)** (`step1_hand_tracking.py`)
  - Real-time webcam capture (`640x480` mirrored AR view) with OpenCV (`cv2`).
  - 21-point 2D/2.5D hand pose estimation with MediaPipe (compatible out-of-the-box with both modern `mediapipe >= 0.10.31 / 1.x` Tasks API and legacy `mp.solutions.hands`).
  - Pre-bundled `models/hand_landmarker.task` model for instant **offline** execution (plus automatic CDN download fallback).
  - Euclidean pixel distance calculation $d = \sqrt{(x_8 - x_4)^2 + (y_8 - y_4)^2}$ between **Thumb Tip (`4`)** and **Index Finger Tip (`8`)** for real-time **Pinch Gesture Detection**.
- [ ] **Step 2: 3D Rendering Engine (Unit 5 — Rendering Techniques)** *(Awaiting Step 1 verification)*
- [ ] **Step 3: Gesture Integration & Optical Flow (Unit 5 — Tracking)** *(Awaiting Step 2 verification)*
- [ ] **Unit 6: Sensor Fusion & Simulink Integration** *(Upcoming)*

---

## Quick Start — Step 1

### 1. Install Dependencies

```bash
pip install opencv-python mediapipe
```
*(Or install via `pip install -r requirements.txt`)*

### 2. Run Step 1

```bash
python step1_hand_tracking.py
```

### Controls

| Key | Action |
| :--- | :--- |
| **`Q`** or **`ESC`** | Quit the application cleanly |
| **`+`** / **`=`** | Increase pinch distance threshold (`+5 px`) |
| **`-`** | Decrease pinch distance threshold (`-5 px`) |
