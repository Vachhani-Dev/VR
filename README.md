# AeroTouch 3D

**Vision-Based Augmented Reality, Optical Flow Tracking, 3D OpenGL Rendering & Sensor Fusion**  
*University Coursework Project — Unit 5 (Vision-Based AR, Tracking, Optical Flow, Rendering) & Unit 6 (Sensor Fusion, Simulink)*

---

## Project Roadmap

- [x] **Step 1: Hand Tracking & Background Setup (Unit 5 — Pose Estimation)** (`step1_hand_tracking.py`)
  - Real-time webcam capture (`640x480` mirrored AR view) with OpenCV (`cv2`).
  - 21-point 2D/2.5D hand pose estimation with MediaPipe (`mediapipe.tasks` v1.x + legacy fallback) with offline `models/hand_landmarker.task` bundle.
  - Euclidean pixel distance calculation $d = \sqrt{(x_8 - x_4)^2 + (y_8 - y_4)^2}$ between **Thumb Tip (`4`)** and **Index Finger Tip (`8`)** for **Pinch Gesture Detection**.
- [x] **Step 2: 3D Rendering Engine (Unit 5 — Rendering Techniques)** (`step2_ar_renderer.py`)
  - Real-time GPU 2D background texture streaming (`GL_TEXTURE_2D` + `glTexSubImage2D`) via `pygame` (`pygame-ce`) & `PyOpenGL`.
  - Perspective 3D **Cube** & **Square Pyramid** (`[S]` toggle) floating in the center of the AR camera view with outward surface normals & depth testing (`GL_DEPTH_TEST`).
  - Real-time rasterizer & shading controls: **`[W]`** Wireframe (`GL_LINE`), **`[P]`** Solid Polygon (`GL_FILL`), and **`[L]`** Dynamic Lighting Presets (`Studio Phong`, `Cyber Cyan`, `Sunset Warm Rim`, `Unlit Flat OFF`).
- [ ] **Step 3: Gesture Integration & Optical Flow (Unit 5 — Tracking)** *(Awaiting Step 2 verification)*
- [ ] **Unit 6: Sensor Fusion & Simulink Integration** *(Upcoming)*

---

## Quick Start

### Install All Dependencies

```powershell
python -m pip install opencv-python mediapipe pygame-ce PyOpenGL
```
*(Or `python -m pip install -r requirements.txt`)*

### Run Step 1 (Hand Pose & Pinch Detection)

```powershell
python step1_hand_tracking.py
```

### Run Step 2 (OpenGL 3D AR Rendering Engine)

```powershell
python step2_ar_renderer.py
```

### Step 2 Keyboard Controls

| Key | Rendering Technique / Action |
| :--- | :--- |
| **`W`** | Switch to **Wireframe Rendering** (`glPolygonMode(..., GL_LINE)`) |
| **`P`** | Switch to **Polygon / Solid Rendering** (`glPolygonMode(..., GL_FILL)`) |
| **`L`** | Cycle **OpenGL Lighting Modes** (`Studio Key`, `Cyber Neon`, `Sunset Rim`, `Unlit Flat OFF`) |
| **`S`** | Switch 3D Geometry between **Cube** and **Pyramid** |
| **`SPACE`** | Pause / Resume 3D auto-rotation |
| **`H`** | Show / Hide 2D hand landmark skeleton overlay |
| **`Q`** or **`ESC`** | Quit the application cleanly |
