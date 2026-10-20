# AeroTouch 3D — AeroDrive 3D Edition

**Vision-Based AR Hand-Gesture & Optical-Flow 3D OpenGL Car Game + Sensor Fusion**  
*University Coursework Project — Unit 5 (Vision-Based AR, Tracking, Optical Flow, Rendering) & Unit 6 (Sensor Fusion, Simulink)*

---

## Project Roadmap

- [x] **Step 1: Hand Tracking & Pinch Detection (`step1_hand_tracking.py`)**
  - Real-time webcam capture (`640x480` mirrored AR view) + 21-landmark MediaPipe hand pose estimation + offline `models/hand_landmarker.task` bundle + Euclidean pinch detection.
- [x] **Step 2: 3D Rendering Engine (`step2_ar_renderer.py`)**
  - Live OpenGL `GL_TEXTURE_2D` camera background + side-docked 3D **Cube** & **Pyramid** + **`[W]`** Wireframe, **`[P]`** Solid Polygon, **`[L]`** 4-Preset Phong Lighting, and **`[D]` / Arrow** positioning.
- [x] **Step 3: AeroDrive 3D — Hand-Gesture & Optical-Flow Car Game (`step3_car_game.py`)**
  - Playable 3D OpenGL AR Highway Racer directly over your live webcam feed (keeping your face & upper body completely clear!).
  - **Hand Pose Steering:** Move hand Left / Right to steer & roll-bank a multi-part 3D Sports Car (chassis, cabin, spoiler, spinning 3D wheels, headlights & Nitro flames).
  - **Pinch Gesture (`Thumb [4]` ↔ `Index [8]`):** Activates **NITRO TURBO BOOST** (2x speed, 2x score multiplier & twin 3D exhaust flames) — and hands-free race restart on Game Over!
  - **Pyramidal Lucas-Kanade Optical Flow (`cv2.calcOpticalFlowPyrLK`):** Draws glowing neon motion trails + velocity arrows on your hand AND launches your car into a **360° Stunt Flip Jump** over road barriers on a quick upward flick!
  - **`[G]` Garage Showroom Mode:** Close-up 3D inspection mode where finger spread distance **scales** and hand motion **rotates** the 3D car (with live **`[W]`**, **`[P]`**, **`[L]`** toggles).
- [ ] **Step 4 (Unit 6): Kalman Sensor Fusion & MATLAB/Simulink Telemetry Module** *(Up next!)*

---

## Quick Start — Play AeroDrive 3D!

```powershell
git pull origin arena/8b9cd07c-vr
python step3_car_game.py
```

### Gesture & Keyboard Controls (`step3_car_game.py`)

| Gesture / Key | Action |
| :--- | :--- |
| **Move Hand Left / Right** | Smoothly steer & tilt your 3D OpenGL Sports Car across the 3 lanes |
| **Pinch (`Thumb + Index`)** | Activate **NITRO BOOST** (2x Speed + 2x Score + 3D Flames) / Restart on Game Over |
| **Quick Flick Hand UP** | **Lucas-Kanade Optical Flow Stunt Jump** (360° flip over barriers for `+250` bonus!) |
| **`G`** | Toggle **Garage Inspect Mode** (spread fingers to **Scale**, move hand to **Rotate** 3D Car) |
| **`W` / `P`** | Switch **Wireframe (`GL_LINE`)** vs **Solid Polygon (`GL_FILL`)** rendering |
| **`L`** | Cycle **4 OpenGL Lighting Modes** (`Studio Key`, `Cyber Neon`, `Sunset Rim`, `Unlit Flat`) |
| **`F`** | Toggle **Lucas-Kanade Optical Flow** motion vector trails |
| **`R`** | Restart race anytime |
| **`Left/Right` / `Space` / `N`** | Optional keyboard steering, jump & Nitro controls |
| **`Q` or `ESC`** | Quit cleanly |
