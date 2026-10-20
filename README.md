# AeroTouch 3D — AeroDrive 2D Top-Down Traffic Racer Edition

**Hand-Gesture & Optical-Flow Controlled Highway Traffic Game + 3D OpenGL Rubric Suite**  
*University Coursework Project — Unit 5 (Vision-Based AR, Pose Estimation, Optical Flow, Rendering) & Unit 6 (Sensor Fusion, Simulink)*

---

## Project Modules

- [x] **Step 1: Hand Tracking & Pinch Detection (`step1_hand_tracking.py`)**
  - Real-time webcam capture + 21-landmark MediaPipe hand pose estimation + offline `models/hand_landmarker.task` bundle + Euclidean pinch detection.
- [x] **Step 2: 3D OpenGL Rendering Engine (`step2_ar_renderer.py`)**
  - Live OpenGL `GL_TEXTURE_2D` camera background + side-docked 3D **Cube** & **Pyramid** + **`[W]`** Wireframe, **`[P]`** Solid Polygon, **`[L]`** 4-Mode Phong Lighting, and **`[D]` / Arrow** positioning.
- [x] **Step 3: AeroDrive 2D — 60 FPS Top-Down Traffic Racer (`step3_car_game.py`)**
  - **Zero-Lag Dual-Window Architecture:** Dedicated **60 FPS** top-down 2D game window paired with an asynchronous background camera/gesture worker and separate **Gesture & Optical Flow Monitor** window.
  - **Easy / Medium / Hard Modes + Score Escalation:**
    - **`[1] EASY`:** Light traffic with wide multi-lane gaps; cars stay straight in their lanes (**no turn indicators or lane switches**).
    - **`[2] MEDIUM`:** Balanced traffic flow; AI cars flash **amber turn-signal indicators + side chevrons** before merging smoothly across lanes.
    - **`[3] HARD`:** Dense high-speed rush-hour traffic (`1.6x` score multiplier) with frequent unpredictable blinker cuts & fast coupes.
    - **Score-Driven Progression:** As your **score** climbs (`STAGE 1 -> STAGE 2 -> STAGE 3+`), speed, traffic density, and traffic pace progressively intensify!
  - **Natural Driving Gestures + Lucas-Kanade Optical Flow (`cv2.calcOpticalFlowPyrLK`):**
    - **Move Hand Left / Right** -> Smoothly steer across all 4 lanes.
    - **Close Hand (Make a Fist)** -> **BRAKE HARD** (red brake lamps + tire skid marks).
    - **Pinch (`Thumb + Index`)** -> **NITRO BOOST** (305+ km/h flames + **2x Score Multiplier**).

---

## Quick Start — Play AeroDrive 2D!

```powershell
git pull origin arena/8b9cd07c-vr
python step3_car_game.py
```

### Gesture & Keyboard Controls (`step3_car_game.py`)

| Hand Gesture / Key | Driving / Menu Action |
| :--- | :--- |
| **Click Card / `1` `2` `3` / Move Hand + Pinch** | Select **Easy**, **Medium**, or **Hard** mode |
| **`M`** | Open the **Difficulty Selection Menu** anytime |
| **Move Hand Left / Right** *(or `←`/`→`, `A`/`D`)* | Steer smoothly across the 4 highway lanes |
| **Close Hand into a FIST** *(or `↓`, `S`, `Space`)* | **BRAKE HARD** (red brake lamps + tire skid marks) |
| **Pinch (`Thumb + Index`)** *(or `↑`, `W`, `N`)* | **NITRO TURBO BOOST** (blue flames + 2x score!) |
| **`C`** | Show / Hide the separate **Camera & Optical Flow Monitor** window |
| **`R`** *(or **Pinch** on Game Over)* | Restart race |
| **`Q` or `ESC`** | Quit cleanly |
