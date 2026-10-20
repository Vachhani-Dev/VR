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
  - **Zero-Lag Dual-Window Architecture:** Dedicated **60 FPS** top-down 2D game window (`~1.3 ms/frame`) paired with an asynchronous background camera/gesture thread and separate companion **Gesture & Optical Flow Monitor** window.
  - **Detailed Top-Down Artwork:** Custom Hero Supercar (dual headlight cones, racing stripes, rear spoiler, glowing red brake lamps, turn signals, tire skid marks & Nitro flames) + **5 AI Vehicle Classes** (*City Sedans, Yellow Taxis, Sport Coupes, Family SUVs, and Heavy Cargo Semi-Trucks*).
  - **Unpredictable Traffic AI with Turn Indicators:** AI vehicles follow traffic speeds, apply red brake lights behind slower trucks, and signal lane changes in advance using **flashing amber turn-signal indicators + side chevrons** before merging across lanes!
  - **Natural Driving Gestures + Lucas-Kanade Optical Flow (`cv2.calcOpticalFlowPyrLK`):**
    - **Move Hand Left / Right** -> Smoothly steer across all 4 lanes (palm center fused with Lucas-Kanade optical flow velocity).
    - **Close Hand (Make a Fist)** -> **BRAKE HARD** (scale-invariant multi-finger curl detection; lights red brake lamps & lays tire marks).
    - **Pinch (`Thumb + Index`)** -> **NITRO BOOST** (305 km/h max speed + **2x Score Multiplier**).
- [ ] **Step 4 (Unit 6): Kalman Filter Sensor Fusion & MATLAB/Simulink Export** *(Upcoming)*

---

## Quick Start — Play AeroDrive 2D!

```powershell
git pull origin arena/8b9cd07c-vr
python step3_car_game.py
```

### Gesture & Keyboard Controls (`step3_car_game.py`)

| Hand Gesture / Key | Driving Action |
| :--- | :--- |
| **Move Hand Left / Right** *(or `←`/`→`, `A`/`D`)* | Steer smoothly across the 4 highway lanes |
| **Open Hand** | Smooth highway cruise acceleration |
| **Close Hand into a FIST** *(or `↓`, `S`, `Space`)* | **BRAKE HARD** (red brake lamps + tire skid marks) |
| **Pinch (`Thumb + Index`)** *(or `↑`, `W`, `N`)* | **NITRO TURBO BOOST** (blue flames + 2x score!) |
| **`C`** | Show / Hide the separate **Camera & Optical Flow Monitor** window |
| **`R`** *(or **Pinch** on Game Over)* | Restart race |
| **`Q` or `ESC`** | Quit cleanly |
