# AeroTouch 3D & AeroDrive 2D — Contactless Hand-Gesture & Optical-Flow Suite

A complete, progressive 3-stage **Computer Vision, OpenGL 3D & 60 FPS Top-Down Traffic Simulation** suite built in Python using **OpenCV (Pyramidal Lucas-Kanade Optical Flow)**, **MediaPipe Hand Pose Estimation**, **PyOpenGL**, and **Pygame (`pygame-ce`)**.

## Project Report & Presentation Submission Package (`docs/`)

Ready for university submission & live demonstration (**13th – 17th October 2026**):
- **Editable Microsoft Word Report (`.docx`):** [`docs/AeroDrive_Project_Report.docx`](docs/AeroDrive_Project_Report.docx) *(Edit bracketed Roll No./PRN, group members & institute placeholders on Page 1)*
- **Print-Ready PDF Report (`.pdf`):** [`docs/AeroDrive_Project_Report.pdf`](docs/AeroDrive_Project_Report.pdf)
- **10-Slide Viva Presentation (`.pptx`):** [`docs/AeroDrive_Presentation.pptx`](docs/AeroDrive_Presentation.pptx)
- **Illustrated GitHub Report (`.md`):** [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md)

## Quick Start (Windows PowerShell)

```powershell
python -m pip install opencv-python mediapipe pygame-ce PyOpenGL PyOpenGL_accelerate numpy
```

## Progressive Project Modules

| Step | Script | Description |
| :--- | :--- | :--- |
| **Step 1** | `step1_hand_tracking.py` | Real-time mirrored webcam capture, 21-landmark hand pose skeleton, Thumb `[4]` & Index `[8]` tracking, and live Euclidean pinch gesture detection. |
| **Step 2** | `step2_ar_renderer.py` | OpenGL 3D AR renderer streaming live camera frames into `GL_TEXTURE_2D` with side-docked 3D **Cube** & **Pyramid** models, `[W]` Wireframe, `[P]` Solid Polygon, and `[L]` Phong Lighting presets. |
| **Step 3** | `step3_car_game.py` | **AeroDrive 2D** — Dedicated **60 FPS** top-down 4-lane highway racer with separate companion Gesture & Optical Flow window, **Easy / Medium / Hard** modes, AI turn-indicator lane changes, and score-driven difficulty progression! |

### AeroDrive 2D Hand Gesture & Mode Controls (`step3_car_game.py`)
- **Select Difficulty Mode (`Easy` / `Medium` / `Hard`):** Hover hand over a mode card + **Pinch** (or click with mouse / press `1`, `2`, `3`, or `SPACE`).
  - **Easy `[1]`:** Light traffic, higher cruise speed margin, **no AI turn indicators / no lane changes**.
  - **Medium `[2]`:** Moderate traffic; AI drivers flash **amber turn indicators** before merging lanes (`1.25x` score multiplier).
  - **Hard `[3]`:** Heavy rush-hour traffic, fast sport coupes, frequent indicator merges (`1.60x` score multiplier).
  - **Dynamic Score Escalation:** Highway cruise speed, traffic density, and lane-merge frequency scale continuously as your **Score** climbs (`STAGE UP` every `1,500` pts)!
- **Move Hand Left / Right:** Analog steering across all 4 highway lanes (fused with Lucas-Kanade optical flow velocity lead)
- **Open Hand:** Smooth auto-cruise highway driving
- **Close Hand (Fist):** Hard proportional **Brake** (red brake lights + tire skid marks to let merging cars pass)
- **Pinch (Thumb + Index):** **Nitro Boost** (twin blue exhaust flames + supersonic speed + `2x` score multiplier)
- **Keyboard Fallbacks:** `1` / `2` / `3` select/switch mode, `M` Mode Menu, `Left/Right` or `A/D` steer, `Down/S` brake, `Up/W/Space` nitro, `R` restart, `C` toggle camera monitor, `ESC` quit.
