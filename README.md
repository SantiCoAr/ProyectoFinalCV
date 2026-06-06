# Visual Password and Two-Player Snake Tracker

Final project for the Computer Vision course at Universidad Pontificia Comillas ICAI.

This repository contains a real-time OpenCV application that combines a visual password system with a two-player, camera-controlled Snake game. The user first unlocks the application by showing a sequence of colored geometric symbols to the webcam. After a successful unlock, the camera feed is split into two play areas and each player controls a snake with a colored marker.

## What the Project Does

The application has two main stages:

1. **Locked mode: visual password**
   - Detects colored geometric patterns from the webcam.
   - Classifies each detection as a combined color/shape label, such as `red_circle`.
   - Accepts a symbol only after it remains stable for several frames.
   - Requires the user to remove the symbol before accepting the next one, avoiding accidental duplicate inputs.
   - Unlocks the game when the full password sequence is correct.

2. **Unlocked mode: two-player Snake**
   - Splits the webcam image into left and right halves.
   - Player 1 uses a green marker on the left side.
   - Player 2 uses a red marker on the right side.
   - Tracks each marker with HSV color segmentation.
   - Smooths and predicts marker motion with an independent 2D Kalman filter for each player.
   - Uses the tracked marker position as the snake head.
   - Ends the round when one player reaches the target score.

The default password in `src/main.py` is:

```text
red_circle -> blue_triangle -> green_square -> yellow_line
```

## Repository Structure

```text
ProyectoFinalCV/
+-- src/
|   +-- main.py                  # Main application: password + tracker + Snake
|   +-- color_shape_detector.py  # HSV color segmentation and shape classification
|   +-- password_decoder.py      # Stable-frame visual password decoder
|   +-- finger_detector.py       # Colored marker detector for player tracking
|   +-- kalman_tracker.py        # 2D Kalman filter wrapper
|   +-- snake_game.py            # Snake game state and scoring logic
|   +-- calibracion.py           # Camera calibration from chessboard images
+-- tests/
|   +-- run_color_shape_live.py        # Live color/shape detector demo
|   +-- run_password_system_live.py    # Live password decoder demo
|   +-- run_snake_tracker_2d_split.py  # Live two-player tracker/Snake demo
+-- data/                        # Calibration and experiment images
+-- report/                      # Final project report PDF
+-- output_pipeline/             # Generated outputs, if produced locally
+-- environment_win.yml          # Conda environment for Windows
+-- environment_unix.yml         # Conda environment for Linux/Unix
+-- environment_macos.yml        # Conda environment for macOS
+-- README.md
```

## Requirements

- Python 3.9
- Conda or Miniconda
- A working webcam
- Good, even lighting
- Physical colored markers or printed colored shapes:
  - Password stage: red circle, blue triangle, green square, yellow line
  - Game stage: green marker for Player 1, red marker for Player 2

The supplied Conda environments include the main dependencies, including OpenCV, NumPy, SciPy, and Matplotlib.

## Installation

Create the Conda environment for your operating system from the repository root.

### Windows

```bash
conda env create -f environment_win.yml
conda activate voi-lab
```

### Linux / Unix

```bash
conda env create -f environment_unix.yml
conda activate voi-lab
```

### macOS

```bash
conda env create -f environment_macos.yml
conda activate voi-lab
```

If an environment already exists and you need to refresh it:

```bash
conda env update -f environment_win.yml --prune
```

Use the matching `environment_*.yml` file for your operating system.

## Running the Main Application

From the repository root:

```bash
python src/main.py
```

The program opens the webcam and starts in `LOCKED` mode. Show the password symbols in order, removing each symbol after it has been accepted. When the sequence is correct, the application switches to the two-player Snake tracker.

If the webcam does not open, check that no other application is using it. If your system exposes the camera at a different index, update `cv2.VideoCapture(0)` in the script you are running.

## Controls

### Locked Mode

| Key | Action |
| --- | --- |
| `r` | Reset the current password attempt |
| `q` | Quit |

### Unlocked Mode

| Key | Action |
| --- | --- |
| `r` | Reset the current Snake round |
| `k` | Reset both Kalman filters |
| `m` | Show or hide segmentation masks for debugging |
| `n` | Start a new round after a winner is shown |
| `q` | Quit |

## Live Component Demos

The `tests/` directory contains standalone scripts for testing individual parts of the pipeline with a webcam:

```bash
python tests/run_color_shape_live.py
python tests/run_password_system_live.py
python tests/run_snake_tracker_2d_split.py
```

These are useful for tuning lighting, marker colors, HSV thresholds, and camera placement before running the full application.

## Technical Overview

The project uses classic computer vision techniques rather than a trained model:

- **HSV segmentation** isolates colored objects from the webcam feed.
- **Morphological opening and closing** reduce noise in binary masks.
- **Contour extraction** finds candidate objects.
- **Shape classification** uses polygon approximation, aspect ratio, and circularity.
- **Stable-frame decoding** makes the password input robust to flickering detections.
- **2D Kalman filtering** estimates marker position and velocity as `[x, y, vx, vy]`.
- **Split-screen tracking** gives each player an independent board and tracker.

## Calibration

`src/calibracion.py` contains a chessboard-based camera calibration script. It loads images from `data/*.jpg`, detects internal chessboard corners, refines them with `cornerSubPix`, and estimates the camera matrix and distortion coefficients with `cv2.calibrateCamera`.

The script assumes a chessboard pattern of `(7, 9)` internal corners and a square size of `20` units. Adjust those values if your calibration board is different.

Run it from the repository root:

```bash
python src/calibracion.py
```

## Practical Tips

- Use saturated colors and avoid backgrounds with similar colors.
- Keep the objects large enough in the frame; very small contours are filtered out as noise.
- Avoid strong shadows and reflections.
- Remove each password symbol from the camera view before showing the next one.
- Keep Player 1 on the left half of the image and Player 2 on the right half after unlocking.
- Use `m` in unlocked mode to inspect the color masks if tracking is unstable.

## Report

The full academic report is available in:

```text
report/Informe_Computer_Vision_Gonzalo_Garcia_Santiago_Cordoba.pdf
```

It includes the project methodology, implementation details, experimental results, and conclusions.

## Authors

- Santiago Cordoba Artieda
- Gonzalo Garcia Martinez-Echevarria

## License

Academic and educational use.
