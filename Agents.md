# AGENTS.md — CityAI Development Context & Mission

## Project Overview
**CityAI** is a Smart City & Urban Mobility Analytics computer vision platform written in Python and PyTorch/YOLOv8. The core goal is to process video streams/webcams to extract high-value telemetry for civil engineering, public safety, and traffic flow management.

Please inspect the existing codebase (`main.py`, `src/detector.py`, `src/visualizer.py`, `requirements.txt`) to familiarize yourself with the current class structures and dependencies.

---

## What Has Been Completed So Far
1. **Core Processing Pipeline (`main.py`):** Accepts video files or live webcam sources (`--source`), executes detection, renders overlays, and saves structured output video (`data/outputs/output.mp4`) and JSON metrics (`data/outputs/output.json`).
2. **Detection & Association Engine (`src/detector.py`):**
   - Employs YOLOv8x for object detection across target classes: `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`.
   - **Active Rider Logic:** Strictly evaluates open micro-mobility classes (`bicycle`, `motorcycle`) while excluding enclosed heavy vehicles (`car`, `bus`, `truck`).
   - **Feet-Anchored Spatial Intersection (IoP):** Computes ground-contact Intersection over Person using the lower 25% of the pedestrian bounding box to eliminate 2D perspective depth stacking (e.g., pedestrians standing near cars/background crowds).
   - Class entry alias `CityMobilityDetector = CityDetector` configured for full backward compatibility.
3. **Visualization System (`src/visualizer.py`):**
   - Renders color-coded bounding boxes: Standalone Pedestrians (Green), Vehicles (Cyan/Blue), Active Riders (Red).
   - Draws yellow spatial association lines linking active riders to their detected bikes/motorcycles.
   - Generates an automated real-time status banner overlay.
   - Class entry alias `Visualizer = CityVisualizer` configured for compatibility.

---

## Operational Constraints & Execution Rules (CRITICAL)

### 1. NO Dependency Installation / Environment Modifications
* **DO NOT** execute system commands, shell commands, or scripts to install packages (e.g., `pip install`, `apt-get install`, `conda install`).
* **DO NOT** attempt to create virtual environments, modify system paths, or write setup scripts.
* **Reason:** All runtime dependencies (`ultralytics`, `opencv-python`, `torch`, `numpy`, etc.) are pre-configured in the Google Colab environment.

### 2. Output Code & Files Only
* Your primary mission is **code construction and feature implementation**.
* Write modular, production-grade Python code that integrates cleanly into the existing directory structure (`src/`, `main.py`, etc.).

---

## Upcoming Architectural Goal: Memory & Tracking Modules
Your core task is to extend `CityAI` into a spatial-temporal analytics system by implementing unique object tracking and crop memory management:

1. **Unique Object Tracking:** Integrate tracking IDs (e.g., ByteTrack/SORT logic) to ensure vehicles and pedestrians are tracked continuously across frames.
2. **License Plate & Vehicle Crop Memory:** Extract and save cropped vehicle/plate images upon unique object creation (avoiding duplicate frame saves).
3. **Face Feature Crop Memory:** Extract and save cropped face/pedestrian instances indexed per unique track ID.

Maintain clear class signatures, comprehensive docstrings, and strict backward compatibility with existing telemetry dictionary formats.
