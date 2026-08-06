# 🏙️ Smart City Urban Mobility & Pedestrian AI Detector

An open-source, standalone computer vision application built with **OpenCV**, **YOLOv8x**, and **PyTorch** for smart-city traffic analytics, pedestrian monitoring, and automated human-to-vehicle usage association.

---

## 📌 Overview

Understanding urban space requires more than just counting cars and people independently. This application uses spatial geometry algorithms to identify **Human-to-Vehicle Spatial Associations**—automatically determining whether a detected human is a **pedestrian walking on foot** or an **active rider/driver** operating a bicycle, motorcycle, car, bus, or truck.

### Key Features
- **High-Accuracy Detection:** Powered by `yolov8x.pt` (YOLOv8 Extra-Large) for dense urban scenes.
- **Human-Transport Association:** Uses **Intersection over Person (IoP)** spatial ratio calculations to link humans to their transportation modes.
- **Smart City Overlay Dashboard:** Live visual dashboard displaying real-time traffic statistics, color-coded bounding boxes, and active rider connector lines.
- **Structured Telemetry Export:** Option to export frame-by-frame JSON metadata for traffic telemetry and municipal databases.
- **Headless & Cloud Execution Ready:** Native command-line flags designed for seamless execution on **Google Colab**, edge servers, or Docker containers.

---

## 🎯 Target Categories & COCO Mapping

The system tracks 7 primary urban mobility targets using native pre-trained COCO dataset classes:

| Class Icon | Class Name | Target Category Description |
| :--- | :--- | :--- |
| 🏃 | **Person** | Pedestrians walking on foot |
| 🚲 | **Bicycle** | Human-powered transport & personal mobility |
| 🚗 | **Car** | Personal travel vehicles |
| 🏍️ | **Motorcycle** | Fast personal & delivery two-wheelers |
| 🚌 | **Bus** | Mass public transit |
| 🚚 | **Truck** | Freight, cargo, and logistics transport |
| 🚐 / 🛴 | **Vans & Scooters** | Mapped natively to `car`/`truck` and `motorcycle`/`bicycle`* |

> **Note on Scooters & Vans:** Pre-trained COCO weights categorize gas/e-scooters under `motorcycle` or `bicycle`, and vans under `car` or `truck`. For strict custom separation, custom fine-tuned model weights can be loaded using the `--weights` flag.

---

## 📂 Repository Structure

```text
city-mobility-ai/
├── AGENT.md               # Directives for AI coding agents
├── README.md              # User manual & setup documentation
├── requirements.txt       # Dependencies (ultralytics, opencv-python, numpy)
├── .gitignore             # Git exclusions
├── main.py                # Command-Line Interface (CLI) entry point
│
└── src/
    ├── __init__.py
    ├── detector.py        # Core YOLO engine & IoP spatial association logic
    └── visualizer.py      # OpenCV bounding box overlays & stats dashboard