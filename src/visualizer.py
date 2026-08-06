"""
City Mobility AI - Urban Transportation & Pedestrian Analytics
Visualizer & Overlay Rendering Module
"""

import cv2
import numpy as np


class Visualizer:
    """
    Renders bounding boxes, spatial association indicators, and analytics dashboard overlays.
    """

    # Color Palette (BGR Format)
    COLOR_PEDESTRIAN = (0, 220, 0)       # Green
    COLOR_RIDER = (0, 0, 255)            # Red
    COLOR_VEHICLE = (255, 180, 0)        # Blue/Cyan
    COLOR_TEXT = (255, 255, 255)         # White
    COLOR_DASH_BG = (20, 20, 20)         # Dark Gray for Header Banner
    COLOR_ACCENT = (0, 215, 255)         # Yellow/Gold for Highlights

    def __init__(self, font_scale: float = 0.5, thickness: int = 2):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness = thickness

    def draw_dashboard(self, frame: np.ndarray, metadata: dict) -> np.ndarray:
        """
        Renders bounding boxes, active rider links, and top summary header overlay on the frame.
        """
        output = frame.copy()

        # Step 1: Draw Vehicles
        for v in metadata.get("vehicles", []):
            x1, y1, x2, y2 = v["bbox"]
            label = f"{v['class'].upper()} {v['confidence']}"
            cv2.rectangle(output, (x1, y1), (x2, y2), self.COLOR_VEHICLE, self.thickness)
            self._draw_label(output, label, x1, y1, self.COLOR_VEHICLE)

        # Step 2: Draw Pedestrians
        for p in metadata.get("pedestrians", []):
            x1, y1, x2, y2 = p["bbox"]
            label = f"Pedestrian {p['confidence']}"
            cv2.rectangle(output, (x1, y1), (x2, y2), self.COLOR_PEDESTRIAN, self.thickness)
            self._draw_label(output, label, x1, y1, self.COLOR_PEDESTRIAN)

        # Step 3: Draw Active Riders/Drivers with Linked Bounding Box
        for rider in metadata.get("active_riders", []):
            px1, py1, px2, py2 = rider["person_bbox"]
            label = f"Rider ({rider['transport_class']}) IoP:{rider['iop_score']}"
            cv2.rectangle(output, (px1, py1), (px2, py2), self.COLOR_RIDER, self.thickness + 1)
            self._draw_label(output, label, px1, py1, self.COLOR_RIDER)

            # Draw connector line from rider center to vehicle center
            vx1, vy1, vx2, vy2 = rider["vehicle_bbox"]
            p_center = ((px1 + px2) // 2, (py1 + py2) // 2)
            v_center = ((vx1 + vx2) // 2, (vy1 + vy2) // 2)
            cv2.line(output, p_center, v_center, self.COLOR_RIDER, 1, cv2.LINE_AA)

        # Step 4: Draw Top Dashboard Banner
        output = self._render_header_banner(output, metadata["counts"])

        return output

    def _draw_label(self, frame: np.ndarray, text: str, x: int, y: int, bg_color: tuple):
        """Draws a solid background text label for high visibility."""
        (text_w, text_h), baseline = cv2.getTextSize(text, self.font, self.font_scale, 1)
        y_text = max(y - 5, text_h + 5)
        
        # Background box behind label text
        cv2.rectangle(
            frame,
            (x, y_text - text_h - 4),
            (x + text_w + 6, y_text + baseline),
            bg_color,
            -1
        )
        # White text inside label box
        cv2.putText(
            frame,
            text,
            (x + 3, y_text),
            self.font,
            self.font_scale,
            self.COLOR_TEXT,
            1,
            cv2.LINE_AA
        )

    def _render_header_banner(self, frame: np.ndarray, counts: dict) -> np.ndarray:
        """Renders top summary statistics bar."""
        h, w, _ = frame.shape
        banner_h = 45

        # Create semi-transparent overlay banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), self.COLOR_DASH_BG, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Text Metrics Line
        stats_text = (
            f"Pedestrians: {counts['pedestrians']} | "
            f"Active Riders: {counts['active_riders']} | "
            f"Vehicles: {counts['total_vehicles']} "
            f"(Cars: {counts['breakdown'].get('car', 0)}, "
            f"Motos: {counts['breakdown'].get('motorcycle', 0)}, "
            f"Bikes: {counts['breakdown'].get('bicycle', 0)}, "
            f"Buses: {counts['breakdown'].get('bus', 0)}, "
            f"Trucks: {counts['breakdown'].get('truck', 0)})"
        )

        cv2.putText(
            frame,
            stats_text,
            (15, 28),
            self.font,
            0.55,
            self.COLOR_ACCENT,
            1,
            cv2.LINE_AA
        )

        return frame