import cv2
import numpy as np

class Visualizer:
    def __init__(self, employee_mode=False):
        # Color palette (BGR format)
        self.COLOR_PEDESTRIAN = (0, 255, 0)     # Green
        self.COLOR_EMPLOYEE = (255, 255, 0)     # Teal/Cyan
        self.COLOR_LOITERING = (0, 0, 255)      # Bright Red / Crimson
        self.COLOR_RIDER = (0, 0, 255)          # Red
        self.COLOR_VEHICLE = (255, 191, 0)      # Deep Blue/Cyan
        self.COLOR_HOTLIST = (0, 0, 255)        # Bright Red
        self.COLOR_LINE = (0, 255, 255)         # Yellow connector line
        self.COLOR_BANNER_BG = (30, 30, 30)     # Dark overlay background
        self.COLOR_TEXT = (255, 255, 255)       # White
        self.employee_mode = employee_mode

    def draw(self, frame, telemetry, line_y=None, occupancy=None):
        img = frame.copy()
        
        # 1. Draw Persons (Green; Teal/Cyan EMPLOYEE boxes in employee mode)
        for ped in telemetry.get('pedestrians', []):
            x1, y1, x2, y2 = map(int, ped['bbox'])
            track_id = ped.get('track_id')
            if self.employee_mode:
                dwell = ped.get('dwell_time_seconds') or 0.0
                if ped.get('is_loitering'):
                    color = self.COLOR_LOITERING
                    label = (f"LOITERING | EMP #{track_id} | DWELL: {dwell:.0f}s"
                             if track_id is not None else f"LOITERING | DWELL: {dwell:.0f}s")
                else:
                    color = self.COLOR_EMPLOYEE
                    label = (f"EMPLOYEE #{track_id} | DWELL: {dwell:.0f}s"
                             if track_id is not None else f"EMPLOYEE | DWELL: {dwell:.0f}s")
            elif track_id is not None:
                color = self.COLOR_PEDESTRIAN
                label = f"Ped #{track_id} {ped['confidence']}"
            else:
                color = self.COLOR_PEDESTRIAN
                label = f"Pedestrian {ped['confidence']}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, max(y1 - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # 2. Draw All Vehicles (Deep Blue/Cyan; Bright Red when hotlist matched)
        for veh in telemetry.get('vehicles', []):
            x1, y1, x2, y2 = map(int, veh['bbox'])
            is_hotlist = bool(veh.get('hotlist_match'))
            color = self.COLOR_HOTLIST if is_hotlist else self.COLOR_VEHICLE
            thickness = 3 if is_hotlist else 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            track_id = veh.get('track_id')
            if is_hotlist:
                label = f"HOTLIST {veh['class'].upper()} #{track_id} {veh['confidence']}"
            elif track_id is not None:
                label = f"{veh['class'].upper()} #{track_id} {veh['confidence']}"
            else:
                label = f"{veh['class'].upper()} {veh['confidence']}"
            cv2.putText(img, label, (x1, max(y1 - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # 3. Draw Active Riders (Red) and connecting lines to their bike/motorcycle
        for rider in telemetry.get('active_riders', []):
            px1, py1, px2, py2 = map(int, rider['bbox'])
            cv2.rectangle(img, (px1, py1), (px2, py2), self.COLOR_RIDER, 2)
            
            label = f"Rider ({rider['rider_of']}) IoP:{rider['iop']}"
            track_id = rider.get('track_id')
            if track_id is not None:
                label = f"Rider #{track_id} ({rider['rider_of']}) IoP:{rider['iop']}"
            cv2.putText(img, label, (px1, max(py1 - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_RIDER, 1)
            
            # Draw line between pedestrian center and target vehicle center
            if 'vehicle_bbox' in rider:
                vx1, vy1, vx2, vy2 = map(int, rider['vehicle_bbox'])
                ped_center = ((px1 + px2) // 2, (py1 + py2) // 2)
                veh_center = ((vx1 + vx2) // 2, (vy1 + vy2) // 2)
                cv2.line(img, ped_center, veh_center, self.COLOR_LINE, 2, cv2.LINE_AA)

        # 4. Render Overlay Metrics Header
        counts = telemetry.get('counts', {})
        bd = counts.get('breakdown', {})
        person_label = "Employees" if self.employee_mode else "Pedestrians"
        banner_text = (
            f"{person_label}: {counts.get('pedestrians', 0)} | "
            f"Active Riders: {counts.get('active_riders', 0)} | "
            f"Vehicles: {counts.get('total_vehicles', 0)} "
            f"(Cars: {bd.get('car', 0)}, Motos: {bd.get('motorcycle', 0)}, "
            f"Bikes: {bd.get('bicycle', 0)}, Buses: {bd.get('bus', 0)}, Trucks: {bd.get('truck', 0)})"
        )
        
        cv2.rectangle(img, (0, 0), (img.shape[1], 35), self.COLOR_BANNER_BG, -1)
        cv2.putText(img, banner_text, (10, 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 1, cv2.LINE_AA)

        # 5. Draw the virtual ENTRY/EXIT crossing line + occupancy HUD
        if self.employee_mode:
            if line_y is not None:
                self._draw_entry_line(img, line_y)
            if occupancy is not None:
                loiter_count = sum(
                    1 for p in telemetry.get('pedestrians', [])
                    if p.get('is_loitering'))
                occ_text = ("Occupancy: {} | Entries: {} | Exits: {} | Loitering: {}").format(
                    occupancy.get('current_occupancy', 0),
                    occupancy.get('total_entries', 0),
                    occupancy.get('total_exits', 0),
                    loiter_count)
                cv2.putText(img, occ_text, (10, img.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_TEXT, 1, cv2.LINE_AA)

        return img

    def _draw_entry_line(self, img, line_y):
        """Render the ENTRY/EXIT crossing line with direction markers."""
        h, w = img.shape[:2]
        y_line = max(0, min(h - 1, int(line_y * h)))
        cv2.line(img, (0, y_line), (w, y_line), self.COLOR_LINE, 2)

        cv2.putText(img, "ENTRY v", (10, max(y_line - 10, 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_LINE, 1, cv2.LINE_AA)
        cv2.putText(img, "EXIT ^", (w - 100, min(y_line + 22, h - 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_LINE, 1, cv2.LINE_AA)
        self._draw_direction_triangle(img, (20, y_line - 6), down=True)
        self._draw_direction_triangle(img, (w - 20, y_line + 6), down=False)

    def _draw_direction_triangle(self, img, tip, down, size=9):
        """Draw a small filled triangle indicating entry/exit direction."""
        tx, ty = tip
        if down:
            pts = np.array([[tx, ty], [tx - size, ty - size], [tx + size, ty - size]], dtype=np.int32)
        else:
            pts = np.array([[tx, ty], [tx - size, ty + size], [tx + size, ty + size]], dtype=np.int32)
        cv2.fillPoly(img, [pts], self.COLOR_LINE)
