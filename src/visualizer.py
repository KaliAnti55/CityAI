import cv2

class CityVisualizer:
    def __init__(self):
        # Color palette (BGR format)
        self.COLOR_PEDESTRIAN = (0, 255, 0)     # Green
        self.COLOR_RIDER = (0, 0, 255)          # Red
        self.COLOR_VEHICLE = (255, 191, 0)      # Deep Blue/Cyan
        self.COLOR_LINE = (0, 255, 255)         # Yellow connector line
        self.COLOR_BANNER_BG = (30, 30, 30)     # Dark overlay background
        self.COLOR_TEXT = (255, 255, 255)       # White

    def draw(self, frame, telemetry):
        img = frame.copy()
        
        # 1. Draw Standalone Pedestrians (Green)
        for ped in telemetry.get('pedestrians', []):
            x1, y1, x2, y2 = map(int, ped['bbox'])
            cv2.rectangle(img, (x1, y1), (x2, y2), self.COLOR_PEDESTRIAN, 2)
            label = f"Pedestrian {ped['confidence']}"
            cv2.putText(img, label, (x1, max(y1 - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_PEDESTRIAN, 1)

        # 2. Draw All Vehicles (Deep Blue/Cyan)
        for veh in telemetry.get('vehicles', []):
            x1, y1, x2, y2 = map(int, veh['bbox'])
            cv2.rectangle(img, (x1, y1), (x2, y2), self.COLOR_VEHICLE, 2)
            label = f"{veh['class'].upper()} {veh['confidence']}"
            cv2.putText(img, label, (x1, max(y1 - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_VEHICLE, 1)

        # 3. Draw Active Riders (Red) and connecting lines to their bike/motorcycle
        for rider in telemetry.get('active_riders', []):
            px1, py1, px2, py2 = map(int, rider['bbox'])
            cv2.rectangle(img, (px1, py1), (px2, py2), self.COLOR_RIDER, 2)
            
            label = f"Rider ({rider['rider_of']}) IoP:{rider['iop']}"
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
        banner_text = (
            f"Pedestrians: {counts.get('pedestrians', 0)} | "
            f"Active Riders: {counts.get('active_riders', 0)} | "
            f"Vehicles: {counts.get('total_vehicles', 0)} "
            f"(Cars: {bd.get('car', 0)}, Motos: {bd.get('motorcycle', 0)}, "
            f"Bikes: {bd.get('bicycle', 0)}, Buses: {bd.get('bus', 0)}, Trucks: {bd.get('truck', 0)})"
        )
        
        cv2.rectangle(img, (0, 0), (img.shape[1], 35), self.COLOR_BANNER_BG, -1)
        cv2.putText(img, banner_text, (10, 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 1, cv2.LINE_AA)
        
        return img
