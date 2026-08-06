"""
City Mobility AI - Urban Transportation & Pedestrian Analytics
Core AI Detection & Spatial Association Engine
"""

import numpy as np
from ultralytics import YOLO


class CityMobilityDetector:
    """
    Core AI Engine for Object Detection and Human-Transport Spatial Association.
    """

    # Direct mapping from COCO dataset class indices to urban mobility target names
    COCO_TARGET_MAP = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(self, model_path: str = "yolov8x.pt", iop_threshold: float = 0.40):
        """
        :param model_path: Path to YOLO model weights (default: yolov8x.pt)
        :param iop_threshold: Minimum Intersection over Person ratio to link human to vehicle
        """
        self.model = YOLO(model_path)
        self.iop_threshold = iop_threshold
        self.target_class_ids = list(self.COCO_TARGET_MAP.keys())

    @staticmethod
    def compute_iop(person_bbox: list, vehicle_bbox: list) -> float:
        """
        Calculates the Intersection over Person (IoP) spatial ratio.

        IoP = Area(Intersection) / Area(Person_BBox)
        """
        px1, py1, px2, py2 = person_bbox
        vx1, vy1, vx2, vy2 = vehicle_bbox

        # Calculate bounding box intersection coordinates
        ix1 = max(px1, vx1)
        iy1 = max(py1, vy1)
        ix2 = min(px2, vx2)
        iy2 = min(py2, vy2)

        intersection_width = max(0, ix2 - ix1)
        intersection_height = max(0, iy2 - iy1)
        intersection_area = intersection_width * intersection_height

        person_area = max(1, (px2 - px1) * (py2 - py1))

        return float(intersection_area / person_area)

    def analyze_frame(self, frame: np.ndarray, confidence: float = 0.45):
        """
        Runs object detection on a single image/frame, filters targets,
        and determines human-transport associations.

        :param frame: BGR NumPy array from OpenCV
        :param confidence: Minimum detection confidence threshold
        :return: (annotated_frame, metadata_dict)
        """
        # Run YOLO inference
        results = self.model(frame, conf=confidence, verbose=False)[0]

        raw_people = []
        raw_vehicles = []
        class_breakdown = {name: 0 for name in self.COCO_TARGET_MAP.values() if name != "person"}

        # Step 1: Extract and sort bounding boxes into Humans vs Vehicles
        for box in results.boxes:
            class_id = int(box.cls[0])

            if class_id in self.target_class_ids:
                bbox = list(map(int, box.xyxy[0]))
                conf = round(float(box.conf[0]), 2)
                label = self.COCO_TARGET_MAP[class_id]

                item = {
                    "class": label,
                    "confidence": conf,
                    "bbox": bbox,
                }

                if label == "person":
                    item["id"] = len(raw_people)
                    raw_people.append(item)
                else:
                    item["id"] = len(raw_vehicles)
                    raw_vehicles.append(item)
                    class_breakdown[label] += 1

        # Step 2: Calculate Spatial Association (IoP)
        active_riders = []
        pedestrians = []
        associated_vehicle_ids = set()

        for person in raw_people:
            best_vehicle = None
            max_iop = 0.0

            for vehicle in raw_vehicles:
                iop = self.compute_iop(person["bbox"], vehicle["bbox"])

                if iop >= self.iop_threshold and iop > max_iop:
                    max_iop = iop
                    best_vehicle = vehicle

            if best_vehicle:
                active_riders.append(
                    {
                        "person_id": person["id"],
                        "person_bbox": person["bbox"],
                        "confidence": person["confidence"],
                        "transport_class": best_vehicle["class"],
                        "vehicle_bbox": best_vehicle["bbox"],
                        "iop_score": round(max_iop, 2),
                    }
                )
                associated_vehicle_ids.add(best_vehicle["id"])
            else:
                pedestrians.append(person)

        # Step 3: Package analytics metadata
        metadata = {
            "counts": {
                "pedestrians": len(pedestrians),
                "active_riders": len(active_riders),
                "total_vehicles": len(raw_vehicles),
                "breakdown": class_breakdown,
            },
            "pedestrians": pedestrians,
            "active_riders": active_riders,
            "vehicles": raw_vehicles,
        }

        return frame, metadata