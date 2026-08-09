import numpy as np
from ultralytics import YOLO

def non_max_suppression(boxes, scores, iou_thresh=0.45):
    """Greedy Non-Maximum Suppression over absolute ``(x1, y1, x2, y2)`` boxes.

    Given per-box confidence scores, iteratively keeps the highest-scoring
    box and suppresses remaining boxes whose IoU with it exceeds
    ``iou_thresh``. Overlapping or duplicate detections of the same
    physical object (e.g. a partially occluded employee) are collapsed
    into a single box.

    Args:
        boxes: Iterable of ``(x1, y1, x2, y2)`` detection boxes.
        scores: Iterable of matching confidence scores.
        iou_thresh: IoU threshold above which a box is suppressed.

    Returns:
        List of kept indices into the original ``boxes`` sequence.
    """
    arr = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if arr.shape[0] == 0:
        return []
    x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    areas = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[rest], x1[i])
        yy1 = np.maximum(y1[rest], y1[i])
        xx2 = np.minimum(x2[rest], x2[i])
        yy2 = np.minimum(y2[rest], y2[i])
        inter = np.maximum(xx2 - xx1, 0.0) * np.maximum(yy2 - yy1, 0.0)
        denom = areas[rest] + areas[i] - inter
        iou = np.divide(inter, denom, out=np.zeros_like(inter), where=denom > 0)
        order = rest[iou <= iou_thresh]
    return keep

class CityDetector:
    """Detection, tracking and micro-mobility association engine.

    Runs YOLOv8x inference with online multi-object tracking (ByteTrack
    via ``model.track``) so every object receives a persistent ``track_id``
    across frames. Preserves the feet-anchored IoP rider association logic.
    In ``employee_mode`` the detection targets are restricted to ``person``
    (COCO index 0) for workplace employee tracking.

    Duplicate/overlapping person detections are collapsed with NMS on the
    tracker output, and the ByteTrack buffer is tuned so track IDs
    survive short occlusions instead of being re-assigned (``#8`` ->
    ``#14``).
    """

    def __init__(self, model_path="yolov8x.pt", conf_thresh=0.25, iop_thresh=0.45,
                 use_tracking=True, employee_mode=False, nms_iou_thresh=0.45,
                 track_buffer=60, match_thresh=0.8, face_detector_backend="opencv"):
        # Accept either a loaded YOLO object or a string file path
        if isinstance(model_path, str):
            self.model = YOLO(model_path)
        else:
            self.model = model_path
            
        self.conf_thresh = conf_thresh
        self.iop_thresh = iop_thresh
        self.use_tracking = use_tracking
        self.employee_mode = employee_mode

        # Face detection stage: DeepFace built-in detector backend (retinaface,
        # mtcnn, opencv, ssd, ...) run on the raw frame. "none"/"disabled"
        # turns the explicit face stage off entirely.
        backend = (face_detector_backend or "").strip().lower()
        self.face_detector_backend = None if backend in ("", "none", "disabled") else backend
        self._face_detector = None

        # NMS pre-filtering for duplicate/overlapping person boxes
        self.nms_iou_thresh = nms_iou_thresh

        # ByteTrack persistence tuning: buffered tracks survive temporary
        # occlusions so identities are kept rather than re-assigned.
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        
        # Micro-mobility classes that allow active riders
        self.rider_vehicle_classes = ['bicycle', 'motorcycle']
        
        # Enclosed / heavy vehicles (tracked separately, cannot have open riders)
        self.enclosed_vehicle_classes = ['car', 'bus', 'truck']
        
        # Target COCO class mapping (restricted to person in employee mode)
        if employee_mode:
            self.target_classes = {
                0: 'person'
            }
        else:
            self.target_classes = {
                0: 'person',
                1: 'bicycle',
                2: 'car',
                3: 'motorcycle',
                5: 'bus',
                7: 'truck'
            }

    def compute_feet_iop(self, ped_box, veh_box):
        px1, py1, px2, py2 = ped_box
        vx1, vy1, vx2, vy2 = veh_box
        
        # Focus on lower 25% of pedestrian bounding box (feet/ground contact)
        ped_height = py2 - py1
        feet_py1 = py2 - (ped_height * 0.25)
        
        ix1 = max(px1, vx1)
        iy1 = max(feet_py1, vy1)
        ix2 = min(px2, vx2)
        iy2 = min(py2, vy2)
        
        inter_w = max(0.0, ix2 - ix1)
        inter_h = max(0.0, iy2 - iy1)
        intersection = inter_w * inter_h
        
        feet_area = (px2 - px1) * (py2 - feet_py1)
        if feet_area <= 0:
            return 0.0
            
        return intersection / feet_area

    def _run_inference(self, frame):
        """Run YOLO inference, enabling online tracking when configured.

        Uses ``model.track`` (ByteTrack) with ``persist=True`` so track
        identities are carried across frames, and with a longer
        ``track_buffer`` / tighter ``match_thresh`` so identities survive
        brief occlusions without being re-assigned fresh IDs. Falls back
        to plain detection if the tracker is unavailable in the current
        runtime.

        Returns:
            Ultralytics result object for the current frame.
        """
        if self.use_tracking:
            try:
                return self.model.track(frame, conf=self.conf_thresh,
                                        persist=True, verbose=False,
                                        track_buffer=self.track_buffer,
                                        match_thresh=self.match_thresh)[0]
            except Exception:
                pass
        return self.model(frame, conf=self.conf_thresh, verbose=False)[0]

    @staticmethod
    def _extract_track_id(box):
        """Extract the unique track id attached to a detection box.

        Args:
            box: Ultralytics box object (may or may not carry an id).

        Returns:
            Integer track id when tracking is active, else None.
        """
        try:
            track_id = box.id
            if track_id is not None:
                return int(track_id[0].item())
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
        return None

    @staticmethod
    def bbox_iou(box_a, box_b):
        """IoU between two absolute ``(x1, y1, x2, y2)`` boxes (floats)."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def detect_faces(self, frame):
        """Run the configured DeepFace face detector on the raw video frame.

        Produces explicit face bounding boxes from face landmark/detection
        outputs (retinaface, mtcnn or opencv backends) instead of relying
        on percentage-based body slicing later in the pipeline.

        Args:
            frame: Current video frame (BGR).

        Returns:
            List of ``(x1, y1, x2, y2, confidence)`` face boxes in absolute
            frame coordinates; empty when the stage is disabled or the
            backend is unavailable.
        """
        if self.face_detector_backend is None:
            return []
        if self._face_detector is None:
            try:
                from deepface import DeepFace
                self._face_detector = DeepFace
            except Exception:
                self._face_detector = False
        if not self._face_detector:
            return []
        try:
            results = self._face_detector.extract_faces(
                img_path=frame, detector_backend=self.face_detector_backend,
                enforce_detection=False, align=False)
            boxes = []
            for result in results:
                area = result.get('facial_area')
                if not area:
                    continue
                confidence = float(result.get('confidence') or 0.0)
                fx, fy, fw, fh = area['x'], area['y'], area['w'], area['h']
                boxes.append((fx, fy, fx + fw, fy + fh, confidence))
            return boxes
        except Exception:
            return []

    def _assign_face_boxes(self, face_boxes, persons):
        """Attach detected face boxes to the closest overlapping person.

        Each face box is greedily assigned to the tracked person whose
        bounding box has the highest IoU with it (min IoU 0.05), so a
        person never consumes another person's face. Persons keep an
        explicit ``face_bbox`` used later for direct face cropping.

        Args:
            face_boxes: List of ``(x1, y1, x2, y2, confidence)`` boxes.
            persons: List of person telemetry items (mutated in place).
        """
        for fx1, fy1, fx2, fy2, _confidence in face_boxes:
            best_person = None
            best_iou = 0.0
            for person in persons:
                if 'face_bbox' in person:
                    continue
                iou = self.bbox_iou(person['bbox'], [fx1, fy1, fx2, fy2])
                if iou > best_iou:
                    best_iou = iou
                    best_person = person
            if best_person is not None and best_iou >= 0.05:
                best_person['face_bbox'] = [fx1, fy1, fx2, fy2]

    def process_frame(self, frame):
        results = self._run_inference(frame)
        
        pedestrians = []
        rider_eligible_vehicles = []
        enclosed_vehicles = []
        
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id not in self.target_classes:
                    continue
                    
                label = self.target_classes[cls_id]
                bbox = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].item())
                
                item = {
                    'bbox': bbox,
                    'confidence': round(conf, 2),
                    'class': label
                }

                track_id = self._extract_track_id(box)
                if track_id is not None:
                    item['track_id'] = track_id
                
                if label == 'person':
                    pedestrians.append(item)
                elif label in self.rider_vehicle_classes:
                    rider_eligible_vehicles.append(item)
                elif label in self.enclosed_vehicle_classes:
                    enclosed_vehicles.append(item)

        # Collapse overlapping / duplicate person detections (e.g. a
        # partially occluded employee) into a single box before association,
        # so one physical person does not produce double bounding boxes
        # (and duplicate face crops downstream).
        if len(pedestrians) > 1:
            keep = non_max_suppression(
                [ped['bbox'] for ped in pedestrians],
                [ped['confidence'] for ped in pedestrians],
                iou_thresh=self.nms_iou_thresh)
            pedestrians = [pedestrians[i] for i in keep]

        active_riders = []
        standalone_pedestrians = []
        
        for ped in pedestrians:
            is_rider = False
            best_iop = 0.0
            matched_vehicle = None
            
            for veh in rider_eligible_vehicles:
                iop = self.compute_feet_iop(ped['bbox'], veh['bbox'])
                if iop >= self.iop_thresh and iop > best_iop:
                    best_iop = iop
                    is_rider = True
                    matched_vehicle = veh
                    
            if is_rider and matched_vehicle is not None:
                ped_copy = dict(ped)
                ped_copy['rider_of'] = matched_vehicle['class']
                ped_copy['vehicle_bbox'] = matched_vehicle['bbox']
                ped_copy['iop'] = round(best_iop, 2)
                active_riders.append(ped_copy)
            else:
                standalone_pedestrians.append(ped)

        # Explicit face detection: detect faces directly on the raw frame and
        # bind each detected face box to the person it belongs to, enabling
        # direct face cropping (no body-ratio head slicing) downstream.
        face_boxes = self.detect_faces(frame)
        if face_boxes:
            self._assign_face_boxes(face_boxes,
                                    standalone_pedestrians + active_riders)

        all_vehicles = rider_eligible_vehicles + enclosed_vehicles
        
        telemetry = {
            'counts': {
                'pedestrians': len(standalone_pedestrians),
                'active_riders': len(active_riders),
                'total_vehicles': len(all_vehicles),
                'breakdown': {
                    'bicycle': sum(1 for v in all_vehicles if v['class'] == 'bicycle'),
                    'car': sum(1 for v in all_vehicles if v['class'] == 'car'),
                    'motorcycle': sum(1 for v in all_vehicles if v['class'] == 'motorcycle'),
                    'bus': sum(1 for v in all_vehicles if v['class'] == 'bus'),
                    'truck': sum(1 for v in all_vehicles if v['class'] == 'truck')
                }
            },
            'pedestrians': standalone_pedestrians,
            'active_riders': active_riders,
            'vehicles': all_vehicles
        }
        
        return telemetry

# Alias for backward compatibility with main.py
CityMobilityDetector = CityDetector
