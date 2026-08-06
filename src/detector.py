import numpy as np

class CityMobilityDetector:
    def __init__(self, model, conf_thresh=0.25, iop_thresh=0.45):
        self.model = model
        self.conf_thresh = conf_thresh
        self.iop_thresh = iop_thresh
        
        # Micro-mobility classes that allow active riders
        self.rider_vehicle_classes = ['bicycle', 'motorcycle']
        
        # Enclosed / heavy vehicles (tracked separately, cannot have open riders)
        self.enclosed_vehicle_classes = ['car', 'bus', 'truck']
        
        # Target COCO class mapping
        self.target_classes = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }

    def compute_feet_iop(self, ped_box, veh_box):
        """
        Calculates Intersection over Person (IoP) focusing strictly on 
        the pedestrian's lower body/feet area to eliminate perspective depth stacking.
        """
        px1, py1, px2, py2 = ped_box
        vx1, vy1, vx2, vy2 = veh_box
        
        # Focus on the lower 25% of the pedestrian bounding box (feet/base contact)
        ped_height = py2 - py1
        feet_py1 = py2 - (ped_height * 0.25)
        
        # Calculate intersection area between pedestrian feet zone and vehicle box
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

    def process_frame(self, frame):
        results = self.model(frame, conf=self.conf_thresh, verbose=False)[0]
        
        pedestrians = []
        rider_eligible_vehicles = []
        enclosed_vehicles = []
        
        # 1. Parse detected objects into categories
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
                
                if label == 'person':
                    pedestrians.append(item)
                elif label in self.rider_vehicle_classes:
                    rider_eligible_vehicles.append(item)
                elif label in self.enclosed_vehicle_classes:
                    enclosed_vehicles.append(item)
                    
        active_riders = []
        standalone_pedestrians = []
        
        # 2. Match pedestrians to open micro-mobility vehicles using feet-anchored IoP
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
                
        all_vehicles = rider_eligible_vehicles + enclosed_vehicles
        
        # 3. Build telemetry object
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
    
