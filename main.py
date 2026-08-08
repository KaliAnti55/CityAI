import argparse
import os
import json
import cv2
from src.detector import CityMobilityDetector
from src.visualizer import Visualizer
from src.memory import CropMemory

def record_face_crops(frame, telemetry, memory):
    """Persist face crops for stable tracked pedestrians and annotate telemetry.

    For every person (standalone pedestrian or active rider) carrying a
    ``track_id``, requests a one-time face crop save from ``memory``
    (which only fires after the track has been stable for consecutive
    frames) and, on first save, attaches the resulting crop path to the
    telemetry entry. Person track ids observed this frame are reported so
    the memory module can reset streaks of tracks that disappeared.
    """
    persons = telemetry.get('pedestrians', []) + telemetry.get('active_riders', [])
    person_ids = set()
    for person in persons:
        track_id = person.get('track_id')
        if track_id is None:
            continue
        person_ids.add(track_id)
        crop_path = memory.save_face_crop(frame, person['bbox'], track_id)
        if crop_path is not None:
            person['face_crop_path'] = crop_path
    memory.mark_frame_observations(person_ids)

def record_vehicle_crops(frame, telemetry, memory):
    """Persist vehicle and license plate crops for newly tracked vehicles.

    A full vehicle crop is saved once per unique ``track_id``; for enclosed
    vehicle classes (car, bus, truck) a license plate region crop is saved
    alongside it. Crop paths are attached to the vehicle telemetry entries
    on first save.
    """
    for vehicle in telemetry.get('vehicles', []):
        track_id = vehicle.get('track_id')
        if track_id is None:
            continue
        vehicle_class = vehicle['class']
        crop_path = memory.save_vehicle_crop(frame, vehicle['bbox'], track_id, vehicle_class)
        if crop_path is not None:
            vehicle['vehicle_crop_path'] = crop_path
        if vehicle_class in memory.ENCLOSED_VEHICLE_CLASSES:
            plate_path = memory.save_plate_crop(frame, vehicle['bbox'], track_id, vehicle_class)
            if plate_path is not None:
                vehicle['plate_crop_path'] = plate_path

def parse_args():
    parser = argparse.ArgumentParser(description="CityAI - Urban Mobility Analytics")
    parser.add_argument("--source", type=str, default="0", help="Video path or webcam index (0)")
    parser.add_argument("--weights", type=str, default="yolov8x.pt", help="YOLO model path")
    parser.add_argument("--plate-weights", type=str,
                        default="models/keremberke_yolov8n-license-plate.pt",
                        help="Secondary license plate YOLO weights (falls back to geometric heuristic if unavailable)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iop-thresh", type=float, default=0.45, help="Feet IoP threshold for riders")
    parser.add_argument("--no-display", action="store_true", help="Disable GUI display window")
    parser.add_argument("--save-json", action="store_true", help="Save output telemetry JSON")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Handle source type (numeric string for camera index vs video file path)
    source = int(args.source) if args.source.isdigit() else args.source
    
    detector = CityMobilityDetector(model_path=args.weights, conf_thresh=args.conf, iop_thresh=args.iop_thresh)
    visualizer = Visualizer()
    memory = CropMemory(plate_model=args.plate_weights)
    if memory.plate_model is None:
        print(f"[-] Plate model '{args.plate_weights}' unavailable; "
              f"falling back to geometric plate heuristic")
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[-] Error: Could not open video source '{args.source}'")
        return

    os.makedirs("data/outputs", exist_ok=True)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    out_video_path = "data/outputs/output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    last_telemetry = None

    print(f"[*] Starting inference stream with model: {args.weights}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        telemetry = detector.process_frame(frame)
        record_face_crops(frame, telemetry, memory)
        record_vehicle_crops(frame, telemetry, memory)
        last_telemetry = telemetry
        
        annotated_frame = visualizer.draw(frame, telemetry)
        out_video.write(annotated_frame)
        
        if frame_count % 30 == 0:
            c = telemetry['counts']
            print(f"Processed frame {frame_count:4d} | Vehicles: {c['total_vehicles']} | "
                  f"Pedestrians: {c['pedestrians']} | Active Riders: {c['active_riders']}")
            
        if not args.no_display:
            cv2.imshow("Urban Mobility AI Monitor", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    
    print(f"[+] Processed video saved to: {out_video_path}")
    memory_stats = memory.summary()
    print(f"[+] Unique face crops saved: {memory_stats['unique_faces_saved']} "
          f"-> {memory_stats['faces_dir']}")
    print(f"[+] Unique vehicle crops saved: {memory_stats['unique_vehicles_saved']} "
          f"-> {memory_stats['vehicles_dir']}")
    print(f"[+] Unique plate crops saved: {memory_stats['unique_plates_saved']} "
          f"-> {memory_stats['plates_dir']}")
    
    if args.save_json and last_telemetry is not None:
        json_path = "data/outputs/output.json"
        with open(json_path, 'w') as f:
            json.dump(last_telemetry, f, indent=4)
        print(f"[+] Full video telemetry saved to: {json_path}")

if __name__ == "__main__":
    main()
