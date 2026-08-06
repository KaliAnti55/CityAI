import argparse
import os
import json
import cv2
from src.detector import CityMobilityDetector
from src.visualizer import Visualizer

def parse_args():
    parser = argparse.ArgumentParser(description="CityAI - Urban Mobility Analytics")
    parser.add_argument("--source", type=str, default="0", help="Video path or webcam index (0)")
    parser.add_argument("--weights", type=str, default="yolov8x.pt", help="YOLO model path")
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
    
    if args.save_json and last_telemetry is not None:
        json_path = "data/outputs/output.json"
        with open(json_path, 'w') as f:
            json.dump(last_telemetry, f, indent=4)
        print(f"[+] Full video telemetry saved to: {json_path}")

if __name__ == "__main__":
    main()
