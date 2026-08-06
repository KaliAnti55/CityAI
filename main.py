"""
City Mobility AI - Urban Transportation & Pedestrian Analytics
Main CLI Execution Entry Point
"""

import argparse
import json
import os
import sys
import cv2

# Import project modules
from src.detector import CityMobilityDetector
from src.visualizer import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart City Transportation & Human Association Detector"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to input image, video file, or camera stream (e.g., sample.mp4 or '0' for webcam)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8x.pt",
        help="YOLO model weights path or name (default: yolov8x.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Confidence detection threshold (default: 0.45)",
    )
    parser.add_argument(
        "--iop-thresh",
        type=float,
        default=0.40,
        help="Intersection over Person (IoP) threshold for human-vehicle link (default: 0.40)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/outputs/output.mp4",
        help="Path to save processed output video or image",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Export analytics metadata to a JSON file",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable live OpenCV window display (useful for Google Colab/Headless servers)",
    )
    return parser.parse_args()


def process_image(args, detector, visualizer):
    """Processes a single static image."""
    frame = cv2.imread(args.source)
    if frame is None:
        sys.exit(f"Error: Could not read image file at {args.source}")

    annotated_frame, metadata = detector.analyze_frame(
        frame, confidence=args.conf
    )
    annotated_frame = visualizer.draw_dashboard(annotated_frame, metadata)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, annotated_frame)
    print(f"[+] Output image saved to: {args.output}")

    if args.save_json:
        json_path = os.path.splitext(args.output)[0] + ".json"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"[+] Metadata saved to: {json_path}")


def process_video(args, detector, visualizer):
    """Processes a video file or webcam stream frame-by-frame."""
    # Handle webcam source integer vs video path string
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        sys.exit(f"Error: Unable to open video source '{args.source}'")

    # Fetch stream properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_count = 0
    full_telemetry = []

    print(f"[*] Starting inference stream with model: {args.weights}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        annotated_frame, metadata = detector.analyze_frame(
            frame, confidence=args.conf
        )
        metadata["frame_id"] = frame_count

        annotated_frame = visualizer.draw_dashboard(annotated_frame, metadata)
        out.write(annotated_frame)

        if args.save_json:
            full_telemetry.append(metadata)

        # Display window if not running headless/colab
        if not args.no_display:
            cv2.imshow("Urban Mobility AI Monitor", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[!] Stream interrupted by user.")
                break

        if frame_count % 30 == 0:
            print(
                f"Processed frame {frame_count} | Vehicles: {metadata['counts']['total_vehicles']} | Pedestrians: {metadata['counts']['pedestrians']} | Active Riders: {metadata['counts']['active_riders']}"
            )

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"[+] Processed video saved to: {args.output}")

    if args.save_json and full_telemetry:
        json_path = os.path.splitext(args.output)[0] + ".json"
        with open(json_path, "w") as f:
            json.dump(full_telemetry, f, indent=4)
        print(f"[+] Full video telemetry saved to: {json_path}")


def main():
    args = parse_args()

    # Initialize model engine
    detector = CityMobilityDetector(
        model_path=args.weights, iop_threshold=args.iop_thresh
    )
    visualizer = Visualizer()

    # Determine source type (Image vs Video/Webcam)
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if isinstance(args.source, str) and args.source.lower().endswith(
        image_extensions
    ):
        process_image(args, detector, visualizer)
    else:
        process_video(args, detector, visualizer)


if __name__ == "__main__":
    main()