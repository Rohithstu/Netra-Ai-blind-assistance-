"""
Netra AI — Layer 3: Vision + Depth Fusion Engine
Combines YOLOv8 object detection (Layer 2) with MiDaS depth estimation (Layer 3)
to produce structured navigation data with object names, distances, zones, and risk levels.
"""
import cv2  # type: ignore
import time
import json
import argparse
import os
import numpy as np  # type: ignore
from ultralytics import YOLO  # type: ignore
from depth_engine import DepthEngine


class FusionEngine:
    """Fuses YOLO object detection with MiDaS depth estimation."""
    
    def __init__(self, yolo_model_path=None, depth_model_type="MiDaS_small"):
        """
        Initialize both detection and depth models.
        
        Args:
            yolo_model_path: Path to YOLO weights (.pt). Falls back to pretrained yolov8s.
            depth_model_type: MiDaS variant ('MiDaS_small', 'DPT_Hybrid', 'DPT_Large').
        """
        print("=" * 60)
        print("  NETRA AI — Fusion Engine (Vision + Depth)")
        print("=" * 60)
        
        # Load YOLO
        if yolo_model_path and os.path.exists(yolo_model_path):
            print(f"\n📦 Loading custom YOLO model: {yolo_model_path}")
            self.yolo = YOLO(yolo_model_path)
        else:
            print("\n📦 Loading pretrained YOLOv8s model...")
            self.yolo = YOLO("yolov8s.pt")
        
        # Load Depth Engine
        print()
        self.depth = DepthEngine(model_type=depth_model_type)
        
        print("\n✅ Fusion Engine ready.\n")
    
    def process_frame(self, frame, conf_threshold=0.5):
        """
        Run full fusion pipeline on a single frame.
        
        Args:
            frame: BGR OpenCV image.
            conf_threshold: Minimum detection confidence.
            
        Returns:
            annotated_frame: Frame with boxes, labels, and depth overlay.
            detections: List of structured detection dictionaries.
        """
        h, w = frame.shape[:2]
        
        # 1. Run depth estimation
        depth_map = self.depth.estimate_depth(frame)
        
        # 2. Run YOLO object detection
        results = self.yolo(frame, verbose=False)[0]
        
        # 3. Create depth overlay as base visualization
        annotated_frame = self.depth.create_depth_overlay(frame, depth_map)
        
        detections = []
        
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            object_name = results.names[class_id]
            
            # 4. Get depth-based distance for this object
            distance = self.depth.get_object_distance(depth_map, [x1, y1, x2, y2])
            
            # 5. Get spatial zone
            zone = self.depth.get_zone([x1, y1, x2, y2], w)
            
            # 6. Get risk level
            risk = self.depth.get_risk_level(distance)
            
            # Build structured output
            detection = {
                "object": object_name,
                "distance": distance,
                "direction": zone,
                "confidence": round(conf, 2),  # type: ignore
                "risk_level": risk,
                "bounding_box": [x1, y1, x2, y2],
                "timestamp": round(float(time.time()), 3)  # type: ignore
            }
            detections.append(detection)
            
            # 7. Draw on frame
            # Color by risk level
            if risk == "high":
                color = (0, 0, 255)       # Red
                label_prefix = "⚠️ "
            elif risk == "medium":
                color = (0, 165, 255)     # Orange
                label_prefix = ""
            else:
                color = (0, 255, 0)       # Green
                label_prefix = ""
            
            label = f"{label_prefix}{object_name} — {distance}m ({zone})"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            
            # Console output
            risk_tag = f" [COLLISION RISK]" if risk == "high" else ""
            print(f"  {object_name.capitalize()} — {distance}m | {zone}{risk_tag}")
        
        return annotated_frame, detections


def run_camera(fusion):
    """Real-time fusion detection on webcam."""
    print("🎥 Starting Fusion Camera Mode (Press 'q' to exit)...")
    cap = cv2.VideoCapture(0)
    
    fps_counter = 0
    fps_time = time.time()
    display_fps = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated, detections = fusion.process_frame(frame)
        
        # FPS counter
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            display_fps = fps_counter
            fps_counter = 0
            fps_time = time.time()
        
        cv2.putText(annotated, f"FPS: {display_fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow("Netra AI — Fusion Engine (Vision + Depth)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def run_video(fusion, input_path):
    """Process a video file through the fusion engine."""
    print(f"🎬 Processing Video: {input_path}")
    cap = cv2.VideoCapture(input_path)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    name, _ = os.path.splitext(os.path.basename(input_path))
    output_path = f"output_{name}_fusion.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    all_detections = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated, detections = fusion.process_frame(frame)
        out.write(annotated)
        
        for d in detections:
            d["frame"] = frame_idx
        all_detections.extend(detections)
        frame_idx += 1
    
    cap.release()
    out.release()
    
    # Save detection log
    log_path = f"output_{name}_fusion_log.json"
    with open(log_path, 'w') as f:
        json.dump(all_detections, f, indent=2)
    
    print(f"✅ Fusion video saved to: {output_path}")
    print(f"📄 Detection log saved to: {log_path} ({len(all_detections)} detections)")


def run_image(fusion, input_path):
    """Process a single image through the fusion engine."""
    print(f"🖼️ Processing Image: {input_path}")
    frame = cv2.imread(input_path)
    if frame is None:
        print("❌ Could not load image.")
        return
    
    annotated, detections = fusion.process_frame(frame)
    
    name, ext = os.path.splitext(os.path.basename(input_path))
    output_path = f"output_{name}_fusion{ext}"
    cv2.imwrite(output_path, annotated)
    
    print(f"\n📊 Structured Navigation Data:")
    print(json.dumps(detections, indent=2))
    print(f"\n✅ Annotated image saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Netra AI — Fusion Engine (Vision + Depth)")
    parser.add_argument('--mode', type=str, choices=['camera', 'video', 'image'], default='camera')
    parser.add_argument('--input', type=str, default='', help="Path to input video or image")
    parser.add_argument('--yolo_model', type=str, default='', help="Path to custom YOLO weights")
    parser.add_argument('--depth_model', type=str, default='MiDaS_small',
                        choices=['DPT_Large', 'DPT_Hybrid', 'MiDaS_small'])
    args = parser.parse_args()
    
    # Resolve YOLO model
    yolo_path = args.yolo_model
    if not yolo_path:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        yolo_path = os.path.join(base_dir, 'models', 'vision', 'netra_vision_model.pt')
    
    fusion = FusionEngine(yolo_model_path=yolo_path, depth_model_type=args.depth_model)
    
    if args.mode == 'camera':
        run_camera(fusion)
    elif args.mode == 'video':
        if not args.input:
            print("❌ --input is required for video mode.")
        else:
            run_video(fusion, args.input)
    elif args.mode == 'image':
        if not args.input:
            print("❌ --input is required for image mode.")
        else:
            run_image(fusion, args.input)
