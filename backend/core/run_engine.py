import cv2
import os
import argparse
import time
import json
from ultralytics import YOLO

def process_frame(model, frame):
    """Run inference and return drawn frame + structured JSON data."""
    results = model(frame, verbose=False)[0] # Run inference on a single frame
    
    structured_data = []
    
    for box in results.boxes:
        conf = float(box.conf[0])
        # Only process high confidence predictions
        if conf > 0.5:
            # Extract coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            object_name = results.names[class_id]
            
            # Format output dictionary according to requirements
            detection = {
                "object_name": object_name,
                "confidence": round(conf, 2),  # type: ignore
                "bounding_box": [x1, y1, x2, y2],
                "timestamp": round(float(time.time()), 3)  # type: ignore
            }
            structured_data.append(detection)
            
            # Draw on OpenCV Frame
            label_text = f"{object_name.capitalize()} ({conf:.2f})"
            color = (0, 255, 0) # Green bounding boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label_text, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Print expected output log
            print(f"{object_name.capitalize()} ({conf:.2f})")
            
    return frame, structured_data

def run_camera(model):
    print("🎥 Starting Real-Time Camera Detection (Press 'q' to exit)...")
    cap = cv2.VideoCapture(0) # Open default webcam
    target_fps = 30
    delay = int(1000 / target_fps)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame, sys_data = process_frame(model, frame)
        
        # Display the result
        cv2.imshow("Netra Vision AI - Camera Mode", processed_frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def run_video(model, input_path):
    print(f"🎬 Processing Video: {input_path}")
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    output_path = f"output_{name}_with_boxes.mp4"
    
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame, sys_data = process_frame(model, frame)
        
        # At this point `sys_data` could be fired off through a WebSocket to Navigation layer
        out.write(processed_frame)
        frame_count += 1
        
    cap.release()
    out.release()
    print(f"✅ Video processing complete. Saved to: {output_path}")

def run_image(model, input_path):
    print(f"🖼️ Processing Image: {input_path}")
    frame = cv2.imread(input_path)
    if frame is None:
         print("❌ Error loading image.")
         return
         
    processed_frame, sys_data = process_frame(model, frame)
    
    # Save Output
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    output_path = f"output_{name}_with_boxes{ext}"
    cv2.imwrite(output_path, processed_frame)
    
    print(f"✅ Image processed. JSON Metadata: {json.dumps(sys_data, indent=2)}")
    print(f"Saved drawn image to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Netra Vision AI Real-Time Detection Engine")
    parser.add_argument('--mode', type=str, choices=['camera', 'video', 'image'], default='camera', help="Mode to run the inference engine")
    parser.add_argument('--input', type=str, default='', help="Path to input video or image if not in camera mode")
    parser.add_argument('--model', type=str, default='', help="Path to custom weights (default: models/vision/netra_vision_model.pt)")
    args = parser.parse_args()

    # Determine model path
    model_path = args.model
    if not model_path:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_path = os.path.join(base_dir, 'models', 'vision', 'netra_vision_model.pt')
        
    print(f"🧠 Loading Network Configuration from {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"[!] Warning: Custom model `{model_path}` not found. Falling back to default pre-trained YOLOv8s for demonstration.")
        model = YOLO('yolov8s.pt')
    else:
        # Load exported or trained YOLOv8 model directly
        model = YOLO(model_path)

    if args.mode == 'camera':
        run_camera(model)
    elif args.mode == 'video':
        if not args.input:
            print("❌ Error: --input is required for video mode.")
        else:
            run_video(model, args.input)
    elif args.mode == 'image':
        if not args.input:
            print("❌ Error: --input is required for image mode.")
        else:
            run_image(model, args.input)
