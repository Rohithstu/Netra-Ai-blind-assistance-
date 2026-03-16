"""
Netra AI — Layer 7: Emotion Detection & Human Behavior Understanding Engine
Detects facial emotions, mixed emotions, intensity levels, facial landmarks,
and integrates with face recognition (Layer 6) and depth estimation (Layer 3).
"""
import cv2  # type: ignore
import torch  # type: ignore
import numpy as np  # type: ignore
import time
import json
import argparse
import os

try:
    from fer import FER
except ImportError:
    from fer.fer import FER  # Fallback for some versions

# Optional imports for integration with other layers
try:
    from face_engine import FaceEngine  # type: ignore
    FACE_ENGINE_AVAILABLE = True
except ImportError:
    FACE_ENGINE_AVAILABLE = False

try:
    from depth_engine import DepthEngine  # type: ignore
    DEPTH_AVAILABLE = True
except ImportError:
    DEPTH_AVAILABLE = False


class EmotionEngine:
    """Facial emotion detection and analysis engine for Netra AI."""
    
    # Primary emotions
    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    
    # Extended emotion combinations (mixed emotions)
    MIXED_EMOTION_MAP = {
        ("happy", "surprise"): "happy-surprised",
        ("sad", "fear"): "sad-fearful",
        ("angry", "disgust"): "angry-disgusted",
        ("sad", "neutral"): "tired",
        ("surprise", "fear"): "shocked",
        ("happy", "neutral"): "content",
        ("angry", "surprise"): "frustrated",
        ("sad", "surprise"): "confused",
    }
    
    # Intensity thresholds
    INTENSITY_HIGH = 0.75
    INTENSITY_MEDIUM = 0.45
    
    # Priority filter: ignore faces beyond this distance
    MAX_REPORT_DISTANCE = 4.0  # meters
    
    # Crowd threshold
    CROWD_THRESHOLD = 5
    
    def __init__(self, use_depth=True, use_face_recognition=True):
        """
        Initialize the emotion detection system.
        
        Args:
            use_depth: Integrate depth estimation for distance filtering.
            use_face_recognition: Integrate face recognition for identity.
        """
        print("=" * 60)
        print("  NETRA AI — Emotion Detection Engine (Layer 7)")
        print("=" * 60)
        
        # Core emotion detector (FER uses a CNN-based classifier)
        print("\n📦 Loading FER emotion classifier...")
        self.emotion_detector = FER(mtcnn=True)
        
        # Depth engine (optional)
        self.depth_engine = None  # type: ignore
        if use_depth and DEPTH_AVAILABLE:
            print("📦 Loading depth engine...")
            self.depth_engine = DepthEngine(model_type="MiDaS_small")
        
        # Face recognition engine (optional)
        self.face_engine = None  # type: ignore
        if use_face_recognition and FACE_ENGINE_AVAILABLE:
            print("📦 Loading face recognition engine...")
            self.face_engine = FaceEngine(use_depth=False)
        
        print("\n✅ Emotion Engine ready.\n")
    
    def detect_emotions(self, frame):
        """
        Detect emotions for all faces in frame.
        
        Args:
            frame: BGR OpenCV image.
            
        Returns:
            List of dicts with box, emotions scores, dominant emotion.
        """
        results = self.emotion_detector.detect_emotions(frame)
        return results
    
    def get_mixed_emotion(self, emotions_dict):
        """
        Detect mixed emotions from the emotion scores.
        
        Args:
            emotions_dict: Dict of {emotion: score} from FER.
            
        Returns:
            mixed_label: String like 'happy-surprised' or single emotion.
            top_emotions: List of (emotion, score) tuples sorted by score.
        """
        sorted_emotions = sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)
        top1 = sorted_emotions[0]
        top2 = sorted_emotions[1] if len(sorted_emotions) > 1 else None
        
        # Check if two emotions are both strong enough to be mixed
        if top2 and top2[1] > 0.20 and top1[1] - top2[1] < 0.30:
            key = tuple(sorted([top1[0], top2[0]]))
            mixed_label = self.MIXED_EMOTION_MAP.get(key, f"{top1[0]}-{top2[0]}")  # type: ignore
            return mixed_label, sorted_emotions
        
        return top1[0], sorted_emotions
    
    def get_intensity(self, confidence):
        """
        Determine emotion intensity from confidence score.
        
        Args:
            confidence: Float 0.0 to 1.0.
            
        Returns:
            'high', 'medium', or 'low'.
        """
        if confidence >= self.INTENSITY_HIGH:
            return "high"
        elif confidence >= self.INTENSITY_MEDIUM:
            return "medium"
        else:
            return "low"
    
    def get_zone(self, bbox, frame_width):
        """Determine spatial zone (left/center/right)."""
        x1, _, x2_offset, _ = bbox  # FER returns (x, y, w, h)
        center_x = x1 + x2_offset / 2.0
        ratio = center_x / frame_width
        
        if ratio < 0.33:
            return "left"
        elif ratio < 0.66:
            return "center"
        else:
            return "right"
    
    def should_report(self, intensity, distance):
        """
        Priority filter: decide whether to report this emotion.
        
        Rules:
        - Only report strong emotions
        - Ignore distant faces (> 4m)
        - Always report high intensity
        """
        if distance is not None and distance > self.MAX_REPORT_DISTANCE:
            return False
        if intensity == "low":
            return False
        return True
    
    def process_frame(self, frame):
        """
        Full emotion analysis pipeline for a single frame.
        
        Args:
            frame: BGR OpenCV image.
            
        Returns:
            annotated_frame: Frame with emotion labels and bounding boxes.
            detections: List of structured emotion detection dicts.
        """
        h, w = frame.shape[:2]
        annotated = frame.copy()
        
        # Get depth map if available
        depth_map = None
        if self.depth_engine:
            depth_map = self.depth_engine.estimate_depth(frame)  # type: ignore
        
        # Detect emotions
        emotion_results = self.detect_emotions(frame)
        
        # Get face identities if available
        face_identities = {}
        if self.face_engine:
            _, face_detections = self.face_engine.process_frame(frame)  # type: ignore
            for fd in face_detections:
                # Map by approximate center position
                bx = (fd["bounding_box"][0] + fd["bounding_box"][2]) // 2
                by = (fd["bounding_box"][1] + fd["bounding_box"][3]) // 2
                face_identities[(bx // 20, by // 20)] = fd.get("person", "Unknown")
        
        detections = []
        
        # Crowd detection
        if len(emotion_results) > self.CROWD_THRESHOLD:
            print(f"  👥 Crowd detected: {len(emotion_results)} people")
            cv2.putText(annotated, f"CROWD: {len(emotion_results)} people", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        for result in emotion_results:
            box = result["box"]  # (x, y, w, h)
            emotions = result["emotions"]
            
            x, y, bw, bh = box
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            
            # Get mixed emotion and intensity
            emotion_label, sorted_emotions = self.get_mixed_emotion(emotions)
            top_confidence = sorted_emotions[0][1]
            intensity = self.get_intensity(top_confidence)
            
            # Get zone
            zone = self.get_zone(box, w)
            
            # Get distance if depth available
            distance = None
            if depth_map is not None and self.depth_engine:
                distance = self.depth_engine.get_object_distance(depth_map, [x1, y1, x2, y2])  # type: ignore
            
            # Try to match with known face identity
            center_key = ((x1 + x2) // 2 // 20, (y1 + y2) // 2 // 20)
            person_name = face_identities.get(center_key, "Unknown person")
            
            # Priority filter
            reportable = self.should_report(intensity, distance)
            
            # Build detection data
            detection = {
                "person": person_name,
                "emotion": emotion_label,
                "intensity": intensity,
                "confidence": round(float(top_confidence), 2),  # type: ignore
                "direction": zone,
                "reportable": reportable,
                "all_emotions": {k: round(float(v), 3) for k, v in emotions.items()},  # type: ignore
                "timestamp": round(float(time.time()), 3)  # type: ignore
            }
            if distance is not None:
                detection["distance"] = distance
            
            detections.append(detection)
            
            # Draw on frame
            if intensity == "high":
                color = (0, 0, 255)    # Red for strong emotions
            elif intensity == "medium":
                color = (0, 165, 255)  # Orange
            else:
                color = (200, 200, 200)  # Gray for mild
            
            # Intensity prefix
            intensity_prefix = ""
            if intensity == "high":
                intensity_prefix = "VERY "
            
            label = f"{person_name}: {intensity_prefix}{emotion_label}"
            if distance is not None:
                label += f" — {distance}m"
            label += f" [{zone}]"
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Console output (only reportable)
            if reportable:
                dist_str = f" — {distance}m" if distance is not None else ""
                print(f"  {person_name} looks {intensity_prefix.lower()}{emotion_label}{dist_str} [{zone}]")
        
        return annotated, detections


def run_camera(engine):
    """Real-time emotion detection on webcam."""
    print("🎥 Starting Emotion Detection Camera Mode (Press 'q' to exit)...")
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated, detections = engine.process_frame(frame)
        
        cv2.imshow("Netra AI — Emotion Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def run_video(engine, input_path):
    """Process a video file through emotion detection."""
    print(f"🎬 Processing Video: {input_path}")
    cap = cv2.VideoCapture(input_path)
    
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    name, _ = os.path.splitext(os.path.basename(input_path))
    output_path = f"output_{name}_emotions.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))
    
    all_detections = []  # type: list
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated, detections = engine.process_frame(frame)
        out.write(annotated)
        
        for d in detections:
            d["frame"] = frame_idx
        all_detections.extend(detections)
        frame_idx += 1
    
    cap.release()
    out.release()
    
    log_path = f"output_{name}_emotions_log.json"
    with open(log_path, 'w') as f:
        json.dump(all_detections, f, indent=2)
    
    print(f"✅ Emotion video saved to: {output_path}")
    print(f"📄 Detection log: {log_path} ({len(all_detections)} detections)")


def run_image(engine, input_path):
    """Process a single image through emotion detection."""
    print(f"🖼️ Processing Image: {input_path}")
    frame = cv2.imread(input_path)
    if frame is None:
        print("❌ Could not load image.")
        return
    
    annotated, detections = engine.process_frame(frame)
    
    name, ext = os.path.splitext(os.path.basename(input_path))
    output_path = f"output_{name}_emotions{ext}"
    cv2.imwrite(output_path, annotated)
    
    print(f"\n📊 Emotion Analysis Results:")
    print(json.dumps(detections, indent=2))
    print(f"\n✅ Saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Netra AI — Emotion Detection Engine (Layer 7)")
    parser.add_argument('--mode', type=str, choices=['camera', 'video', 'image'], default='camera')
    parser.add_argument('--input', type=str, default='', help="Path to input video or image")
    parser.add_argument('--no_depth', action='store_true', help="Disable depth integration")
    parser.add_argument('--no_face', action='store_true', help="Disable face identity integration")
    args = parser.parse_args()
    
    engine = EmotionEngine(use_depth=not args.no_depth, use_face_recognition=not args.no_face)
    
    if args.mode == 'camera':
        run_camera(engine)
    elif args.mode == 'video':
        if not args.input:
            print("❌ --input is required for video mode.")
        else:
            run_video(engine, args.input)
    elif args.mode == 'image':
        if not args.input:
            print("❌ --input is required for image mode.")
        else:
            run_image(engine, args.input)
