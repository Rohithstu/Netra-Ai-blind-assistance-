"""
Netra AI — Layer 7: Human Behavior Analysis Engine
Detects body posture, movement patterns, social interactions, and behavioral cues
using MediaPipe pose estimation and heuristic analysis.
"""
import cv2  # type: ignore
import numpy as np  # type: ignore
import time
import json
import argparse
import os

try:
    import mediapipe as mp  # type: ignore
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    from depth_engine import DepthEngine  # type: ignore
    DEPTH_AVAILABLE = True
except ImportError:
    DEPTH_AVAILABLE = False


class BehaviorEngine:
    """Human behavior and body language analysis engine for Netra AI."""
    
    def __init__(self, use_depth=True):
        print("=" * 60)
        print("  NETRA AI — Behavior Analysis Engine (Layer 7)")
        print("=" * 60)
        
        if not MEDIAPIPE_AVAILABLE:
            print("⚠️ MediaPipe not available. Install with: pip install mediapipe")
            self.detector = None
            self.mp_pose = None
            self.mp_draw = None
        else:
            # Use modern Tasks API
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import mediapipe as mp
            import mediapipe.python.solutions.pose as mp_pose
            import mediapipe.python.solutions.drawing_utils as mp_draw
            
            self.mp_pose = mp_pose
            self.mp_draw = mp_draw
            
            # Use cached model file if possible, or assume it's in the correct path
            model_path = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")
            if not os.path.exists(model_path):
                 print(f"⚠️ Pose model not found at {model_path}. Trying fallback...")
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE, # IMAGE mode for simple integration
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            try:
                self.detector = vision.PoseLandmarker.create_from_options(options)
            except Exception as e:
                print(f"❌ Failed to load PoseLandmarker: {e}")
                self.detector = None
        
        # Depth engine (optional)
        self.depth_engine = None  # type: ignore
        if use_depth and DEPTH_AVAILABLE:
            print("📦 Loading depth engine...")
            self.depth_engine = DepthEngine(model_type="MiDaS_small")
        
        # Track previous frame poses for movement analysis
        self.prev_landmarks = []  # type: list
        self.prev_time = time.time()
        
        print("\n✅ Behavior Engine ready.\n")
    
    def analyze_pose(self, landmarks, h, w):
        """
        Analyze body posture from pose landmarks.
        
        Args:
            landmarks: MediaPipe pose landmarks.
            h, w: Frame dimensions.
            
        Returns:
            behaviors: List of detected behavior strings.
            body_language: Interpreted body language string.
        """
        behaviors = []  # type: list
        body_language = "neutral"
        
        if not landmarks:
            return behaviors, body_language
        
        # In Tasks API, landmarks is the list of PoseLandmarkerResult.pose_landmarks
        # We take the first person
        lm = landmarks[0] # List of NormalizedLandmark
        
        # Key landmarks indices for Pose Landmarker
        NOSE = 0
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_HIP = 23
        RIGHT_HIP = 24

        nose = lm[NOSE]
        left_shoulder = lm[LEFT_SHOULDER]
        right_shoulder = lm[RIGHT_SHOULDER]
        left_elbow = lm[LEFT_ELBOW]
        right_elbow = lm[RIGHT_ELBOW]
        left_wrist = lm[LEFT_WRIST]
        right_wrist = lm[RIGHT_WRIST]
        left_hip = lm[LEFT_HIP]
        right_hip = lm[RIGHT_HIP]
        
        # --- Behavior: Waving ---
        # Hand raised above shoulder
        if (left_wrist.y < left_shoulder.y - 0.1 and left_wrist.visibility > 0.5):
            behaviors.append("waving")
        if (right_wrist.y < right_shoulder.y - 0.1 and right_wrist.visibility > 0.5):
            behaviors.append("waving")
        
        # --- Behavior: Pointing ---
        # Arm extended horizontally
        if (abs(right_wrist.y - right_shoulder.y) < 0.08 and
            abs(right_wrist.x - right_shoulder.x) > 0.2 and
            right_wrist.visibility > 0.5):
            behaviors.append("pointing")
        if (abs(left_wrist.y - left_shoulder.y) < 0.08 and
            abs(left_wrist.x - left_shoulder.x) > 0.2 and
            left_wrist.visibility > 0.5):
            behaviors.append("pointing")
        
        # --- Body language: Arms crossed ---
        if (left_wrist.visibility > 0.5 and right_wrist.visibility > 0.5):
            wrist_dist = abs(left_wrist.x - right_wrist.x)
            shoulder_dist = abs(left_shoulder.x - right_shoulder.x)
            if wrist_dist < shoulder_dist * 0.4 and left_wrist.y > left_shoulder.y:
                body_language = "defensive (arms crossed)"
                behaviors.append("arms_crossed")
        
        # --- Body language: Leaning forward ---
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        torso_angle = shoulder_center_y - hip_center_y
        
        if torso_angle < -0.05:
            body_language = "leaning forward (interested)"
            behaviors.append("leaning_forward")
        
        # --- Behavior: Standing still ---
        if not behaviors:
            behaviors.append("standing_still")
            body_language = "neutral"
        
        return behaviors, body_language
    
    def detect_movement(self, current_landmarks, h, w):
        """
        Detect approaching/leaving movement by comparing to previous frame.
        
        Returns:
            movement: 'approaching', 'leaving', 'stationary', or None.
        """
        if not current_landmarks or not self.prev_landmarks or self.mp_pose is None:
            self.prev_landmarks = current_landmarks
            return None
        
        try:
            # Indices for Tasks API
            NOSE = 0
            LEFT_SHOULDER = 11
            RIGHT_SHOULDER = 12
            
            curr_nose = current_landmarks[NOSE]
            prev_nose = self.prev_landmarks[NOSE]  # type: ignore
            
            # Shoulder width as proxy for distance (bigger = closer)
            curr_shoulder_w = abs(
                current_landmarks[LEFT_SHOULDER].x -
                current_landmarks[RIGHT_SHOULDER].x
            )
            prev_shoulder_w = abs(
                self.prev_landmarks[LEFT_SHOULDER].x -  # type: ignore
                self.prev_landmarks[RIGHT_SHOULDER].x  # type: ignore
            )
            
            size_change = curr_shoulder_w - prev_shoulder_w
            
            self.prev_landmarks = current_landmarks
            
            if size_change > 0.01:
                return "approaching"
            elif size_change < -0.01:
                return "leaving"
            else:
                return "stationary"
        except Exception:
            self.prev_landmarks = current_landmarks
            return None
    
    def detect_social_interaction(self, all_poses, h, w):
        """
        Detect if multiple people are interacting.
        
        Args:
            all_poses: List of pose landmark sets.
            
        Returns:
            interactions: List of interaction descriptions.
        """
        interactions = []  # type: list
        
        if len(all_poses) < 2:
            return interactions
        
        # Check if people are facing each other (nose positions converging)
        if len(all_poses) >= 2 and self.mp_pose:
            for i in range(len(all_poses)):
                for j in range(i + 1, len(all_poses)):
                    try:
                        nose_i = all_poses[i].landmark[self.mp_pose.PoseLandmark.NOSE]  # type: ignore
                        nose_j = all_poses[j].landmark[self.mp_pose.PoseLandmark.NOSE]  # type: ignore
                        
                        dist = ((nose_i.x - nose_j.x) ** 2 + (nose_i.y - nose_j.y) ** 2) ** 0.5
                        
                        if dist < 0.3:
                            interactions.append({
                                "type": "conversation",
                                "participants": 2,
                                "description": "Two people are talking nearby."
                            })
                    except Exception:
                        pass
        
        if len(all_poses) >= 3:
            interactions.append({
                "type": "group_conversation",
                "participants": len(all_poses),
                "description": f"Group of {len(all_poses)} people nearby."
            })
        
        return interactions
    
    def process_frame(self, frame):
        """
        Full behavior analysis pipeline for a single frame.
        
        Args:
            frame: BGR OpenCV image.
            
        Returns:
            annotated_frame: Frame with pose overlays and behavior labels.
            detections: List of structured behavior detection dicts.
        """
        h, w = frame.shape[:2]
        annotated = frame.copy()
        detections = []  # type: list
        
        if not self.detector:
            return annotated, detections
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        from mediapipe.python.solutions.drawing_utils import DrawingSpec
        from mediapipe import Image, ImageFormat
        
        # Wrapping frame in MediaPipe Image if using Tasks API
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        results = self.detector.detect(mp_image)
        
        if not results.pose_landmarks:
            return annotated, detections
        
        landmarks = results.pose_landmarks[0] # Take first pose detected
        
        # Draw pose skeleton
        if self.mp_draw and self.mp_pose:
            # Need to convert NormalizedLandmarks to LandmarkProto for drawing if using legacy draw
            # For simplicity in bypass: we just use the analysis data
            pass
        
        # Analyze behaviors
        behaviors, body_language = self.analyze_pose(landmarks, h, w)
        movement = self.detect_movement(landmarks, h, w)
        
        if movement and movement != "stationary":
            behaviors.append(movement)
        
        # Get depth-based distance
        distance = None
        if self.depth_engine:
            depth_map = self.depth_engine.estimate_depth(frame)  # type: ignore
            # Use nose position for distance
            NOSE = 0
            nose = landmarks[NOSE]
            nx, ny = int(nose.x * w), int(nose.y * h)
            bbox = [max(0, nx - 30), max(0, ny - 30), min(w, nx + 30), min(h, ny + 30)]
            distance = self.depth_engine.get_object_distance(depth_map, bbox)  # type: ignore
        
        detection = {
            "behaviors": behaviors,
            "body_language": body_language,
            "movement": movement or "stationary",
            "timestamp": round(float(time.time()), 3)  # type: ignore
        }
        if distance is not None:
            detection["distance"] = distance
        
        detections.append(detection)
        
        # Draw behavior labels on frame
        y_offset = 30
        for behavior in behaviors:
            label = f"Behavior: {behavior}"
            if distance is not None:
                label += f" — {distance}m"
            cv2.putText(annotated, label, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
        
        # Body language label
        cv2.putText(annotated, f"Body: {body_language}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        
        # Console output
        for b in behaviors:
            dist_str = f" — {distance}m" if distance is not None else ""
            print(f"  Behavior: {b}{dist_str}")
        print(f"  Body language: {body_language}")
        
        return annotated, detections


def run_camera(engine):
    """Real-time behavior analysis on webcam."""
    print("🎥 Starting Behavior Analysis Camera Mode (Press 'q' to exit)...")
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated, detections = engine.process_frame(frame)
        
        cv2.imshow("Netra AI — Behavior Analysis", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def run_video(engine, input_path):
    """Process a video file through behavior analysis."""
    print(f"🎬 Processing Video: {input_path}")
    cap = cv2.VideoCapture(input_path)
    
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    name, _ = os.path.splitext(os.path.basename(input_path))
    output_path = f"output_{name}_behavior.mp4"
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
    
    log_path = f"output_{name}_behavior_log.json"
    with open(log_path, 'w') as f:
        json.dump(all_detections, f, indent=2)
    
    print(f"✅ Behavior video saved to: {output_path}")
    print(f"📄 Detection log: {log_path}")


def run_image(engine, input_path):
    """Process a single image through behavior analysis."""
    print(f"🖼️ Processing Image: {input_path}")
    frame = cv2.imread(input_path)
    if frame is None:
        print("❌ Could not load image.")
        return
    
    annotated, detections = engine.process_frame(frame)
    
    name, ext = os.path.splitext(os.path.basename(input_path))
    output_path = f"output_{name}_behavior{ext}"
    cv2.imwrite(output_path, annotated)
    
    print(f"\n📊 Behavior Analysis Results:")
    print(json.dumps(detections, indent=2))
    print(f"\n✅ Saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Netra AI — Behavior Analysis Engine (Layer 7)")
    parser.add_argument('--mode', type=str, choices=['camera', 'video', 'image'], default='camera')
    parser.add_argument('--input', type=str, default='', help="Path to input video or image")
    parser.add_argument('--no_depth', action='store_true', help="Disable depth integration")
    args = parser.parse_args()
    
    engine = BehaviorEngine(use_depth=not args.no_depth)
    
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
