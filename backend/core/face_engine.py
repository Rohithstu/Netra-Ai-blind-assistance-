"""
Netra AI — Layer 6: Face Recognition Engine
Uses MTCNN for face detection and FaceNet (InceptionResnetV1) for face recognition.
Integrates with the face memory database and depth estimation for distance-aware identity output.
"""
import cv2  # type: ignore
import torch  # type: ignore
import numpy as np  # type: ignore
import time
import json
import argparse
import os
import sys

from facenet_pytorch import MTCNN, InceptionResnetV1  # type: ignore
from face_database import FaceDatabase

# Optional: import depth engine for distance integration
try:
    from depth_engine import DepthEngine  # type: ignore
    DEPTH_AVAILABLE = True
except ImportError:
    DEPTH_AVAILABLE = False


class FaceEngine:
    """MTCNN + FaceNet face detection and recognition engine for Netra AI."""
    
    RECOGNITION_THRESHOLD = 0.7  # Cosine similarity threshold for positive match
    CROWD_THRESHOLD = 5          # Number of faces to trigger "crowd" alert
    
    def __init__(self, use_depth=True):
        """
        Initialize face detection and recognition models.
        
        Args:
            use_depth: Whether to integrate depth estimation for distance.
        """
        print("=" * 60)
        print("  NETRA AI — Face Recognition Engine (Layer 6)")
        print("=" * 60)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n   Device: {self.device}")
        
        # Face detection model (MTCNN)
        print("📦 Loading MTCNN face detector...")
        self.detector = MTCNN(
            keep_all=True,
            device=self.device,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7]
        )
        
        # Face embedding model (FaceNet / InceptionResnetV1 pretrained on VGGFace2)
        print("📦 Loading FaceNet embedding model (VGGFace2)...")
        self.recognizer = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Face memory database
        self.db = FaceDatabase()
        self.known_embeddings = []  # type: list
        self.known_names = []       # type: list
        self.known_ids = []         # type: list
        self._load_known_faces()
        
        # Depth engine (optional)
        self.depth_engine = None
        if use_depth and DEPTH_AVAILABLE:
            print("\n📦 Loading depth engine for distance integration...")
            self.depth_engine = DepthEngine(model_type="MiDaS_small")
        
        print(f"\n✅ Face Engine ready. Known people: {len(self.known_names)}\n")
    
    def _load_known_faces(self):
        """Load all known face embeddings from the database."""
        people = self.db.get_all_people()
        self.known_embeddings = []
        self.known_names = []
        self.known_ids = []
        
        for person in people:
            self.known_embeddings.append(
                torch.tensor(person["embedding"]).to(self.device)
            )
            self.known_names.append(person["name"])
            self.known_ids.append(person["person_id"])
        
        print(f"   Loaded {len(self.known_names)} known face(s) from database.")
    
    def detect_faces(self, frame):
        """
        Detect all faces in a frame using MTCNN.
        
        Args:
            frame: BGR OpenCV image.
            
        Returns:
            boxes: List of [x1, y1, x2, y2] bounding boxes.
            confidences: List of detection confidence scores.
            face_tensors: Cropped and aligned face tensors for embedding extraction.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        boxes, confidences = self.detector.detect(rgb_frame)
        
        if boxes is None:
            return [], [], []
        
        # Get aligned and cropped face tensors
        face_tensors = self.detector(rgb_frame)
        
        valid_boxes = []
        valid_confs = []
        valid_faces = []
        
        for i, (box, conf) in enumerate(zip(boxes, confidences)):  # type: ignore
            if conf is not None and conf > 0.5 and face_tensors is not None and i < len(face_tensors):
                valid_boxes.append([int(b) for b in box])
                valid_confs.append(float(conf))
                valid_faces.append(face_tensors[i])
        
        return valid_boxes, valid_confs, valid_faces
    
    def get_embedding(self, face_tensor):
        """
        Extract the 512-dimensional face embedding vector.
        
        Args:
            face_tensor: Aligned face tensor from MTCNN.
            
        Returns:
            embedding: Normalized embedding vector.
        """
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.recognizer(face_tensor)
        return embedding.squeeze()
    
    def recognize_face(self, embedding):
        """
        Compare a face embedding against all known faces.
        
        Args:
            embedding: Face embedding vector.
            
        Returns:
            name: Recognized person's name or "Unknown".
            confidence: Similarity score.
            person_id: Database ID or None.
        """
        if len(self.known_embeddings) == 0:
            return "Unknown", 0.0, None
        
        best_score = -1.0
        best_name = "Unknown"
        best_id = None
        
        for i, known_emb in enumerate(self.known_embeddings):
            # Cosine similarity
            similarity = float(torch.nn.functional.cosine_similarity(
                embedding.unsqueeze(0), known_emb.unsqueeze(0)
            ))
            
            if similarity > best_score:
                best_score = similarity
                best_name = self.known_names[i]
                best_id = self.known_ids[i]
        
        if best_score >= self.RECOGNITION_THRESHOLD:
            # Update last seen in database
            self.db.update_last_seen(best_id)
            return best_name, best_score, best_id
        else:
            return "Unknown", best_score, None
    
    def remember_person(self, frame, name, notes=""):
        """
        Capture a face from the current frame and store it in the database.
        
        Args:
            frame: BGR OpenCV image with a face visible.
            name: Name to assign to the person.
            notes: Optional notes.
            
        Returns:
            success: Whether the face was captured and stored.
        """
        boxes, confs, faces = self.detect_faces(frame)
        
        if len(faces) == 0:
            print("❌ No face detected in frame. Please try again.")
            return False
        
        # Use the highest confidence face
        best_idx = confs.index(max(confs))
        embedding = self.get_embedding(faces[best_idx])
        embedding_list = embedding.cpu().tolist()
        
        person_id = self.db.add_person(name, embedding_list, notes)
        
        # Reload known faces
        self._load_known_faces()
        
        print(f"✅ Remembered: {name}")
        return True
    
    def get_zone(self, bbox, frame_width):
        """Determine spatial zone (left/center/right) for a face."""
        x1, _, x2, _ = bbox
        center_x = (x1 + x2) / 2.0
        ratio = center_x / frame_width
        
        if ratio < 0.33:
            return "left"
        elif ratio < 0.66:
            return "center"
        else:
            return "right"
    
    def process_frame(self, frame):
        """
        Full face recognition pipeline for a single frame.
        
        Args:
            frame: BGR OpenCV image.
            
        Returns:
            annotated_frame: Frame with face boxes and identity labels.
            detections: List of structured face detection dictionaries.
        """
        h, w = frame.shape[:2]
        annotated = frame.copy()
        
        # Get depth map if available
        depth_map = None
        if self.depth_engine:
            depth_map = self.depth_engine.estimate_depth(frame)  # type: ignore
        
        # Detect faces
        boxes, confs, faces = self.detect_faces(frame)
        
        detections = []
        
        # Crowd detection
        if len(boxes) > self.CROWD_THRESHOLD:
            print(f"  👥 Crowd detected: {len(boxes)} people")
            cv2.putText(annotated, f"CROWD: {len(boxes)} people", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        for i, (box, conf, face_tensor) in enumerate(zip(boxes, confs, faces)):  # type: ignore
            x1, y1, x2, y2 = box
            
            # Get face embedding and recognize
            embedding = self.get_embedding(face_tensor)
            name, similarity, person_id = self.recognize_face(embedding)
            
            # Get spatial zone
            zone = self.get_zone(box, w)
            
            # Get distance if depth is available
            distance = None
            if depth_map is not None:
                distance = self.depth_engine.get_object_distance(depth_map, box)  # type: ignore
            
            # Build detection data
            detection = {
                "person": name,
                "confidence": round(float(similarity), 2),  # type: ignore
                "direction": zone,
                "bounding_box": [x1, y1, x2, y2],
                "timestamp": round(float(time.time()), 3)  # type: ignore
            }
            if distance is not None:
                detection["distance"] = distance
            
            detections.append(detection)
            
            # Draw on frame
            if name != "Unknown":
                color = (0, 255, 0)  # Green for known
                label = f"{name} ({similarity:.2f})"
            else:
                color = (0, 0, 255)  # Red for unknown
                label = f"Unknown ({conf:.2f})"
            
            if distance is not None:
                label += f" — {distance}m"
            
            label += f" [{zone}]"
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            
            # Console output
            dist_str = f" — {distance}m" if distance is not None else ""
            print(f"  {name}{dist_str} → {zone}")
        
        return annotated, detections


def run_camera(engine):
    """Real-time face recognition on webcam."""
    print("🎥 Starting Face Recognition Camera Mode (Press 'q' to exit, 'r' to remember a face)...")
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated, detections = engine.process_frame(frame)
        
        cv2.imshow("Netra AI — Face Recognition", annotated)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Remember the face currently visible
            name = input("\n📝 Enter person's name: ").strip()
            if name:
                engine.remember_person(frame, name)
    
    cap.release()
    cv2.destroyAllWindows()


def run_video(engine, input_path):
    """Process a video file through face recognition."""
    print(f"🎬 Processing Video: {input_path}")
    cap = cv2.VideoCapture(input_path)
    
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    name, _ = os.path.splitext(os.path.basename(input_path))
    output_path = f"output_{name}_faces.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))
    
    all_detections = []
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
    
    log_path = f"output_{name}_faces_log.json"
    with open(log_path, 'w') as f:
        json.dump(all_detections, f, indent=2)
    
    print(f"✅ Face video saved to: {output_path}")
    print(f"📄 Detection log: {log_path} ({len(all_detections)} detections)")


def run_image(engine, input_path):
    """Process a single image through face recognition."""
    print(f"🖼️ Processing Image: {input_path}")
    frame = cv2.imread(input_path)
    if frame is None:
        print("❌ Could not load image.")
        return
    
    annotated, detections = engine.process_frame(frame)
    
    name, ext = os.path.splitext(os.path.basename(input_path))
    output_path = f"output_{name}_faces{ext}"
    cv2.imwrite(output_path, annotated)
    
    print(f"\n📊 Face Recognition Results:")
    print(json.dumps(detections, indent=2))
    print(f"\n✅ Saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Netra AI — Face Recognition Engine (Layer 6)")
    parser.add_argument('--mode', type=str, choices=['camera', 'video', 'image'], default='camera')
    parser.add_argument('--input', type=str, default='', help="Path to input video or image")
    parser.add_argument('--no_depth', action='store_true', help="Disable depth integration")
    args = parser.parse_args()
    
    engine = FaceEngine(use_depth=not args.no_depth)
    
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
