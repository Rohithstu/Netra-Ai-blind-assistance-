"""
Netra AI — Layer 3: Depth Estimation Engine
Uses MiDaS (Monocular Depth Estimation) to generate depth maps from single camera frames.
Provides distance estimation, spatial zone detection, and collision risk analysis.
"""
import cv2  # type: ignore
import torch  # type: ignore
import numpy as np  # type: ignore
import time
import argparse
import os


class DepthEngine:
    """MiDaS-based monocular depth estimation engine for Netra AI."""
    
    # Spatial zones for navigation guidance
    ZONES = {
        "left": (0.0, 0.33),
        "center": (0.33, 0.66),
        "right": (0.66, 1.0)
    }
    
    # Risk thresholds (in normalized depth; lower = closer)
    COLLISION_THRESHOLD = 0.25   # Very close → HIGH risk
    MEDIUM_RISK_THRESHOLD = 0.45  # Moderately close → MEDIUM risk

    def __init__(self, model_type="MiDaS_small"):
        """
        Initialize the MiDaS depth estimation model.
        
        Args:
            model_type: One of 'DPT_Large', 'DPT_Hybrid', 'MiDaS_small'.
                        'MiDaS_small' is fastest and recommended for real-time.
        """
        print(f"🧠 Loading MiDaS depth model ({model_type})...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Device: {self.device}")
        
        # Load MiDaS model from local cache to avoid Windows handle issues (WinError 6)
        hub_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub")
        midas_repo_dir = os.path.join(hub_dir, "intel-isl_MiDaS_master")
        
        if os.path.exists(midas_repo_dir):
            print(f"📦 Loading MiDaS from local cache: {midas_repo_dir}")
            self.model = torch.hub.load(midas_repo_dir, model_type, source='local')
            midas_transforms = torch.hub.load(midas_repo_dir, "transforms", source='local')
        else:
            print("⚠️ Local cache not found. Falling back to hub load (may fail on Windows)...")
            self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            
        self.model.to(self.device)
        self.model.eval()
        
        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
        
        print("✅ Depth model loaded successfully.")
    
    def estimate_depth(self, frame):
        """
        Generate a depth map from a single RGB frame.
        
        Args:
            frame: BGR OpenCV image (numpy array).
            
        Returns:
            depth_map: Normalized depth map (0.0 = far, 1.0 = close).
        """
        # Convert BGR to RGB for MiDaS
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply MiDaS transforms
        input_batch = self.transform(rgb_frame).to(self.device)
        
        # Run inference
        with torch.no_grad():
            prediction = self.model(input_batch)
            
            # Resize to original frame size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Normalize to 0-1 range (1.0 = closest, 0.0 = farthest)
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 0:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        
        return depth_map
    
    def get_object_distance(self, depth_map, bbox):
        """
        Estimate distance for a specific bounding box region.
        
        Args:
            depth_map: Normalized depth map.
            bbox: [x1, y1, x2, y2] bounding box coordinates.
            
        Returns:
            estimated_distance: Approximate distance in meters.
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape
        
        # Clamp coordinates
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        
        # Extract depth region for the bounding box
        region = depth_map[y1:y2, x1:x2]
        
        if region.size == 0:
            return 10.0  # Default far distance
        
        # Use median depth of the region (robust to outliers)
        median_depth = float(np.median(region))
        
        # Convert normalized depth to approximate meters
        # MiDaS provides relative depth; we use a simple inverse mapping
        # Calibration: depth_value 1.0 (closest) ≈ 0.5m, 0.0 (farthest) ≈ 10m
        if median_depth > 0.01:
            estimated_distance = round(0.5 + (1.0 - median_depth) * 9.5, 1)  # type: ignore
        else:
            estimated_distance = 10.0
        
        return estimated_distance
    
    def get_zone(self, bbox, frame_width):
        """
        Determine which spatial zone an object is in (left, center, right).
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box.
            frame_width: Width of the frame in pixels.
            
        Returns:
            zone_name: 'left', 'center', or 'right'.
        """
        x1, _, x2, _ = bbox
        center_x = (x1 + x2) / 2.0
        normalized_x = center_x / frame_width
        
        for zone_name, (low, high) in self.ZONES.items():
            if low <= normalized_x < high:
                return zone_name
        return "center"
    
    def get_risk_level(self, distance):
        """
        Determine collision risk level based on distance.
        
        Args:
            distance: Estimated distance in meters.
            
        Returns:
            risk_level: 'high', 'medium', or 'low'.
        """
        if distance < 1.0:
            return "high"
        elif distance < 2.0:
            return "medium"
        else:
            return "low"
    
    def create_depth_overlay(self, frame, depth_map):
        """
        Create a colorized depth heatmap overlay for visualization.
        
        Args:
            frame: Original BGR frame.
            depth_map: Normalized depth map.
            
        Returns:
            overlay: Blended frame with depth heatmap.
        """
        # Convert depth to a color heatmap
        depth_colormap = cv2.applyColorMap(
            (depth_map * 255).astype(np.uint8), cv2.COLORMAP_MAGMA
        )
        
        # Blend original frame with depth overlay
        overlay = cv2.addWeighted(frame, 0.6, depth_colormap, 0.4, 0)
        return overlay
    
    def process_frame(self, frame):
        """
        Full depth processing pipeline for a single frame.
        
        Args:
            frame: BGR OpenCV image.
            
        Returns:
            depth_map: Normalized depth map.
            overlay: Visualization with depth heatmap.
        """
        depth_map = self.estimate_depth(frame)
        overlay = self.create_depth_overlay(frame, depth_map)
        return depth_map, overlay


def run_camera_depth(engine):
    """Run depth estimation on live webcam feed."""
    print("🎥 Starting Real-Time Depth Estimation (Press 'q' to exit)...")
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        depth_map, overlay = engine.process_frame(frame)
        
        cv2.imshow("Netra AI — Depth View", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def run_image_depth(engine, image_path):
    """Run depth estimation on a single image."""
    print(f"🖼️ Processing depth for: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print("❌ Could not load image.")
        return
    
    depth_map, overlay = engine.process_frame(frame)
    
    name, ext = os.path.splitext(os.path.basename(image_path))
    output_path = f"output_{name}_depth{ext}"
    cv2.imwrite(output_path, overlay)
    print(f"✅ Depth overlay saved to: {output_path}")


def run_video_depth(engine, video_path):
    """Run depth estimation on a video file."""
    print(f"🎬 Processing depth for video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    name, _ = os.path.splitext(os.path.basename(video_path))
    output_path = f"output_{name}_depth.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        _, overlay = engine.process_frame(frame)
        out.write(overlay)
        frame_count += 1
    
    cap.release()
    out.release()
    print(f"✅ Depth video saved to: {output_path} ({frame_count} frames)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Netra AI — Depth Estimation Engine (Layer 3)")
    parser.add_argument('--mode', type=str, choices=['camera', 'video', 'image'], default='camera')
    parser.add_argument('--input', type=str, default='', help="Path to input video or image")
    parser.add_argument('--model_type', type=str, default='MiDaS_small',
                        choices=['DPT_Large', 'DPT_Hybrid', 'MiDaS_small'],
                        help="MiDaS model variant (default: MiDaS_small for speed)")
    args = parser.parse_args()
    
    engine = DepthEngine(model_type=args.model_type)
    
    if args.mode == 'camera':
        run_camera_depth(engine)
    elif args.mode == 'video':
        if not args.input:
            print("❌ --input is required for video mode.")
        else:
            run_video_depth(engine, args.input)
    elif args.mode == 'image':
        if not args.input:
            print("❌ --input is required for image mode.")
        else:
            run_image_depth(engine, args.input)
