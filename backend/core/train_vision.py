import os
from ultralytics import YOLO

def train():
    print("🚀 Initializing Netra Vision AI Training Pipeline...")

    # Define paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_yaml_path = os.path.join(base_dir, 'datasets', 'data.yaml')

    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"Configuration file {data_yaml_path} not found. Please ensure Layer 1 dataset preparation is complete.")

    # Load a pretrained YOLOv8 model for transfer learning
    # Using 'yolov8s.pt' (Small) as a good balance between speed (for blind nav) and accuracy
    model = YOLO('yolov8s.pt') 

    print("✅ Model loaded. Starting training process...")
    print(f"📁 Dataset configuration: {data_yaml_path}")
    
    # Train the model with exact hyperparameters specified in Layer 2 requirements
    results = model.train(
        data=data_yaml_path,
        epochs=100,            # High epoch count for accurate convergence
        imgsz=640,            # Standard resolution
        batch=16,            # Fits well on most 8GB+ GPUs
        optimizer='AdamW',     # Adam Weight Decay optimizer
        lr0=0.001,           # Initial learning rate
        
        # Augmentation Settings explicitly requested
        scale=0.5,           # Random scaling
        fliplr=0.5,          # Random horizontal flipping
        hsv_h=0.015,         # Color jitter (Hue)
        hsv_s=0.7,           # Color jitter (Saturation)
        hsv_v=0.4,           # Color jitter (Value)
        
        # Hardware optimization
        device=0,            # Try CUDA GPU 0. Use 'cpu' if no NVIDIA GPU exists.
        workers=4,           # Dataloader threads
        project='runs/netra_vision', # Save location
        name='train_layer2'
    )

    print("\n🎯 Training Complete!")
    print("The best weights are stored at: runs/netra_vision/train_layer2/weights/best.pt")
    print("To export these weights for real-time inference, run: python scripts/export_model.py")

if __name__ == '__main__':
    train()
