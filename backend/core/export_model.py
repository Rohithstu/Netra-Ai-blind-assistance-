import os
import shutil
from ultralytics import YOLO

def export_model():
    print("🛠️ Netra AI Vision Model Exporter")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Path where YOLO saves the best weights by default in our train script
    best_weights_path = os.path.join(base_dir, 'runs', 'netra_vision', 'train_layer2', 'weights', 'best.pt')
    
    # Target folder to store final optimized models
    target_dir = os.path.join(base_dir, 'models', 'vision')
    os.makedirs(target_dir, exist_ok=True)
    
    if not os.path.exists(best_weights_path):
        print(f"❌ Error: Could not find trained weights at {best_weights_path}")
        print("Please run `python scripts/train_vision.py` completely before exporting.")
        return

    print("✅ Found trained weights. Loading into memory...")
    model = YOLO(best_weights_path)
    
    print("\n📦 Exporting to ONNX format (High-performance cross-platform inference)...")
    # This generates best.onnx next to best.pt
    onnx_path = model.export(format='onnx', dynamic=True, simplify=True)
    
    print("\n📦 Exporting to TorchScript format (Native PyTorch optimization)...")
    # This generates best.torchscript next to best.pt
    ts_path = model.export(format='torchscript')
    
    # Optional: TensorRT is skipped by default as it requires dedicated NVIDIA Linux/WSL environments
    # model.export(format='engine', device=0) 
    
    # Move models to the final directory
    print(f"\n🚚 Moving optimized models to `{target_dir}`...")
    
    final_pt_path = os.path.join(target_dir, 'netra_vision_model.pt')
    shutil.copy(best_weights_path, final_pt_path)
    
    if onnx_path and os.path.exists(onnx_path):
        final_onnx_path = os.path.join(target_dir, 'netra_vision_model.onnx')
        shutil.copy(onnx_path, final_onnx_path)
        
    if ts_path and os.path.exists(ts_path):
        final_ts_path = os.path.join(target_dir, 'netra_vision_model.torchscript')
        shutil.copy(ts_path, final_ts_path)
        
    print(f"🎉 Model export complete! The core PyTorch model is stored at:\n{final_pt_path}")

if __name__ == '__main__':
    export_model()
