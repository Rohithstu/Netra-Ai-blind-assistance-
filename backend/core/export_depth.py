"""
Netra AI — Layer 3: Depth Model Export
Exports the MiDaS depth model to TorchScript format for optimized deployment.
"""
import torch  # type: ignore
import os


def export_depth_model(model_type="MiDaS_small"):
    print(f"🛠️ Netra AI — Depth Model Exporter ({model_type})")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    target_dir = os.path.join(base_dir, 'models', 'depth')
    os.makedirs(target_dir, exist_ok=True)
    
    # Load model from hub
    print("📦 Loading MiDaS model from PyTorch Hub...")
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.eval()
    
    # Export to TorchScript
    print("📦 Exporting to TorchScript...")
    if model_type == "MiDaS_small":
        dummy_input = torch.randn(1, 3, 256, 256)
    else:
        dummy_input = torch.randn(1, 3, 384, 384)
    
    traced_model = torch.jit.trace(model, dummy_input)
    
    output_path = os.path.join(target_dir, 'netra_depth_model.pt')
    traced_model.save(output_path)
    
    print(f"🎉 Depth model exported to: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")


if __name__ == '__main__':
    export_depth_model()
