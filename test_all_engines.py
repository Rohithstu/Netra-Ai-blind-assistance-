import sys
import os
import traceback

# Setup paths
base_dir = os.getcwd()
backend_dir = os.path.join(base_dir, "backend")
core_dir = os.path.join(backend_dir, "core")
sys.path.append(backend_dir)
sys.path.append(core_dir)

# Patch torch.hub
import torch.hub
def patched_get_git_branch(repo_dir): return 'master'
torch.hub._get_git_branch = patched_get_git_branch

try:
    print("1. Initializing FusionEngine...")
    from fusion_engine import FusionEngine
    engine = FusionEngine(yolo_model_path=os.path.join(backend_dir, "yolov8s.pt"))
    print("✅ FusionEngine Ready.")

    print("\n2. Initializing EmotionEngine...")
    from emotion_engine import EmotionEngine
    emotion_engine = EmotionEngine(use_depth=False) # Disable depth for isolation
    print("✅ EmotionEngine Ready.")

    print("\n3. Initializing BehaviorEngine...")
    from behavior_engine import BehaviorEngine
    behavior_engine = BehaviorEngine(use_depth=False) # Disable depth for isolation
    print("✅ BehaviorEngine Ready.")

    print("\n🚀 ALL ENGINES INITIALIZED SUCCESSFULLY.")
except Exception as e:
    print(f"\n❌ INITIALIZATION FAILED: {e}")
    traceback.print_exc()
