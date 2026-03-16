import torch.hub
import os

# Monkey-patch git branch check
def patched_get_git_branch(repo_dir): return 'master'
torch.hub._get_git_branch = patched_get_git_branch

hub_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub")
midas_repo_dir = os.path.join(hub_dir, "intel-isl_MiDaS_master")

try:
    print(f"Testing local load from: {midas_repo_dir}")
    model = torch.hub.load(midas_repo_dir, 'MiDaS_small', source='local')
    print("Success loading model!")
    transforms = torch.hub.load(midas_repo_dir, 'transforms', source='local')
    print("Success loading transforms!")
except Exception as e:
    import traceback
    print(f"FAILED: {e}")
    traceback.print_exc()
