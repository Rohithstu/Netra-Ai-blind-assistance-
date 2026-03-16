import sys
import os
import cv2
import numpy as np
import base64
import json
import time
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Add the core directory to path so we can import fusion_engine
sys.path.append(os.path.join(os.path.dirname(__file__), "core"))

# --- CRITICAL PATCH FOR WINDOWS WinError 6 ---
try:
    import torch.hub
    def patched_get_git_branch(repo_dir): return 'master'
    def patched_generate_repo_dir(model_dir, repo_owner, repo_name, ref):
        return os.path.join(model_dir, f"{repo_owner}_{repo_name}_{ref}")
    
    # Force torch.hub to bypass git/subprocess if it fails
    torch.hub._get_git_branch = patched_get_git_branch
    # If the handle is invalid, it's usually because it tries to call git.
    # We can try to prevent git calls by making it think it's already there or not needed.
except Exception as e:
    print(f"⚠️ Failed to apply torch.hub patch: {e}")
# ---------------------------------------------

from fusion_engine import FusionEngine
from emotion_engine import EmotionEngine
from behavior_engine import BehaviorEngine

app = FastAPI(title="Netra AI Core Server")

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🚀 Initializing Netra Fusion Engine (YOLOv8 + MiDaS)...")
try:
    engine = FusionEngine(yolo_model_path=os.path.join(os.path.dirname(__file__), "yolov8s.pt"))
except Exception as e:
    print(f"❌ Failed to initialize Fusion Engine: {e}")
    engine = None

print("🧠 Initializing Social & Emotional Intelligence Engines...")
try:
    emotion_engine = EmotionEngine(use_depth=True)
except Exception as e:
    print(f"⚠️ Failed to initialize Emotion Engine: {e}")
    emotion_engine = None

try:
    behavior_engine = BehaviorEngine(use_depth=True)
except Exception as e:
    print(f"⚠️ Failed to initialize Behavior Engine: {e}")
    behavior_engine = None

@app.get("/")
async def health_check():
    return {"status": "online", "engine": "YOLOv8 + MiDaS"}

@app.websocket("/ws/vision")
async def vision_socket(websocket: WebSocket):
    await websocket.accept()
    print("🔌 Browser connected to Netra Core via WebSocket")
    
    try:
        while True:
            # Receive base64 frame from browser
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if "image" not in message:
                continue
                
            # Decode base64 image
            img_data = base64.b64decode(message["image"].split(",")[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
            
            # 1. Process via Fusion Engine
            if engine:
                annotated, detections = engine.process_frame(frame, conf_threshold=0.4)
            else:
                detections = []
            
            # 2. Process via Emotion Engine
            emotional_results = []
            if emotion_engine:
                _, emotional_results = emotion_engine.process_frame(frame)
            
            # 3. Process via Behavior Engine
            behavioral_results = []
            if behavior_engine:
                _, behavioral_results = behavior_engine.process_frame(frame)
            
            # 4. Transform detections to the format expected by PerceptionEngine
            processed_results = []
            for d in detections:
                x1, y1, x2, y2 = d["bounding_box"]
                width = x2 - x1
                height = y2 - y1
                
                processed_results.append({
                    "class": d["object"],
                    "score": d["confidence"],
                    "bbox": [x1, y1, width, height],
                    "distance": d["distance"],
                    "risk": d["risk_level"], 
                    "depthZone": d["direction"] 
                })

            # 5. Response
            response = {
                "detections": processed_results,
                "emotions": emotional_results,
                "behaviors": behavioral_results,
                "fps": 0, 
                "server_time": time.time()
            }
            
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        print("🔌 Browser disconnected from Netra Core")
    except Exception as e:
        print(f"❌ Server Error: {e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
