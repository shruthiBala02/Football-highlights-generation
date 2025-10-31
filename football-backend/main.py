import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "FOOTBALLLLLL")))
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import shutil
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware import Middleware
from fastapi.middleware.gzip import GZipMiddleware
# --- Global variable to track the last uploaded video ---
LAST_UPLOADED_VIDEO_PATH = None


app = FastAPI(
    title="Football Backend",
    middleware=[
        Middleware(GZipMiddleware, minimum_size=1000),  # allows compression
    ],
)

# increase upload limit
from starlette.requests import Request
from starlette.responses import Response
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.types import ASGIApp, Receive, Scope, Send

class LimitUploadSizeMiddleware:
    def __init__(self, app: ASGIApp, max_upload_size: int = 2_000_000_000):
        self.app = app
        self.max_upload_size = max_upload_size

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http" and scope["method"] == "POST":
            received = 0
            async def limited_receive():
                nonlocal received
                message = await receive()
                if message["type"] == "http.request":
                    body = message.get("body", b"")
                    received += len(body)
                    if received > self.max_upload_size:
                        raise RuntimeError("File too large")
                return message
            scope["_body"] = b""
            await self.app(scope, limited_receive, send)
        else:
            await self.app(scope, receive, send)

app.add_middleware(LimitUploadSizeMiddleware, max_upload_size=2_000_000_000)

# ------------------------------------------------------
# Import FOOTBALLLLLL pipeline
# ------------------------------------------------------
sys.path.append(r"C:\Users\MYPC\Desktop\Football application\football-backend\FOOTBALLLLLL")
from FOOTBALLLLLL.k_run_pipeline import run_k_pipeline_live
  # this must yield async updates

# ------------------------------------------------------
# FastAPI app setup
# ------------------------------------------------------
app = FastAPI(title="Football Highlight Backend")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
UPLOAD_DIR = os.path.join(OUTPUT_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# ------------------------------------------------------
# Active WebSocket clients
# ------------------------------------------------------
clients = []

@app.websocket("/ws/goals")
async def goal_ws(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client connected")
    clients.append(websocket)

    try:
        # 🟡 Wait until upload is complete instead of returning early
        if LAST_UPLOADED_VIDEO_PATH is None:
            await websocket.send_json({
                "type": "status",
                "message": "No uploaded video found. Waiting for upload..."
            })
            while LAST_UPLOADED_VIDEO_PATH is None:
                await asyncio.sleep(1)

        # ✅ Once upload is ready, start pipeline
        await websocket.send_json({
            "type": "status",
            "message": f"Starting pipeline for {os.path.basename(LAST_UPLOADED_VIDEO_PATH)}"
        })

        # 🔁 Stream continuous updates from pipeline
        async for update in run_k_pipeline_live(LAST_UPLOADED_VIDEO_PATH):
            await websocket.send_json(update)

        # ✅ Signal completion cleanly
        await websocket.send_json({
            "type": "status",
            "message": "Pipeline complete."
        })

    except Exception as e:
        print(f"⚠️ Pipeline WebSocket error: {e}")
        await websocket.send_json({
            "type": "status",
            "message": f"Pipeline error: {str(e)}"
        })

    finally:
        print("❌ Client disconnected")
        if websocket in clients:
            clients.remove(websocket)


# ------------------------------------------------------
# Helper functions
# ------------------------------------------------------
async def broadcast_status(message: str):
    """Send status/progress updates to all connected clients."""
    payload = {"type": "status", "message": message}
    print("STATUS:", message)
    for ws in clients:
        try:
            await ws.send_json(payload)
        except Exception:
            pass


async def send_goal_update(goal_id: int, clip_path: str):
    """Send a new goal highlight to all connected clients."""
    # Ensure clip path always starts with 'outputs/'
    if not clip_path.startswith("outputs/"):
        clip_path = f"outputs/{clip_path.lstrip('/')}"
    
    # Build proper public URL served by FastAPI StaticFiles
    url = f"http://127.0.0.1:8000/{clip_path.replace('\\', '/')}"

    data = {
        "type": "goal",
        "goal_id": goal_id,
        "clip_name": os.path.basename(clip_path),
        "clip_rel_path": clip_path,
        "url": url,
    }

    print("🎯 GOAL EVENT SENT:", data)
    
    # Send to all connected WebSocket clients
    for ws in clients:
        try:
            await ws.send_json(data)
        except Exception as e:
            print("⚠️ WebSocket send failed:", e)



# ------------------------------------------------------
# POST /process_video — Upload + Run K-Pipeline
# ------------------------------------------------------
@app.post("/process_video")
async def process_video(file: UploadFile = File(...)):
    print("/process_video endpoint HIT!") 
    """Receives an uploaded football video and runs the highlight pipeline."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    video_path = os.path.join(UPLOAD_DIR, file.filename)
    global LAST_UPLOADED_VIDEO_PATH 

    # Save uploaded file
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    LAST_UPLOADED_VIDEO_PATH = video_path

    print(f"🎥 Uploaded video saved to: {video_path}")
    await broadcast_status("Video upload successful. Starting highlight extraction...")

    try:
        # Run pipeline generator
        async for update in run_k_pipeline_live(video_path):
            kind = update.get("type")

            if kind == "status":
                await broadcast_status(update["message"])

            elif kind == "frame":
                frame_no = update.get("frame_no", 0)
                await broadcast_status(f"Frame {frame_no} processed")

            elif kind == "goal":
                goal_id = update.get("goal_id", 0)
                clip_rel_path = update.get("clip_rel_path")

                # prevent duplicates — skip if this goal already sent
                if not hasattr(process_video, "_sent_goals"):
                    process_video._sent_goals = set()

                if clip_rel_path not in process_video._sent_goals:
                    process_video._sent_goals.add(clip_rel_path)
                    await send_goal_update(goal_id, clip_rel_path)



            elif kind == "done":
                elapsed = update.get("elapsed", 0)
                await broadcast_status(f"Processing complete in {elapsed:.1f} seconds")

        return {"status": "ok", "message": "Highlights extraction completed"}

    except Exception as e:
        err_msg = f"Pipeline failed: {e}"
        print(err_msg)
        await broadcast_status(err_msg)
        return {"status": "error", "message": str(e)}
from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="../React/football-highlights/build", html=True), name="frontend")
