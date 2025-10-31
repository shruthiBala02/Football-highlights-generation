"""
k4_highlight_manager.py (K-Pipeline)
------------------------------------
Event handler for highlight (snippet) creation + latency accounting.

- Filename format: Highlight_{score1}_{score2}.mp4  (from event_row)
- Default clip window: [exact_ts - 60s, exact_ts + 10s]
- Uses FFmpeg fast-copy if available, else OpenCV re-encode fallback
- Returns: highlight_latency_s, total_latency_s, snippet_path
"""

import os
import shutil
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

import cv2

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# --------------------------- Utilities ---------------------------

def _which_ffmpeg() -> Optional[str]:
    for name in ("ffmpeg", "ffmpeg.exe"):
        p = shutil.which(name)
        if p:
            return p
    return None

def _safe_name(name: str) -> str:
    keep = "-_.()[]{} "
    return "".join(ch if ch.isalnum() or ch in keep else "_" for ch in name).strip(" ._")

def _clip_with_ffmpeg(ffmpeg_bin: str, src: str, start_s: float, end_s: float, dst: str) -> None:
    # Fast path: copy streams, accurate enough for scoreboard highlights.
    # -ss before -i is key for fast seek; we also include -to for end time.
    duration = max(0.0, end_s - start_s)
    if duration <= 0.0:
        raise ValueError("Invalid duration for clipping.")
    cmd = [
        ffmpeg_bin,
        "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_s:.3f}",
        "-i", src,
        "-t", f"{duration:.3f}",
        "-c", "copy",
        "-map", "0",
        dst,
    ]
    subprocess.run(cmd, check=True)

def _clip_with_opencv(src: str, start_s: float, end_s: float, dst: str) -> None:
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    start_idx = max(0, int(round(start_s * fps)))
    end_idx = min(total - 1, int(round(end_s * fps)))
    if end_idx <= start_idx:
        cap.release()
        raise ValueError("Invalid frame range for clipping.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(dst, fourcc, fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    idx = start_idx
    while idx <= end_idx:
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)
        idx += 1

    out.release()
    cap.release()

# ------------------------- Event Handler -------------------------

class HighlightManager:
    """
    Threaded highlight maker. Submits a clipping job and waits for the result
    (blocking until completion) so we can return highlight_latency + total_latency
    to the caller (K3). Events are rare, so brief blocking is acceptable.
    """

    def __init__(
        self,
        snippets_dir: str = os.path.join("k_OUTPUTS", "k_highlights"),
        pre_roll_s: float = 60.0,
        post_roll_s: float = 10.0,
        prefer_ffmpeg: bool = True,
        max_workers: int = 2,
    ):
        self.snippets_dir = snippets_dir
        self.pre_roll_s = float(pre_roll_s)
        self.post_roll_s = float(post_roll_s)
        self.ffmpeg_bin = _which_ffmpeg() if prefer_ffmpeg else None
        Path(self.snippets_dir).mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="K4Clip")

    def _clip(self, video_path: str, start_s: float, end_s: float, out_path: str) -> None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if self.ffmpeg_bin:
            _clip_with_ffmpeg(self.ffmpeg_bin, video_path, start_s, end_s, out_path)
        else:
            _clip_with_opencv(video_path, start_s, end_s, out_path)

    def __call__(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expects:
          event = {
            "video_path": str,
            "event_row": { "exact_ts": float, "detect_latency_s": float, "score1": int/str, "score2": int/str, ... },
            "suggested_name": "Highlight_{S1}_{S2}.mp4"   # optional
          }
        Returns:
          {
            "highlight_latency_s": float,
            "total_latency_s": float,
            "snippet_path": str
          }
        """
        video_path = event["video_path"]
        row = event["event_row"]
        exact_ts = float(row["exact_ts"])
        detect_latency_s = float(row.get("detect_latency_s", 0.0))

        # Build filename using the requested format Highlight_{score1}_{score2}.mp4
        s1 = str(row.get("score1", ""))
        s2 = str(row.get("score2", ""))
        base_name = event.get("suggested_name", f"Highlight_{s1}_{s2}.mp4")
        base_name = _safe_name(base_name) or "Highlight.mp4"

        out_path = os.path.join(self.snippets_dir, base_name)

        # Compute clip window [exact_ts - pre_roll, exact_ts + post_roll]
        start_s = max(0.0, exact_ts - self.pre_roll_s)
        end_s = max(start_s, exact_ts + self.post_roll_s)

        t0 = time.perf_counter()
        fut = self.pool.submit(self._clip, video_path, start_s, end_s, out_path)
        # Block until finished so we can return concrete latencies to CSV
        exc: Optional[Exception] = None
        try:
            fut.result()
        except Exception as e:
            exc = e
        t1 = time.perf_counter()

        if exc:
            log.warning(f"[K4] Highlight creation failed: {exc}")
            return {
                "highlight_latency_s": 0.0,
                "total_latency_s": detect_latency_s,  # no added time
                "snippet_path": "",
            }

        highlight_latency = t1 - t0
        total_latency = detect_latency_s + highlight_latency

        return {
            "highlight_latency_s": round(highlight_latency, 3),
            "total_latency_s": round(total_latency, 3),
            "snippet_path": out_path,
        }

# --------------------- Convenience Builder ----------------------

def build_event_handler(
    snippets_dir: str = os.path.join("k_OUTPUTS", "k_highlights"),
    pre_roll_s: float = 60.0,
    post_roll_s: float = 10.0,
    prefer_ffmpeg: bool = True,
    max_workers: int = 2,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Factory to create a callable event handler for K3.
    """
    return HighlightManager(
        snippets_dir=snippets_dir,
        pre_roll_s=pre_roll_s,
        post_roll_s=post_roll_s,
        prefer_ffmpeg=prefer_ffmpeg,
        max_workers=max_workers,
    )

# --------------------------- Demo ---------------------------

if __name__ == "__main__":
    # Minimal demo simulating a single event call
    demo_video = os.path.join("Inputs", "Original_video.mp4")
    handler = build_event_handler(snippets_dir=os.path.join("k_OUTPUTS", "k_highlights_demo"))

    fake_row = {
        "exact_ts": 120.0,           # pretend the score first appears at t=120s
        "detect_latency_s": 1.25,    # pretend detection took 1.25s (live-like)
        "score1": 1,
        "score2": 0,
    }
    event = {
        "video_path": demo_video,
        "event_row": fake_row,
        "suggested_name": "Highlight_1_0.mp4"
    }
    res = handler(event)
    print("[K4] Demo result:", res)
