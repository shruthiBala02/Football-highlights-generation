"""
k1_frames_extractor.py (K-Pipeline Final)
----------------------------------------
Thread-safe frame extractor with constant-rate sampling (default: 1 frame / 2 seconds).
Supports optional live emulation (wall-clock pacing), visible live banner,
and pinned-memory-ready torch tensor export.
"""

import os
import cv2
import time
import math
import logging
import threading
from typing import Optional, Generator, Tuple

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


class FrameStreamer:
    """
    Thread-safe VideoCapture wrapper providing:
      - get_frame_at(ts): grab nearest frame at timestamp (seconds)
      - frames(...): constant-rate sequential generator (default 1 frame / 2s)
      - to_tensor(frame, pin_memory=True): optional torch tensor export
    """

    def __init__(self, video_path: str):
        self.lock = threading.Lock()

        # Accept numeric camera index or file path
        if os.path.exists(video_path):
            self.cap = cv2.VideoCapture(video_path)
        else:
            try:
                self.cap = cv2.VideoCapture(int(video_path))
            except Exception:
                self.cap = cv2.VideoCapture(video_path)

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {video_path}")

        self.src_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration_s = (self.frame_count / self.src_fps) if self.src_fps > 0 else 0.0
        log.info(f"[K1] FrameStreamer ready | FPS={self.src_fps:.2f} | Frames={self.frame_count} | Duration={self.duration_s/60:.1f} min")

    def close(self):
        with self.lock:
            if getattr(self, "cap", None) is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _seek_frame_index(self, frame_index: int) -> bool:
        """Internal: seek to absolute frame index safely."""
        if frame_index < 0:
            frame_index = 0
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

    def get_frame_at(self, ts: float) -> Optional[np.ndarray]:
        """Return the frame closest to timestamp ts (in seconds), or None if EOF."""
        with self.lock:
            if self.cap is None:
                raise RuntimeError("Capture is closed")

            ts = max(0.0, float(ts))
            frame_no = int(round(ts * self.src_fps))
            if self.frame_count > 0 and frame_no >= self.frame_count:
                return None

            self._seek_frame_index(frame_no)
            ok, frame = self.cap.read()
            return frame if ok else None

    @staticmethod
    def _now() -> float:
        return time.perf_counter()

    def frames(
        self,
        start_ts: float = 0.0,
        seconds_per_frame: float = 2.0,
        end_ts: Optional[float] = None,
        emulate_live: bool = False,
        max_frames: Optional[int] = None,
        copy_frame: bool = True,
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Constant-rate sequential sampler.

        Yields: (frame_idx, timestamp_s, frame_bgr)

        Params
        ------
        seconds_per_frame : float
            Sampling period in seconds (default 2.0 => 1 frame per 2 seconds).
        emulate_live : bool
            If True, wall-clock pacing is enforced to mimic live feed.
        """
        if seconds_per_frame <= 0:
            raise ValueError("seconds_per_frame must be > 0")

        start_ts = max(0.0, float(start_ts))
        target_ts = start_ts
        idx = 0
        emitted = 0

        live_t0 = self._now() if emulate_live else None

        if emulate_live:
            log.info("🔴 LIVE EMULATION ENABLED — running in real time pacing.\n")

        with self.lock:
            if self.cap is None:
                raise RuntimeError("Capture is closed")
            start_frame = int(round(start_ts * self.src_fps))
            self._seek_frame_index(start_frame)

        while True:
            if end_ts is not None and target_ts > float(end_ts) + 1e-9:
                break
            if max_frames is not None and emitted >= max_frames:
                break

            frame = self.get_frame_at(target_ts)
            if frame is None:
                break

            # Wall-clock sync for live emulation
            if emulate_live and live_t0 is not None:
                desired_elapsed = target_ts - start_ts
                actual_elapsed = self._now() - live_t0
                sleep_s = desired_elapsed - actual_elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)

            out = frame.copy() if copy_frame else frame
            yield (idx, target_ts, out)

            emitted += 1
            idx += 1
            target_ts = start_ts + idx * seconds_per_frame

    @staticmethod
    def to_tensor(frame_bgr: np.ndarray, pin_memory: bool = True) -> "torch.Tensor":
        """
        Convert BGR uint8 HxWxC numpy frame to torch CHW float tensor in [0,1].
        Uses pinned memory if available and requested. Requires PyTorch.
        """
        if not _HAS_TORCH:
            raise ImportError("PyTorch not available.")
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Invalid frame for to_tensor().")

        rgb = frame_bgr[..., ::-1]
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
        if pin_memory:
            try:
                tensor = tensor.pin_memory()
            except Exception:
                pass
        return tensor


def stream_frames(
    video_path: str,
    start_ts: float = 0.0,
    seconds_per_frame: float = 2.0,
    end_ts: Optional[float] = None,
    emulate_live: bool = False,
    max_frames: Optional[int] = None,
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """Convenience wrapper that ensures cleanup."""
    with FrameStreamer(video_path) as fs:
        yield from fs.frames(
            start_ts=start_ts,
            seconds_per_frame=seconds_per_frame,
            end_ts=end_ts,
            emulate_live=emulate_live,
            max_frames=max_frames,
        )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="K1 Frame Extractor Demo (1 frame / 2 sec)")
    ap.add_argument("--input", "-i", type=str, default=os.path.join("Inputs", "Original_video.mp4"))
    ap.add_argument("--seconds-per-frame", "-spf", type=float, default=2.0)
    ap.add_argument("--max-frames", "-n", type=int, default=5)
    ap.add_argument("--emulate-live", action="store_true")
    args = ap.parse_args()

    try:
        with FrameStreamer(args.input) as fs:
            log.info(f"[K1] Demo: reading {args.max_frames} frames @ every {args.seconds_per_frame:.2f}s")
            for idx, ts, frame in fs.frames(
                seconds_per_frame=args.seconds_per_frame,
                emulate_live=args.emulate_live,
                max_frames=args.max_frames,
            ):
                log.info(f"  -> Frame {idx} @ {ts:.2f}s | shape={frame.shape}")
    except Exception as e:
        log.error(f"[K1] Demo failed: {e}")
