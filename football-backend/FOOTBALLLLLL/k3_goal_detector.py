# """
# k3_goal_detector.py (K-Pipeline, OCR-Restored + Live Logging)
# -------------------------------------------------------------
# Single-pass, real-time goal-change detector using OCR logic (from S-pipeline).
# Enhancements:
# - Live-mode progress with ETA, elapsed, and frame stats.
# - Big ⚽ CHANGE DETECTED message with team and new score.
# - Optional 🔴 LIVE EMULATION mode synced to wall clock.
# - CSV logging + score_progression graph preserved.
# """

# import os
# import csv
# import re
# import time
# import logging
# from pathlib import Path
# from typing import Optional, List, Dict, Any, Tuple

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator, FuncFormatter
# from tqdm import tqdm

# from ultralytics import YOLO
# import easyocr

# from k1_frames_extractor import FrameStreamer
# from k2_image_processing import enhance_for_ocr
# from k4_highlight_manager import build_event_handler

# logging.basicConfig(level=logging.INFO, format="%(message)s")
# log = logging.getLogger(__name__)

# # ----------------------------- CONFIG -----------------------------
# SECONDS_PER_FRAME = 2.0
# ROI_REFRESH_N = 10
# OCR_SKIP_ON_IDENTICAL = True
# OCR_HASH_SIZE = (96, 24)

# OUT_DIR = "k_OUTPUTS"
# CSV_PATH = os.path.join(OUT_DIR, "k_parsed_events.csv")
# GRAPH_PATH = os.path.join(OUT_DIR, "score_progression.png")

# # ----------------------------- HELPERS -----------------------------
# _num_pair_re = re.compile(r'^\s*(\d+)\s*[:\-\./\s]+\s*(\d+)\s*$')
# _num_only_re = re.compile(r'^\d+$')

# def fmt_hhmmss(seconds: float, pos=None) -> str:
#     if seconds is None or seconds < 0:
#         return "00:00:00"
#     s = int(round(seconds))
#     h, m, sec = s // 3600, (s % 3600) // 60, s % 60
#     return f"{h:02d}:{m:02d}:{sec:02d}"

# def normalize_team_singleword(s: str) -> str:
#     if not s: return ""
#     s2 = re.sub(r'[^A-Z ]+', '', s.upper()).strip()
#     parts = [p for p in s2.split() if p]
#     return parts[0] if parts else ""

# def centroid_x_from_bbox(bbox: List[List[float]]) -> float:
#     xs = [p[0] for p in bbox] if bbox else []
#     return float(sum(xs)/len(xs)) if xs else 0.0

# def split_token_scores(token: str) -> Tuple[Optional[int], Optional[int]]:
#     if token is None: return None, None
#     t = str(token).strip().upper().replace('O', '0')
#     m = _num_pair_re.match(t)
#     if m:
#         try: return int(m.group(1)), int(m.group(2))
#         except: return None, None
#     if _num_only_re.match(t) and len(t) >= 2:
#         return int(t[:-1]), int(t[-1:])
#     return None, None

# def extract_between_team_scores(sorted_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
#     texts = [e["text"].strip().upper().replace('O', '0') for e in sorted_entries if e.get("text")]
#     for i, t in enumerate(texts):
#         Ls, Rs = split_token_scores(t)
#         if Ls is not None and Rs is not None:
#             left = [normalize_team_singleword(x) for x in texts[:i] if x]
#             right = [normalize_team_singleword(x) for x in texts[i+1:] if x]
#             if left and right:
#                 return {"left_team_raw": left[0], "right_team_raw": right[0],
#                         "left_score_raw": str(Ls), "right_score_raw": str(Rs)}
#     for i in range(len(texts)-1):
#         if _num_only_re.match(texts[i]) and _num_only_re.match(texts[i+1]):
#             left = [normalize_team_singleword(x) for x in texts[:i] if x]
#             right = [normalize_team_singleword(x) for x in texts[i+2:] if x]
#             if left and right:
#                 return {"left_team_raw": left[0], "right_team_raw": right[0],
#                         "left_score_raw": texts[i], "right_score_raw": texts[i+1]}
#     return {}

# class FuzzyMapper:
#     def __init__(self, threshold:int=85):
#         self.canonical: List[str] = []
#         self.map_cache: Dict[str,str] = {}
#         self.threshold = threshold
#     def map(self, observed: str) -> str:
#         key = normalize_team_singleword(observed)
#         if not key: return ""
#         if key in self.map_cache: return self.map_cache[key]
#         if key not in self.canonical:
#             self.canonical.append(key)
#         self.map_cache[key] = key
#         return key

# def fast_roi_hash(bgr: np.ndarray) -> int:
#     if bgr is None or bgr.size == 0:
#         return 0
#     try: g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
#     except cv2.error: g = bgr[..., 0]
#     small = cv2.resize(g, OCR_HASH_SIZE, interpolation=cv2.INTER_AREA)
#     return int(np.uint64(np.sum(small.astype(np.uint64) * 1315423911)) % (1<<61))

# def detect_best_roi_in_frame(frame_bgr: np.ndarray, model: YOLO, device: Optional[str] = None) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
#     H, W = frame_bgr.shape[:2]
#     res_list = model(frame_bgr, verbose=False, device=device)
#     if not res_list or not getattr(res_list[0], "boxes", None) or len(res_list[0].boxes) == 0:
#         return np.array([]), (0,0,0,0)
#     bidx = int(res_list[0].boxes.conf.argmax())
#     best_box = res_list[0].boxes[bidx]
#     x1, y1, x2, y2 = best_box.xyxy[0].int().tolist()
#     pad_x = int(round(0.02 * W))
#     x1, x2 = max(0, x1 - pad_x), min(W - 1, x2 + pad_x)
#     y1, y2 = max(0, y1), min(H - 1, y2)
#     if y2 <= y1 or x2 <= x1:
#         return np.array([]), (0,0,0,0)
#     return frame_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)

# # ----------------------------- CORE ENGINE -----------------------------
# def run_k3_detector(
#     video: str,
#     weights: str,
#     out_dir: str = OUT_DIR,
#     device: str = "cpu",
#     snippets_dir: str = os.path.join(OUT_DIR, "k_highlights"),
#     seconds_per_frame: float = SECONDS_PER_FRAME,
#     emulate_live: bool = False,
#     save_graph: bool = True
# ) -> Dict[str, Any]:

#     Path(out_dir).mkdir(parents=True, exist_ok=True)
#     Path(snippets_dir).mkdir(parents=True, exist_ok=True)

#     header = ["frame_no","ts","t1","t2","score1","score2","change",
#               "exact_ts","detect_latency_s","frame_proc_ms",
#               "highlight_latency_s","total_latency_s","snippet_path"]
#     new_file = not os.path.exists(CSV_PATH)
#     with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
#         if new_file:
#             csv.writer(f).writerow(header)

#     log.info(f"[K3] Loading YOLO on {device.upper()}...")
#     yolo = YOLO(weights)
#     if device:
#         try: yolo.to(device)
#         except Exception: pass

#     log.info("[K3] Initializing EasyOCR...")
#     reader = easyocr.Reader(['en'], gpu=('cuda' in device), verbose=False)

#     dummy = np.zeros((224,224,3), dtype=np.uint8)
#     try:
#         _ = yolo(dummy, verbose=False, device=device)
#         _ = reader.readtext(dummy)
#         log.info("[K3] Models warmed up.")
#     except Exception:
#         pass

#     mapper = FuzzyMapper(85)
#     canonical_order: List[str] = []
#     last_scores: Dict[str, Optional[int]] = {}
#     last_bbox: Optional[Tuple[int,int,int,int]] = None
#     last_roi_hash: int = 0

#     cap = cv2.VideoCapture(video)
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
#     duration = (total_frames / fps) if fps > 0 else 0.0
#     cap.release()
#     total_samples = int(duration / seconds_per_frame) + 1

#     print("-----------------------------------------------------------------------------------")
#     print(f"VIDEO: {os.path.basename(video)} | DURATION: {fmt_hhmmss(duration)} | FRAME GAP: {seconds_per_frame:.1f}s")
#     print(f"TOTAL SAMPLES: ~{total_samples} | DEVICE: {device.upper()}")
#     if emulate_live:
#         print("🔴 LIVE EMULATION ENABLED — Running in Real-Time pacing.\n")
#     print("-----------------------------------------------------------------------------------")

#     stream_wall_t0 = time.perf_counter()
#     run_t0 = time.time()
#     detected_changes = 0
#     events: List[Dict[str, Any]] = []
#     handler = build_event_handler(snippets_dir=snippets_dir)

#     with FrameStreamer(video) as fs, tqdm(total=total_samples, desc="[K3] Processing", ncols=100) as pbar:
#         for idx, ts_s, frame in fs.frames(seconds_per_frame=seconds_per_frame, emulate_live=emulate_live):
#             t_frame_start = time.perf_counter()

#             need_refresh = (last_bbox is None) or (idx % ROI_REFRESH_N == 0)
#             if need_refresh:
#                 crop, bbox = detect_best_roi_in_frame(frame, yolo, device)
#                 if crop.size == 0:
#                     last_bbox = None
#                     pbar.update(1)
#                     continue
#                 last_bbox = bbox
#             else:
#                 x1, y1, x2, y2 = last_bbox
#                 crop = frame[y1:y2, x1:x2] if (y2>y1 and x2>x1) else np.array([])
#                 if crop.size == 0:
#                     last_bbox = None
#                     pbar.update(1)
#                     continue

#             do_ocr = True
#             if OCR_SKIP_ON_IDENTICAL:
#                 rh = fast_roi_hash(crop)
#                 if rh == last_roi_hash:
#                     do_ocr = False
#                 else:
#                     last_roi_hash = rh

#             change_flag = 0
#             if do_ocr:
#                 pre = enhance_for_ocr(crop)
#                 raw = reader.readtext(pre, detail=1)
#                 struct = sorted(
#                     [{"text": txt, "bbox": bb, "cx": centroid_x_from_bbox(bb)} for (bb,txt,conf) in raw],
#                     key=lambda x: x["cx"]
#                 )
#                 parsed = extract_between_team_scores(struct)
#                 lt_raw, rt_raw = parsed.get("left_team_raw",""), parsed.get("right_team_raw","")
#                 ls_raw, rs_raw = parsed.get("left_score_raw",""), parsed.get("right_score_raw","")
#                 lt_can, rt_can = mapper.map(lt_raw), mapper.map(rt_raw)

#                 if lt_can and rt_can and not canonical_order:
#                     canonical_order.extend([lt_can, rt_can])
#                     last_scores[lt_can], last_scores[rt_can] = None, None

#                 if len(canonical_order) >= 2 and lt_can and rt_can:
#                     team1, team2 = canonical_order[0], canonical_order[1]
#                     try: ls_tmp, rs_tmp = int(ls_raw), int(rs_raw)
#                     except: ls_tmp = rs_tmp = None
#                     if lt_can == team2 and rt_can == team1:
#                         ls_tmp, rs_tmp = rs_tmp, ls_tmp
#                     prev1, prev2 = last_scores.get(team1), last_scores.get(team2)
#                     if ls_tmp is not None and rs_tmp is not None:
#                         s1_changed = (prev1 is not None and ls_tmp == prev1 + 1)
#                         s2_changed = (prev2 is not None and rs_tmp == prev2 + 1)
#                         if s1_changed and s2_changed: change_flag = 12
#                         elif s1_changed: change_flag = 1
#                         elif s2_changed: change_flag = 2
#                         last_scores[team1], last_scores[team2] = ls_tmp, rs_tmp

#             frame_proc_ms = (time.perf_counter() - t_frame_start) * 1000.0

#             if change_flag and len(canonical_order) >= 2:
#                 team1, team2 = canonical_order[0], canonical_order[1]
#                 exact_ts = ts_s
#                 detect_latency_s = max(0.0, (time.perf_counter() - stream_wall_t0) - exact_ts)
#                 row = {
#                     "frame_no": idx, "ts": ts_s,
#                     "t1": team1, "t2": team2,
#                     "score1": last_scores.get(team1, ""),
#                     "score2": last_scores.get(team2, ""),
#                     "change": change_flag, "exact_ts": exact_ts,
#                     "detect_latency_s": round(detect_latency_s, 3),
#                     "frame_proc_ms": round(frame_proc_ms, 2),
#                     "highlight_latency_s": "", "total_latency_s": "", "snippet_path": ""
#                 }
#                 try:
#                     event_res = handler({
#                         "video_path": video,
#                         "event_row": row.copy(),
#                         "suggested_name": f"Highlight_{row['score1']}_{row['score2']}.mp4"
#                     }) or {}
#                     row["highlight_latency_s"] = round(float(event_res.get("highlight_latency_s", 0.0)), 3)
#                     row["total_latency_s"] = round(float(event_res.get("total_latency_s", row["detect_latency_s"])), 3)
#                     row["snippet_path"] = event_res.get("snippet_path", "")
#                 except Exception as e:
#                     log.warning(f"[K3] Highlight handler error: {e}")

#                 with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
#                     csv.writer(f).writerow([row[h] for h in header])

#                 events.append(row)
#                 detected_changes += 1
#                 scorer = team1 if change_flag in (1,12) else team2
#                 print(f"\n⚽ CHANGE DETECTED!!!!!!!! ({fmt_hhmmss(exact_ts)})")
#                 print(f"   Team: {scorer}")
#                 print(f"   New Score: {row['score1']} - {row['score2']}")
#                 print(f"   Latencies → Detect: {row['detect_latency_s']}s | Highlight: {row['highlight_latency_s']}s | Total: {row['total_latency_s']}s")
#                 if row['snippet_path']: print(f"   Highlight saved → {row['snippet_path']}")

#             pbar.update(1)

#     print("\n-----------------------------------------------------------------------------------")
#     print(f"✅ [K3] Completed | Events detected: {detected_changes} | CSV → {CSV_PATH}")

#     if save_graph and events:
#         try:
#             t1, t2 = events[0]["t1"], events[0]["t2"]
#             xs, s1_vals, s2_vals = [], [], []
#             for ev in sorted(events, key=lambda e: e["exact_ts"]):
#                 xs.append(ev["exact_ts"])
#                 s1_vals.append(int(ev["score1"]))
#                 s2_vals.append(int(ev["score2"]))
#             plt.figure(figsize=(12, 5))
#             plt.plot(xs, s1_vals, "-o", label=t1)
#             plt.plot(xs, s2_vals, "-o", label=t2)
#             plt.xlabel("Match Time (HH:MM:SS)")
#             plt.ylabel("Score")
#             plt.grid(alpha=0.3)
#             plt.legend()
#             plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda v, p: fmt_hhmmss(v)))
#             plt.tight_layout()
#             plt.savefig(GRAPH_PATH)
#             plt.close()
#             print(f" Graph saved → {GRAPH_PATH}")
#         except Exception as e:
#             print(f"⚠️ Graph generation error: {e}")

#     print("-----------------------------------------------------------------------------------")
#     return {"csv_path": CSV_PATH, "graph_path": GRAPH_PATH, "total_events": detected_changes}

# # ----------------------------- CLI -----------------------------
# if __name__ == "__main__":
#     import argparse
#     ap = argparse.ArgumentParser(description="K3 OCR Goal Detector with Live Progress")
#     ap.add_argument("--video", "-v", type=str, default=os.path.join("Inputs","Original_video.mp4"))
#     ap.add_argument("--weights", "-w", type=str, default=os.path.join("Models","Yolo_Scoreboard_best.pt"))
#     ap.add_argument("--device", type=str, default="cpu")
#     ap.add_argument("--spf", type=float, default=SECONDS_PER_FRAME)
#     ap.add_argument("--live", action="store_true")
#     args = ap.parse_args()
#     run_k3_detector(args.video, args.weights, device=args.device, seconds_per_frame=args.spf, emulate_live=args.live)
"""
k3_goal_detector.py
-------------------
Real-time goal-change detector with live console timer and playback-speed indicator.

Features (logic preserved):
- YOLO scoreboard detection + EasyOCR team/score parsing
- Constant sampling (default: 1 frame / 2 seconds)
- Single-line live timer showing Elapsed, Video TS, and speed ratio
- Prints goal-change events as bursts
- Generates CSV, score graph, and run report with timings

Additions:
- Optional `progress_callback(dict)` invoked for progress and goal events
- No emojis in logs/prints
"""

import os
import csv
import re
import sys
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from ultralytics import YOLO
import easyocr

from k1_frames_extractor import FrameStreamer
from k2_image_processing import enhance_for_ocr
from k4_highlight_manager import build_event_handler

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ----------------------------- CONFIG -----------------------------
SECONDS_PER_FRAME = 2.0
ROI_REFRESH_N = 10
OCR_SKIP_ON_IDENTICAL = True
OCR_HASH_SIZE = (96, 24)
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, "k_parsed_events.csv")
GRAPH_PATH = os.path.join(OUT_DIR, "score_progression.png")
REPORT_PATH = os.path.join(OUT_DIR, "k3_run_report.txt")

# ----------------------------- HELPERS -----------------------------
_num_pair_re = re.compile(r'^\s*(\d+)\s*[:\-\./\s]+\s*(\d+)\s*$')
_num_only_re = re.compile(r'^\d+$')

def fmt_hhmmss(seconds: float, pos=None) -> str:
    if seconds is None or seconds < 0:
        return "00:00:00"
    s = int(round(seconds))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"

def normalize_team_singleword(s: str) -> str:
    if not s:
        return ""
    s2 = re.sub(r'[^A-Z ]+', '', s.upper()).strip()
    parts = [p for p in s2.split() if p]
    return parts[0] if parts else ""

def centroid_x_from_bbox(bbox: List[List[float]]) -> float:
    xs = [p[0] for p in bbox] if bbox else []
    return float(sum(xs) / len(xs)) if xs else 0.0

def split_token_scores(token: str) -> Tuple[Optional[int], Optional[int]]:
    if token is None:
        return None, None
    t = str(token).strip().upper().replace('O', '0')
    m = _num_pair_re.match(t)
    if m:
        try:
            return int(m.group(1)), int(m.group(2))
        except:
            return None, None
    if _num_only_re.match(t) and len(t) >= 2:
        return int(t[:-1]), int(t[-1:])
    return None, None

def extract_between_team_scores(sorted_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    texts = [e["text"].strip().upper().replace('O', '0') for e in sorted_entries if e.get("text")]
    for i, t in enumerate(texts):
        Ls, Rs = split_token_scores(t)
        if Ls is not None and Rs is not None:
            left = [normalize_team_singleword(x) for x in texts[:i] if x]
            right = [normalize_team_singleword(x) for x in texts[i + 1:] if x]
            if left and right:
                return {
                    "left_team_raw": left[0],
                    "right_team_raw": right[0],
                    "left_score_raw": str(Ls),
                    "right_score_raw": str(Rs),
                }
    for i in range(len(texts) - 1):
        if _num_only_re.match(texts[i]) and _num_only_re.match(texts[i + 1]):
            left = [normalize_team_singleword(x) for x in texts[:i] if x]
            right = [normalize_team_singleword(x) for x in texts[i + 2:] if x]
            if left and right:
                return {
                    "left_team_raw": left[0],
                    "right_team_raw": right[0],
                    "left_score_raw": texts[i],
                    "right_score_raw": texts[i + 1],
                }
    return {}

class FuzzyMapper:
    def __init__(self):
        self.canonical: List[str] = []
        self.map_cache: Dict[str, str] = {}
    def map(self, observed: str) -> str:
        key = normalize_team_singleword(observed)
        if not key:
            return ""
        if key in self.map_cache:
            return self.map_cache[key]
        if key not in self.canonical:
            self.canonical.append(key)
        self.map_cache[key] = key
        return key

def fast_roi_hash(bgr: np.ndarray) -> int:
    if bgr is None or bgr.size == 0:
        return 0
    try:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        g = bgr[..., 0]
    small = cv2.resize(g, OCR_HASH_SIZE, interpolation=cv2.INTER_AREA)
    return int(np.uint64(np.sum(small.astype(np.uint64) * 1315423911)) % (1 << 61))

def detect_best_roi_in_frame(frame_bgr: np.ndarray, model: YOLO, device: Optional[str] = None):
    H, W = frame_bgr.shape[:2]
    res_list = model(frame_bgr, verbose=False, device=device)
    if not res_list or not getattr(res_list[0], "boxes", None) or len(res_list[0].boxes) == 0:
        return np.array([]), (0, 0, 0, 0)
    bidx = int(res_list[0].boxes.conf.argmax())
    x1, y1, x2, y2 = res_list[0].boxes[bidx].xyxy[0].int().tolist()
    pad_x = int(round(0.02 * W))
    x1, x2 = max(0, x1 - pad_x), min(W - 1, x2 + pad_x)
    y1, y2 = max(0, y1), min(H - 1, y2)
    if y2 <= y1 or x2 <= x1:
        return np.array([]), (0, 0, 0, 0)
    return frame_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)

# ----------------------------- CORE ENGINE -----------------------------
def run_k3_detector(
    video: str,
    weights: str,
    out_dir: str = OUT_DIR,
    device: str = "cpu",
    snippets_dir: str = os.path.join(OUT_DIR, "k_highlights"),
    seconds_per_frame: float = SECONDS_PER_FRAME,
    emulate_live: bool = False,
    save_graph: bool = True,
    progress_callback=None,
) -> Dict[str, Any]:

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(snippets_dir).mkdir(parents=True, exist_ok=True)

    header = [
        "frame_no",
        "ts",
        "t1",
        "t2",
        "score1",
        "score2",
        "change",
        "exact_ts",
        "detect_latency_s",
        "frame_proc_ms",
        "highlight_latency_s",
        "total_latency_s",
        "snippet_path",
    ]
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    print("-----------------------------------------------------------------------------------")
    print(f"VIDEO SOURCE: {os.path.basename(video)} | FRAME GAP: {seconds_per_frame:.1f}s")
    print(f"DEVICE: {device.upper()} | MODE: {'LIVE STREAM EMULATION' if emulate_live else 'FAST OFFLINE'}")
    print("-----------------------------------------------------------------------------------\n")

    yolo = YOLO(weights)
    if device:
        try:
            yolo.to(device)
        except:
            pass
    reader = easyocr.Reader(['en'], gpu=('cuda' in device), verbose=False)

    # warmup
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    _ = yolo(dummy, verbose=False, device=device)
    _ = reader.readtext(dummy)

    mapper = FuzzyMapper()
    canonical_order: List[str] = []
    last_scores: Dict[str, Optional[int]] = {}
    last_bbox = None
    last_roi_hash = 0
    handler = build_event_handler(snippets_dir=snippets_dir)

    stream_wall_t0 = time.perf_counter()
    run_start_wall = time.perf_counter()
    events: List[Dict[str, Any]] = []
    detected_changes = 0

    with FrameStreamer(video) as fs:
        for idx, ts_s, frame in fs.frames(seconds_per_frame=seconds_per_frame, emulate_live=emulate_live):
            elapsed_wall = time.perf_counter() - run_start_wall
            speed_ratio = (ts_s / elapsed_wall) if elapsed_wall > 0 else 0.0
            sys.stdout.write(
                f"\rLIVE | Elapsed: {fmt_hhmmss(elapsed_wall)} | Video TS: {fmt_hhmmss(ts_s)} | Frame #{idx:05d} | x{speed_ratio:4.2f}"
            )
            sys.stdout.flush()

            if progress_callback:
                try:
                    progress_callback({
                        "type": "progress",
                        "frame_no": idx,
                        "elapsed": fmt_hhmmss(elapsed_wall),
                        "video_time": fmt_hhmmss(ts_s),
                        "speed_ratio": round(speed_ratio, 2),
                    })
                except Exception as e:
                    log.warning(f"Progress callback failed: {e}")

            frame_t0 = time.perf_counter()
            need_refresh = (last_bbox is None) or (idx % ROI_REFRESH_N == 0)
            crop, bbox = (np.array([]), (0, 0, 0, 0))
            if need_refresh:
                crop, bbox = detect_best_roi_in_frame(frame, yolo, device)
                if crop.size == 0:
                    last_bbox = None
                    continue
                last_bbox = bbox
            else:
                x1, y1, x2, y2 = last_bbox
                crop = frame[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else np.array([])
                if crop.size == 0:
                    last_bbox = None
                    continue

            do_ocr = True
            if OCR_SKIP_ON_IDENTICAL:
                rh = fast_roi_hash(crop)
                if rh == last_roi_hash:
                    do_ocr = False
                else:
                    last_roi_hash = rh

            change_flag = 0
            if do_ocr:
                pre = enhance_for_ocr(crop)
                raw = reader.readtext(pre, detail=1)
                struct = sorted(
                    [{"text": txt, "bbox": bb, "cx": centroid_x_from_bbox(bb)} for (bb, txt, conf) in raw],
                    key=lambda x: x["cx"],
                )
                parsed = extract_between_team_scores(struct)
                lt_raw, rt_raw = parsed.get("left_team_raw", ""), parsed.get("right_team_raw", "")
                ls_raw, rs_raw = parsed.get("left_score_raw", ""), parsed.get("right_score_raw", "")
                lt_can, rt_can = mapper.map(lt_raw), mapper.map(rt_raw)

                if lt_can and rt_can and not canonical_order:
                    canonical_order = [lt_can, rt_can]
                    last_scores[lt_can] = None
                    last_scores[rt_can] = None

                if len(canonical_order) >= 2 and lt_can and rt_can:
                    team1, team2 = canonical_order
                    try:
                        ls_tmp = int(ls_raw)
                    except:
                        ls_tmp = None
                    try:
                        rs_tmp = int(rs_raw)
                    except:
                        rs_tmp = None

                    # Swap if OCR order is reversed
                    if lt_can == team2 and rt_can == team1:
                        ls_tmp, rs_tmp = rs_tmp, ls_tmp

                    prev1, prev2 = last_scores.get(team1), last_scores.get(team2)
                    if ls_tmp is not None and rs_tmp is not None:
                        s1_changed = (prev1 is not None and ls_tmp == prev1 + 1)
                        s2_changed = (prev2 is not None and rs_tmp == prev2 + 1)
                        if s1_changed and s2_changed:
                            change_flag = 12
                        elif s1_changed:
                            change_flag = 1
                        elif s2_changed:
                            change_flag = 2
                        last_scores[team1], last_scores[team2] = ls_tmp, rs_tmp

            frame_proc_ms = (time.perf_counter() - frame_t0) * 1000.0

            if change_flag and len(canonical_order) >= 2:
                team1, team2 = canonical_order
                exact_ts = ts_s
                detect_latency_s = max(0.0, (time.perf_counter() - stream_wall_t0) - exact_ts)
                row = {
                    "frame_no": idx,
                    "ts": ts_s,
                    "t1": team1,
                    "t2": team2,
                    "score1": last_scores.get(team1, ""),
                    "score2": last_scores.get(team2, ""),
                    "change": change_flag,
                    "exact_ts": exact_ts,
                    "detect_latency_s": round(detect_latency_s, 3),
                    "frame_proc_ms": round(frame_proc_ms, 2),
                    "highlight_latency_s": "",
                    "total_latency_s": "",
                    "snippet_path": "",
                }

                try:
                    event_res = handler(
                        {
                            "video_path": video,
                            "event_row": row.copy(),
                            "suggested_name": f"Highlight_{row['score1']}_{row['score2']}.mp4",
                        }
                    ) or {}
                    row["highlight_latency_s"] = round(float(event_res.get("highlight_latency_s", 0.0)), 3)
                    row["total_latency_s"] = round(
                        float(event_res.get("total_latency_s", row["detect_latency_s"])), 3
                    )
                    row["snippet_path"] = event_res.get("snippet_path", "")
                except Exception as e:
                    log.warning(f"[K3] Highlight handler error: {e}")

                with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([row[h] for h in header])

                events.append(row)
                detected_changes += 1

                print("\n\nCHANGE DETECTED")
                scorer = team1 if change_flag in (1, 12) else team2
                print(f"   Team: {scorer}")
                print(f"   New Score: {row['score1']} - {row['score2']}")
                print(
                    f"   Latencies -> Detect: {row['detect_latency_s']}s | Highlight: {row['highlight_latency_s']}s | Total: {row['total_latency_s']}s\n"
                )

                if progress_callback:
                    try:
                        progress_callback({
                            "type": "goal",
                            "frame_no": idx,
                            "video_time": fmt_hhmmss(exact_ts),
                            "team": scorer,
                            "score": f"{row['score1']}-{row['score2']}",
                            "snippet_path": row["snippet_path"],
                            "detect_latency_s": row["detect_latency_s"],
                            "highlight_latency_s": row["highlight_latency_s"],
                            "total_latency_s": row["total_latency_s"],
                        })
                    except Exception as e:
                        log.warning(f"Progress callback (goal) failed: {e}")

    # --- END SUMMARY ---
    end_wall = time.perf_counter()
    total_live_s = end_wall - run_start_wall
    video_time = events[-1]["exact_ts"] if events else 0.0

    print("\n-----------------------------------------------------------------------------------")
    print(f"[K3] Completed | Goals detected: {detected_changes}")
    print(f"   Stream duration (real time): {fmt_hhmmss(total_live_s)}")
    print(f"   Video time processed:        {fmt_hhmmss(video_time)}")
    print(f"   CSV saved -> {CSV_PATH}")

    # --- GRAPH ---
    graph_t0 = time.perf_counter()
    if save_graph and events:
        try:
            team1, team2 = events[0].get("t1", "Team1"), events[0].get("t2", "Team2")
            xs, s1_vals, s2_vals = [], [], []
            for ev in sorted(events, key=lambda e: e["exact_ts"]):
                xs.append(ev["exact_ts"])
                s1_vals.append(int(ev["score1"]))
                s2_vals.append(int(ev["score2"]))
            plt.figure(figsize=(12, 5))
            plt.plot(xs, s1_vals, "-o", label=team1)
            plt.plot(xs, s2_vals, "-o", label=team2)
            plt.xlabel("Match Time (HH:MM:SS)")
            plt.ylabel("Score")
            ax = plt.gca()
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: fmt_hhmmss(v)))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.grid(True, alpha=0.3)
            plt.legend(loc="upper left")
            plt.tight_layout()
            plt.savefig(GRAPH_PATH)
            plt.close()
            print(f"   Graph saved -> {GRAPH_PATH}")
        except Exception as e:
            print(f"   Graph generation skipped (error: {e})")
    graph_time = time.perf_counter() - graph_t0

    # --- REPORT ---
    rep_t0 = time.perf_counter()
    with open(REPORT_PATH, "w", encoding="utf-8") as rep:
        rep.write("K3 PIPELINE RUN REPORT\n")
        rep.write("===========================================\n")
        rep.write(f"Video Source: {os.path.basename(video)}\n")
        rep.write(f"Device: {device}\n")
        rep.write(f"Live Mode: {emulate_live}\n")
        rep.write(f"Goals Detected: {detected_changes}\n")
        rep.write(f"Stream Duration (real): {fmt_hhmmss(total_live_s)}\n")
        rep.write(f"Video Duration Processed: {fmt_hhmmss(video_time)}\n")
        rep.write(f"CSV Path: {CSV_PATH}\n")
        rep.write(f"Graph Path: {GRAPH_PATH}\n")
        rep.write("===========================================\n")
    report_time = time.perf_counter() - rep_t0
    print(f"   Report saved -> {REPORT_PATH}")
    print(f"   CSV+Graph+Report generation time: {graph_time + report_time:.2f}s")
    print("-----------------------------------------------------------------------------------")

    return {
        "csv_path": CSV_PATH,
        "graph_path": GRAPH_PATH,
        "report_path": REPORT_PATH,
        "total_events": detected_changes,
    }

# ----------------------------- CLI -----------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="K3 (OCR) Goal Detector with Live Console + Report + Speed Monitor")
    ap.add_argument("--video", "-v", type=str, default=os.path.join("Inputs", "Original_video.mp4"))
    ap.add_argument("--weights", "-w", type=str, default=os.path.join("Models", "Yolo_Scoreboard_best.pt"))
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--spf", type=float, default=SECONDS_PER_FRAME)
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    run_k3_detector(
        video=args.video,
        weights=args.weights,
        device=args.device,
        seconds_per_frame=args.spf,
        emulate_live=args.live,
    )
