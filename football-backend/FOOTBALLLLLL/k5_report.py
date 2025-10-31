"""
k5_report.py (K-Pipeline)
-------------------------
Lightweight final report generator for the K-pipeline.

Reads:  k_OUTPUTS/k_parsed_events.csv
Writes: k_OUTPUTS/k_FINAL_REPORT.txt
Also references: k_OUTPUTS/score_progression.png (if present)

Report includes:
- Video metadata (optional if --video provided)
- Per-event table with separate latencies:
    * detect_latency_s
    * highlight_latency_s
    * total_latency_s
- Final score + winner/draw
- Averages of each latency column
- System & hardware info (OS, CPU, RAM, GPU, Disk)
"""

import os
import csv
import time
import math
import platform
from pathlib import Path
from typing import List, Dict, Any, Optional

# Optional system libs
try:
    import psutil
except Exception:
    psutil = None

try:
    import torch
except Exception:
    torch = None

import cv2


DEFAULT_OUT_DIR = os.path.join("k_OUTPUTS")
DEFAULT_CSV     = os.path.join(DEFAULT_OUT_DIR, "k_parsed_events.csv")
DEFAULT_REPORT  = os.path.join(DEFAULT_OUT_DIR, "k_FINAL_REPORT.txt")
DEFAULT_GRAPH   = os.path.join(DEFAULT_OUT_DIR, "score_progression.png")


def fmt_hhmmss(seconds: float) -> str:
    if seconds is None or seconds < 0:
        return "00:00:00"
    s = int(round(seconds))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def read_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None or x == "":
            return default
        return int(x)
    except Exception:
        return default


def summarize_latencies(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    det_vals, high_vals, total_vals = [], [], []
    for r in rows:
        d = r.get("detect_latency_s", "")
        h = r.get("highlight_latency_s", "")
        t = r.get("total_latency_s", "")
        if d != "": det_vals.append(safe_float(d))
        if h != "": high_vals.append(safe_float(h))
        if t != "": total_vals.append(safe_float(t))
    def avg(v: List[float]) -> float:
        return sum(v) / len(v) if v else 0.0
    return {
        "avg_detect": round(avg(det_vals), 3),
        "avg_highlight": round(avg(high_vals), 3),
        "avg_total": round(avg(total_vals), 3),
        "count": len(rows)
    }


def compute_final_score(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Determine team names and final score from the last known values in CSV.
    """
    team1 = team2 = ""
    last_s1 = last_s2 = 0
    for r in rows:
        if r.get("t1"): team1 = r["t1"]
        if r.get("t2"): team2 = r["t2"]
        s1 = safe_int(r.get("score1"), None)
        s2 = safe_int(r.get("score2"), None)
        if s1 is not None: last_s1 = s1
        if s2 is not None: last_s2 = s2
    return {"team1": team1, "team2": team2, "score1": last_s1, "score2": last_s2}


def winner_line(team1: str, s1: int, team2: str, s2: int) -> str:
    if s1 == s2:
        return "Result: Draw"
    if s1 > s2:
        return f"Result: {team1} won by {s1 - s2}"
    return f"Result: {team2} won by {s2 - s1}"


def get_video_meta(video_path: Optional[str]) -> Dict[str, Any]:
    meta = {"fps": 0.0, "frames": 0, "duration_s": 0.0}
    if not video_path or not os.path.exists(video_path):
        return meta
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return meta
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    dur = (frames / fps) if (fps and fps > 0) else 0.0
    meta.update({"fps": fps, "frames": frames, "duration_s": dur})
    return meta


def system_info(out_dir_for_disk: str) -> List[str]:
    lines = []
    try:
        uname = platform.uname()
        lines.append(f"OS:            {uname.system} {uname.release}")
        lines.append(f"Architecture:  {uname.machine}")
        lines.append(f"Python Ver:    {platform.python_version()}")
        if psutil:
            try:
                cpu_freq = psutil.cpu_freq()
                mem = psutil.virtual_memory()
                lines.append(f"CPU:           {psutil.cpu_count(logical=False)} Cores, {psutil.cpu_count(logical=True)} Threads")
                if cpu_freq:
                    lines.append(f"CPU Freq:      {cpu_freq.current:.0f}MHz (Max: {cpu_freq.max:.0f}MHz)")
                lines.append(f"Memory (RAM):  {mem.total/(1024**3):.1f}GB ({mem.percent}%)")
            except Exception as e:
                lines.append(f"CPU/MEM Info:  Error ({e})")
        if torch and torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    lines.append(f"GPU {i}:        {props.name} ({props.total_memory/(1024**3):.1f}GB)")
            except Exception as e:
                lines.append(f"GPU:           CUDA available, error reading props ({e})")
        else:
            lines.append("GPU:           Not Available")
        try:
            Path(out_dir_for_disk).mkdir(parents=True, exist_ok=True)
            st = os.statvfs(out_dir_for_disk) if hasattr(os, "statvfs") else None
            if st:
                total = st.f_frsize * st.f_blocks
                free = st.f_frsize * st.f_bfree
            else:
                # Fallback using shutil for cross-platform
                import shutil as _sh
                du = _sh.disk_usage(out_dir_for_disk)
                total, free = du.total, du.free
            used = total - free
            used_pct = (used / total * 100.0) if total > 0 else 0.0
            lines.append(f"Disk (Output): {free/(1024**3):.1f}GB Free ({used_pct:.1f}% used)")
        except Exception as e:
            lines.append(f"Disk:          Error ({e})")
    except Exception as e:
        lines.append(f"System Info:   Error ({e})")
    return lines


def build_report(
    csv_path: str = DEFAULT_CSV,
    report_path: str = DEFAULT_REPORT,
    video_path: Optional[str] = None,
    graph_path: str = DEFAULT_GRAPH
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    rows = read_csv(csv_path)
    out_dir = os.path.dirname(report_path) or "."
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Video meta (optional)
    vmeta = get_video_meta(video_path)
    duration_hhmmss = fmt_hhmmss(vmeta["duration_s"]) if vmeta["duration_s"] else "Unknown"
    fps_str = f"{vmeta['fps']:.2f}" if vmeta["fps"] else "Unknown"
    frames_str = f"{vmeta['frames']}" if vmeta["frames"] else "Unknown"

    # Scores & latencies
    final = compute_final_score(rows)
    lat = summarize_latencies(rows)

    team1 = final["team1"] or "Team 1"
    team2 = final["team2"] or "Team 2"
    s1 = int(final["score1"])
    s2 = int(final["score2"])

    # Build the report text
    header = "K5 FINAL MATCH REPORT"
    sep = "=" * max(len(header), 60)
    lines: List[str] = []
    lines.append(header)
    lines.append(sep)
    if video_path:
        lines.append(f"Video:         {os.path.abspath(video_path)}")
    lines.append(f"CSV Source:    {os.path.abspath(csv_path)}")
    lines.append(f"Duration:      {duration_hhmmss}  |  FPS: {fps_str}  |  Frames: {frames_str}")
    if os.path.exists(graph_path):
        lines.append(f"Graph:         {os.path.abspath(graph_path)}")
    lines.append("")

    # Per-event listing
    lines.append("DETECTED EVENTS")
    lines.append("-" * 80)
    lines.append("TIME (HH:MM:SS) | SCORE | DETECT_LAT(s) | HIGHLIGHT_LAT(s) | TOTAL_LAT(s) | CLIP_NAME")
    lines.append("-" * 80)

    for r in rows:
        ts = safe_float(r.get("exact_ts"), 0.0)
        sc1 = r.get("score1", "")
        sc2 = r.get("score2", "")
        det = r.get("detect_latency_s", "")
        high = r.get("highlight_latency_s", "")
        tot = r.get("total_latency_s", "")
        clip = os.path.basename(r.get("snippet_path", "")) if r.get("snippet_path") else ""
        lines.append(
            f"{fmt_hhmmss(ts):>14s} | {str(sc1)}-{str(sc2):<3s} | {str(det):>13s} | {str(high):>16s} | {str(tot):>11s} | {clip}"
        )

    lines.append("-" * 80)
    lines.append("")
    lines.append("FINAL SCORE")
    lines.append(f"  {team1} {s1} - {s2} {team2}")
    lines.append(f"  {winner_line(team1, s1, team2, s2)}")
    lines.append("")
    lines.append("LATENCY AVERAGES")
    lines.append(f"  Average Detection Latency: {lat['avg_detect']} s")
    lines.append(f"  Average Highlight Latency: {lat['avg_highlight']} s")
    lines.append(f"  Average Total Latency:     {lat['avg_total']} s")
    lines.append("")

    # System section
    lines.append("SYSTEM & HARDWARE INFO")
    lines.append("-" * 80)
    lines.extend(system_info(out_dir))
    lines.append("-" * 80)

    t1 = time.perf_counter()
    lines.append(f"Report generated in {t1 - t0:.2f}s")

    # Write report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {
        "report_path": report_path,
        "events": len(rows),
        "final_score": f"{team1} {s1} - {s2} {team2}",
        "avg_detect_latency_s": lat["avg_detect"],
        "avg_highlight_latency_s": lat["avg_highlight"],
        "avg_total_latency_s": lat["avg_total"],
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="K5 Report Generator")
    ap.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Path to k_parsed_events.csv")
    ap.add_argument("--report", type=str, default=DEFAULT_REPORT, help="Output TXT path")
    ap.add_argument("--video", type=str, default=None, help="Optional video path for duration/FPS")
    ap.add_argument("--graph", type=str, default=DEFAULT_GRAPH, help="Optional PNG graph path")
    args = ap.parse_args()

    res = build_report(
        csv_path=args.csv,
        report_path=args.report,
        video_path=args.video,
        graph_path=args.graph
    )
    print("K5 Report Summary:", res)
