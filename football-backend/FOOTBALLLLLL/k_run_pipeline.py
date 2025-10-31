# """
# k_run_pipeline.py
# -----------------
# Unified controller for the K-Pipeline.
# Runs the full pipeline (K1–K5) sequentially with minimal latency.

# Pipeline Flow:
#   [Video Input] → K3 (Detection + OCR + Logging)
#                 → K4 (Highlights + Latencies)
#                 → K5 (Report Summary)
# """

# import os
# import sys
# import time
# import torch
# import logging
# from pathlib import Path

# # --- Local imports ---
# from k3_goal_detector import run_k3_detector
# from k5_report import build_report

# # --- Logging setup ---
# logging.basicConfig(level=logging.INFO, format="%(message)s")
# log = logging.getLogger("K-RUN")

# # --- Configurable paths ---
# CWD = os.path.dirname(os.path.abspath(__file__))
# INPUT_VIDEO = os.path.join(CWD, "Inputs", "FB BULI Bayern München - Dortmund 07. Spieltag 2526 Scoutingfeed.mp4")
# YOLO_WEIGHTS = os.path.join(CWD, "Models", "Yolo_Scoreboard_best.pt")
# OUT_DIR = os.path.join(CWD, "k_OUTPUTS")
# Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# CSV_PATH = os.path.join(OUT_DIR, "k_parsed_events.csv")
# REPORT_PATH = os.path.join(OUT_DIR, "k_FINAL_REPORT.txt")
# GRAPH_PATH = os.path.join(OUT_DIR, "score_progression.png")
# SNIPPETS_DIR = os.path.join(OUT_DIR, "k_highlights")

# # --- Device selection ---
# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# log.info(f"🚀 K-PIPELINE INITIALIZED | Using device: {DEVICE.upper()}")

# # --- File checks ---
# def check_file(p: str, name: str) -> bool:
#     if not os.path.exists(p):
#         log.error(f"❌ {name} not found at: {p}")
#         return False
#     return True

# if not all([check_file(INPUT_VIDEO, "Input video"), check_file(YOLO_WEIGHTS, "YOLO weights")]):
#     sys.exit("Missing required files. Please check paths and rerun.")

# # --- Pipeline execution ---
# if __name__ == "__main__":
#     log.info("\n============================")
#     log.info("🔥 K-PIPELINE: FULL EXECUTION")
#     log.info("============================\n")

#     total_start = time.time()

#     try:
#         # Stage 3 + 4 combined inside k3_goal_detector
#         log.info("[Stage K3+K4] Detecting score changes and creating highlights...\n")
#         res_k3 = run_k3_detector(
#             video=INPUT_VIDEO,
#             weights=YOLO_WEIGHTS,
#             out_dir=OUT_DIR,
#             device=DEVICE,
#             snippets_dir=SNIPPETS_DIR,
#             seconds_per_frame=2.0,   # 1 frame per 2 seconds
#             emulate_live=True,       # ✅ Enable real-time live emulation
#             save_graph=True
#         )

#         log.info(f"\n✅ Stage K3+K4 complete. Parsed CSV saved to: {res_k3['csv_path']}\n")

#         # Stage 5: Report generation
#         log.info("[Stage K5] Generating final report...\n")
#         res_k5 = build_report(
#             csv_path=CSV_PATH,
#             report_path=REPORT_PATH,
#             video_path=INPUT_VIDEO,
#             graph_path=GRAPH_PATH
#         )

#         log.info("\n✅ Stage K5 complete.")
#         log.info(f"📄 Final report saved to: {res_k5['report_path']}")
#         log.info(f"🏁 Final score: {res_k5['final_score']}")
#         log.info(f"⏱  Avg Total Latency: {res_k5['avg_total_latency_s']} s")

#     except Exception as e:
#         log.error(f"FATAL: Pipeline execution failed: {e}")
#         sys.exit(1)

#     total_time = time.time() - total_start
#     h, m, s = int(total_time // 3600), int((total_time % 3600) // 60), int(total_time % 60)

#     log.info("\n=====================================")
#     log.info("✅ K-PIPELINE COMPLETE")
#     log.info(f"Total runtime: {h}h {m}m {s}s")
#     log.info(f"Outputs saved in: {OUT_DIR}")
#     log.info("=====================================\n")
# k_run_pipeline.py
# -------------------------------------------------------------
# Unified controller for the K-Pipeline with live progress output.
# - Works standalone (python k_run_pipeline.py --video <path>)
# - Exposes an async generator `run_k_pipeline_live(video_path)`
#   that yields progress + goal snippet events for FastAPI.
# -------------------------------------------------------------
import os
import sys
import time
import argparse
import logging
import threading
import asyncio
from pathlib import Path
import torch

# --- Local imports ---
from FOOTBALLLLLL.k3_goal_detector import run_k3_detector

from k5_report import build_report


import subprocess
import shlex
from pathlib import Path
import os

def ensure_browser_playable(video_path: str):
    """
    Converts a video file to browser-friendly MP4 (H.264 + AAC) in-place.
    Handles spaces in file/folder names safely.
    """
    src = Path(video_path)
    tmp_path = src.with_suffix(".web.mp4")

    # Quote paths safely
    quoted_input = shlex.quote(str(src))
    quoted_output = shlex.quote(str(tmp_path))

    cmd = f'ffmpeg -y -i {quoted_input} -vcodec libx264 -acodec aac -movflags +faststart {quoted_output}'

    try:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(tmp_path, src)
        print(f"🎞️ Converted {src} → browser-safe MP4")
    except Exception as e:
        print(f"⚠️ FFmpeg conversion failed for {src}: {e}")

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("K-RUN")

# --- Defaults / Paths ---
CWD = Path(__file__).resolve().parent
DEFAULT_INPUT = CWD / "Inputs" / "FB BULI Bayern München - Dortmund 07. Spieltag 2526 Scoutingfeed.mp4"
YOLO_WEIGHTS = CWD / "Models" / "Yolo_Scoreboard_best.pt"

# Output directory now points to FastAPI's "outputs/"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUT_DIR / "k_parsed_events.csv"
REPORT_PATH = OUT_DIR / "k_FINAL_REPORT.txt"
GRAPH_PATH = OUT_DIR / "score_progression.png"
SNIPPETS_DIR = OUT_DIR / "k_highlights"
SNIPPETS_DIR.mkdir(parents=True, exist_ok=True)

# --- Device selection ---
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
log.info(f"K-PIPELINE INITIALIZED | Using device: {DEVICE.upper()}")

# -------------------------------------------------------------
# Async generator for FastAPI: yields progress + goal clip events
# -------------------------------------------------------------
async def run_k_pipeline_live(video_path: str):
    """
    Async generator that runs K3→K5 and streams updates to FastAPI.

    Yields:
        {"type": "status", "message": "..."}
        {"type": "goal", "goal_id": int, "clip_name": "filename.mp4", "clip_rel_path": "outputs/k_highlights/filename.mp4"}
        {"type": "done", "elapsed": float, "outputs_dir": str}
    """
    start_ts = time.time()
    yield {"type": "status", "message": f"Pipeline started for {os.path.basename(video_path)}"}
    for old in SNIPPETS_DIR.glob("*.mp4"):
        try:
            old.unlink()
            print(f"🧹 Cleared old highlight: {old.name}")
        except Exception as e:
            print(f"⚠️ Could not delete {old.name}: {e}")

    # Track new snippets as they appear
    seen = set(p.name for p in SNIPPETS_DIR.glob("*.mp4"))
    result_holder = {"ok": False, "err": None, "csv_path": None}

    def _runner():
        """Run blocking detection + report in thread."""
        try:
            res_k3 = run_k3_detector(
                video=str(video_path),
                weights=str(YOLO_WEIGHTS),
                out_dir=str(OUT_DIR),
                device=DEVICE,
                snippets_dir=str(SNIPPETS_DIR),
                seconds_per_frame=2.0,
                emulate_live=False,
                save_graph=True,
            )
            result_holder["csv_path"] = res_k3.get("csv_path", str(CSV_PATH))
            # Build the final report (kept on disk)
            build_report(
                csv_path=result_holder["csv_path"],
                report_path=str(REPORT_PATH),
                video_path=str(video_path),
                graph_path=str(GRAPH_PATH),
            )
            result_holder["ok"] = True
        except Exception as e:
            result_holder["err"] = e

    # Start the blocking runner in a thread
    thr = threading.Thread(target=_runner, daemon=True)
    thr.start()

    goal_id = 0
        # --- LIVE PROGRESS LOOP ---
    last_elapsed = -1.0
    while thr.is_alive():
        elapsed = time.time() - start_ts

        # Emit periodic status with live elapsed time
        yield {
            "type": "status",
            "message": f"Processing… elapsed {elapsed:.1f}s"
        }

        # Detect and yield any new goal clips
        for p in sorted(SNIPPETS_DIR.glob("*.mp4")):
            if p.name not in seen:
                seen.add(p.name)
                goal_id += 1
                print(p)
                ensure_browser_playable(str(p))
                rel_path = f"k_highlights/{p.name}".replace("\\", "/")
                print(f"Yielding goal → {rel_path}")

                yield {
                    "type": "goal",
                    "goal_id": goal_id,
                    "clip_name": p.name,
                    "clip_rel_path": rel_path
                }

        await asyncio.sleep(1.0)


    # After thread finishes: final summary
    new_snippets = sorted(SNIPPETS_DIR.glob("*.mp4"))
    if not new_snippets:
        yield {"type": "status", "message": "No goal highlights were generated (pipeline ended)."}
    else:
        for p in new_snippets:
            if p.name not in seen:
                seen.add(p.name)
                goal_id += 1
                rel_path = f"outputs/k_highlights/{p.name}".replace("\\", "/")  # ✅ define here too
                print(f"🎯HIGHLIGHT FOUND → {rel_path}")
                yield {
                    "type": "goal",
                    "goal_id": goal_id,
                    "clip_name": p.name,
                    "clip_rel_path": rel_path
                }



    # Final completion signal
    if not result_holder["ok"]:
        err = result_holder["err"]
        yield {"type": "status", "message": "No goal highlights were generated (pipeline error)."}
        raise err if err else RuntimeError("Pipeline failed without error details")

    elapsed = time.time() - start_ts
    yield {"type": "done", "elapsed": elapsed, "outputs_dir": str(OUT_DIR)}

# -------------------------------------------------------------
# CLI entrypoint (optional standalone use)
# -------------------------------------------------------------
def _check_file(p: Path, what: str):
    if not p.exists():
        raise FileNotFoundError(f"{what} not found at: {p}")

def main():
    parser = argparse.ArgumentParser(description="Run K-Pipeline end-to-end.")
    parser.add_argument("--video", type=str, default=str(DEFAULT_INPUT), help="Path to input video")
    args = parser.parse_args()

    video_path = Path(args.video)
    _check_file(video_path, "Input video")
    _check_file(YOLO_WEIGHTS, "YOLO weights")

    log.info("\n============================")
    log.info("K-PIPELINE: FULL EXECUTION")
    log.info("============================\n")

    total_start = time.time()
    try:
        log.info("[Stage K3+K4] Detecting goals + creating highlights...\n")
        res_k3 = run_k3_detector(
            video=str(video_path),
            weights=str(YOLO_WEIGHTS),
            out_dir=str(OUT_DIR),
            device=DEVICE,
            snippets_dir=str(SNIPPETS_DIR),
            seconds_per_frame=2.0,
            emulate_live=False,
            save_graph=True,
        )

        log.info("[Stage K5] Generating report...\n")
        res_k5 = build_report(
            csv_path=str(res_k3.get("csv_path", CSV_PATH)),
            report_path=str(REPORT_PATH),
            video_path=str(video_path),
            graph_path=str(GRAPH_PATH),
        )

        log.info("K5 complete.")
        log.info(f"Final report: {res_k5['report_path']}")
        log.info(f"Final score: {res_k5['final_score']}")
        log.info(f"Average latency: {res_k5['avg_total_latency_s']} s")

    except Exception as e:
        log.error(f"Pipeline failed: {e}")
        sys.exit(1)

    total_time = time.time() - total_start
    h, m, s = int(total_time // 3600), int((total_time % 3600) // 60), int(total_time % 60)
    log.info(f"Total runtime: {h:02d}:{m:02d}:{s:02d}")
    log.info(f"Outputs saved in: {OUT_DIR}\n")

if __name__ == "__main__":
    main()
