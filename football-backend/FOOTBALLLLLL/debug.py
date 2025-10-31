"""
generate_debug_images.py

A standalone utility to extract and save pre-processed scoreboard images
for debugging OCR performance without running the full r4/r5 pipeline.

This script will:
1. Stream frames from a video at a specified interval.
2. Use YOLO to detect the scoreboard's Region of Interest (ROI).
3. Apply standard Idefics2 image preprocessing.
4. Run OCR on the enhanced image.
5. Parse team names and scores.
6. Save the image with OCR bounding boxes and the full parsed result drawn on it.

This helps you quickly see exactly what the OCR engine is "seeing" and how it's being parsed.

Usage (from your terminal):
> python generate_debug_images.py
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional
import cv2
from PIL import Image
import numpy as np
import easyocr

# Ensure local imports from the pipeline work correctly
sys.path.insert(0, os.path.dirname(__file__) or ".")

# --- Import necessary components from your existing pipeline ---
try:
    from r1_frames_extractor import stream_frames
    from r3_roi_detector import prep_for_idefics
    # MODIFIED: Import more helpers for full parsing
    from r4_parsed_singlecsv import (
        detect_best_roi_in_frame,
        extract_between_team_scores,
        centroid_x_from_bbox,
        FuzzyMapper
    )
    from ultralytics import YOLO
except ImportError as e:
    print(f"FATAL: Could not import a required pipeline module. Make sure this script is in the same directory as r1, r3, and r4 files.")
    print(f"Error details: {e}")
    sys.exit(1)


def generate_images(
    video: str,
    output_dir: str,
    weights: str,
    start_ts: float = 0.0,
    steps: float = 1.0 / 60.0,
    max_frames: Optional[int] = 100,
    device: Optional[str] = None
):
    """
    Connects to a video, detects ROIs, preprocesses, runs OCR, parses, and saves annotated images.
    """
    outd = Path(output_dir)
    outd.mkdir(parents=True, exist_ok=True)
    print(f"Saving debug images to: {outd.resolve()}")

    if not os.path.exists(weights):
        print(f"Error: YOLO weights not found at '{weights}'")
        return

    # --- Setup ---
    print("Loading YOLO model...")
    model = YOLO(weights)
    print("Loading OCR model (easyocr)...")
    reader = easyocr.Reader(['en'], gpu=(device is not None and 'cpu' not in device))
    # MODIFIED: Instantiate the FuzzyMapper for team name normalization
    mapper = FuzzyMapper()
    
    if device:
        try:
            model.to(device)
            print(f"Models moved to device: {device}")
        except Exception as e:
            print(f"Warning: Could not move model to device '{device}'. Error: {e}")

    print("Starting frame processing...")
    frames_gen = stream_frames(video, start_ts=start_ts, steps=steps)
    if frames_gen is None:
        print("Error: Could not start frame streamer.")
        return

    processed_count = 0
    saved_count = 0

    # --- Main Loop ---
    try:
        for idx, ts, frame in frames_gen:
            if max_frames is not None and processed_count >= max_frames:
                print(f"Reached max_frames limit of {max_frames}.")
                break
            
            processed_count += 1
            print(f"Processing frame {idx} at timestamp {ts:.2f}s...")

            # 1. Detect the scoreboard ROI
            crop, bbox = detect_best_roi_in_frame(frame, model, device)
            
            if crop.size == 0:
                print(f"  - No scoreboard detected for frame {idx}.")
                continue

            # 2. Apply Idefics2 preprocessing
            preprocessed_bgr = prep_for_idefics(crop)

            # 3. Run OCR on the preprocessed image
            raw_ocr_results = reader.readtext(preprocessed_bgr)
            
            # 4. Parse full results (teams and scores)
            struct_sorted = sorted(
                [{"text": text, "bbox": bbox, "cx": centroid_x_from_bbox(bbox)} for bbox, text, conf in raw_ocr_results],
                key=lambda x: x["cx"]
            )
            parsed_result = extract_between_team_scores(struct_sorted)
            
            # Get raw names and scores
            lt_raw = parsed_result.get("left_team_raw", "")
            rt_raw = parsed_result.get("right_team_raw", "")
            ls_raw = parsed_result.get("left_score_raw", "-")
            rs_raw = parsed_result.get("right_score_raw", "-")

            # Get canonical team names using the mapper
            lt_can = mapper.map(lt_raw)
            rt_can = mapper.map(rt_raw)
            
            print(f"  - OCR Parsed: {lt_can or 'N/A'} {ls_raw} - {rs_raw} {rt_can or 'N/A'}")

            # 5. Create annotated image for saving
            annotated_img = preprocessed_bgr.copy()

            # Draw OCR bounding boxes on the image
            for result in raw_ocr_results:
                bbox, text, conf = result
                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))
                cv2.rectangle(annotated_img, top_left, bottom_right, (0, 255, 0), 1)

            # Add the final parsed score text at the bottom
            score_text = f"PARSED: {lt_can or 'N/A'} {ls_raw} - {rs_raw} {rt_can or 'N/A'}"
            h, w, _ = annotated_img.shape
            cv2.rectangle(annotated_img, (0, h-20), (w, h), (0,0,0), -1) # Black bar
            cv2.putText(annotated_img, score_text, (5, h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # 6. Convert the annotated image to RGB and save
            annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            try:
                img_path = outd / f"debug_frame_{idx:05d}_ts_{ts:.2f}s.jpg"
                Image.fromarray(annotated_rgb).save(img_path, quality=95)
                saved_count += 1
                print(f"  + Saved ANNOTATED ROI to {img_path.name}")
            except Exception as e:
                print(f"  - ERROR: Could not save image for frame {idx}. Details: {e}")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    finally:
        if 'frames_gen' in locals() and frames_gen:
            frames_gen.close()
    
    print("-" * 50)
    print("Debug image generation complete.")
    print(f"Total frames processed: {processed_count}")
    print(f"Total images saved: {saved_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate annotated scoreboard images for OCR debugging.")
    parser.add_argument("--video", type=str, default=os.path.join("Inputs", "Original_video.mp4"), help="Path to the input video file.")
    parser.add_argument("--output", type=str, default=os.path.join("Outputs", "debug_roi"), help="Directory to save the output images.")
    parser.add_argument("--weights", type=str, default=os.path.join("Models", "Yolo_Scoreboard_best.pt"), help="Path to the YOLO model weights.")
    parser.add_argument("--max_frames", type=int, default=100, help="Maximum number of frames to process and save.")
    parser.add_argument("--steps", type=float, default=60.0, help="Seconds between each frame to process. Default: 60 (1 frame per minute).")
    parser.add_argument("--start_ts", type=float, default=0.0, help="Start time in seconds in the video.")
    parser.add_argument("--device", type=str, default=None, help="Computation device, e.g., 'cpu' or 'cuda:0'.")
    
    args = parser.parse_args()

    generate_images(
        video=args.video,
        output_dir=args.output,
        weights=args.weights,
        start_ts=args.start_ts,
        steps=1.0/args.steps, # The function expects steps as frames-per-second
        max_frames=args.max_frames,
        device=args.device
    )

