# # io_utils.py
# """
# Centralized IO helpers:
#  - write_rows_xlsx_or_csv(path_no_ext, header, rows) -> returns saved path (prefers .xlsx)
#  - read_rows_xlsx_or_csv(path) -> (rows, header)
#  - save_snippet_from_caps(video_path, start_s, end_s, out_path) -> writes MP4 snippet via OpenCV
# """

# import os
# import csv
# from pathlib import Path
# from typing import List, Dict, Any, Tuple

# def write_rows_xlsx_or_csv(path_no_ext: str, header: List[str], rows: List[List[Any]]) -> str:
#     """
#     path_no_ext: "Outputs/GOALSSparsed/parsed_array" (without extension)
#     header: list of column names
#     rows: list of rows (list matching header)
#     Returns path written.
#     Prefers .xlsx if pandas+openpyxl available, else writes .csv
#     """
#     base = Path(path_no_ext)
#     base.parent.mkdir(parents=True, exist_ok=True)
#     try:
#         import pandas as pd
#         df = pd.DataFrame(rows, columns=header)
#         out_xlsx = base.with_suffix(".xlsx")
#         df.to_excel(str(out_xlsx), index=False)
#         return str(out_xlsx)
#     except Exception:
#         out_csv = base.with_suffix(".csv")
#         with open(out_csv, "w", newline="", encoding="utf-8") as f:
#             w = csv.writer(f)
#             w.writerow(header)
#             for r in rows:
#                 w.writerow(r)
#         return str(out_csv)

# def read_rows_xlsx_or_csv(path: str) -> Tuple[List[Dict[str,str]], List[str]]:
#     """
#     Read CSV or XLSX into list of dicts and header list.
#     """
#     p = Path(path)
#     if not p.exists():
#         raise FileNotFoundError(f"File not found: {path}")
#     if p.suffix.lower() in (".xlsx", ".xls"):
#         try:
#             import pandas as pd
#             df = pd.read_excel(str(p), engine='openpyxl' if 'openpyxl' in __import__('sys').modules else None)
#             df = df.fillna("")
#             rows = df.to_dict(orient="records")
#             header = list(df.columns)
#             return rows, header
#         except Exception:
#             # fallback to CSV read
#             pass
#     # CSV
#     with open(path, newline='', encoding='utf-8') as f:
#         reader = csv.DictReader(f)
#         header = reader.fieldnames or []
#         rows = [dict(r) for r in reader]
#     return rows, header

# def save_snippet_from_caps(video_path: str, start_s: float, end_s: float, out_path: str) -> None:
#     """
#     Save snippet [start_s, end_s] from video_path into out_path (mp4).
#     Uses OpenCV VideoCapture/VideoWriter. Overwrites if exists.
#     """
#     import cv2, math, os
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise RuntimeError(f"Unable to open video for snippet: {video_path}")
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fourcc = cv2.VideoWriter_fourcc(*'m','p','4','v')
#     out_writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
#     start_frame = int(math.floor(max(0.0, start_s) * fps))
#     end_frame = int(math.ceil(end_s * fps))
#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
#     for fno in range(start_frame, end_frame + 1):
#         ret, frame = cap.read()
#         if not ret:
#             break
#         out_writer.write(frame)
#     out_writer.release()
#     cap.release()
#     return
"""
Centralized IO helpers:
 - write_rows_xlsx_or_csv(path_no_ext, header, rows) -> returns saved path (prefers .xlsx)
 - read_rows_xlsx_or_csv(path) -> (rows, header)
 - save_snippet_from_caps(video_path, start_s, end_s, out_path) -> writes MP4 snippet via OpenCV
"""

import os
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple

def write_rows_xlsx_or_csv(path_no_ext: str, header: List[str], rows: List[List[Any]]) -> str:
    """
    path_no_ext: "Outputs/GOALSSparsed/parsed_array" (without extension)
    header: list of column names
    rows: list of rows (list matching header)
    Returns path written.
    Prefers .xlsx if pandas+openpyxl available, else writes .csv
    """
    base = Path(path_no_ext)
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
        df = pd.DataFrame(rows, columns=header)
        out_xlsx = base.with_suffix(".xlsx")
        df.to_excel(str(out_xlsx), index=False)
        return str(out_xlsx)
    except Exception:
        out_csv = base.with_suffix(".csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                w.writerow(r)
        return str(out_csv)

def read_rows_xlsx_or_csv(path: str) -> Tuple[List[Dict[str,str]], List[str]]:
    """
    Read CSV or XLSX into list of dicts and header list.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.suffix.lower() in (".xlsx", ".xls"):
        try:
            import pandas as pd
            # Make sure to have openpyxl installed: pip install openpyxl
            df = pd.read_excel(str(p))
            df = df.fillna("")
            # Convert all data to string to match the CSV reader's output format
            for col in df.columns:
                df[col] = df[col].astype(str)
            rows = df.to_dict(orient="records")
            header = list(df.columns)
            return rows, header
        except Exception as e:
            print(f"Failed to read Excel file {path}, falling back to CSV reader. Error: {e}")
            # Fallback to CSV read below

    # CSV reading logic
    rows = []
    header = []
    with open(path, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return [], [] # empty file
        for row in reader:
            rows.append(dict(zip(header, row)))
    return rows, header


def save_snippet_from_caps(video_path: str, start_s: float, end_s: float, out_path: str) -> None:
    """
    Save snippet [start_s, end_s] from video_path into out_path (mp4).
    Uses OpenCV VideoCapture/VideoWriter. Overwrites if exists.
    """
    import cv2
    import math
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for snippet: {video_path}")
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # More compatible codec
        out_writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        
        start_frame = int(math.floor(max(0.0, start_s) * fps))
        end_frame = int(math.ceil(end_s * fps))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for fno in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out_writer.write(frame)
    finally:
        if 'out_writer' in locals() and out_writer.isOpened():
            out_writer.release()
        if cap.isOpened():
            cap.release()
