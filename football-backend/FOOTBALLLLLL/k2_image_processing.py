"""
k2_image_processing.py (K-Pipeline)
-----------------------------------
Lightweight, stateless image enhancement pipeline for scoreboard OCR.
Optimized for speed, thread-safety, and consistent enhancement under varied lighting.
"""

import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# --- CONFIGURATION CONSTANTS ---
USE_CLAHE = True
TOPHAT_BLEND = 0.35
GAMMA = 1.05
MIN_SIDE = 160
AUTO_CONTRAST = True
SHARPEN_ENABLE = True
SHARPEN_BLUR_SIGMA = 1.5
SHARPEN_ORIG_WEIGHT = 1.60
SHARPEN_BLUR_WEIGHT = -0.60
SHARPEN_PASSES = 1
DENOISE_BEFORE = True
DENOISE_PARAMS = dict(d=9, sigmaColor=75, sigmaSpace=75)


# --- HELPER FUNCTIONS ---
def auto_contrast_rgb(bgr: np.ndarray, p_low=2, p_high=98) -> np.ndarray:
    """Percentile-based contrast stretch for each channel."""
    out = bgr.copy().astype(np.float32)
    for c in range(3):
        ch = out[..., c]
        lo, hi = np.percentile(ch, [p_low, p_high])
        if hi > lo:
            out[..., c] = np.clip((ch - lo) * (255.0 / (hi - lo)), 0, 255)
    return out.astype(np.uint8)


# --- CORE FUNCTION ---
def enhance_for_ocr(bgr: np.ndarray) -> np.ndarray:
    """
    Standardized enhancement pipeline for OCR:
    Upscale → CLAHE → Tophat → Gamma → Sharpen → Auto-contrast
    Thread-safe and stateless.
    """
    if bgr is None or bgr.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    # Step 1: Upscale very small crops
    h, w = bgr.shape[:2]
    if min(h, w) < MIN_SIDE and min(h, w) > 0:
        scale = MIN_SIDE / min(h, w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Step 2: CLAHE (local contrast)
    if USE_CLAHE:
        try:
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        except cv2.error:
            pass

    # Step 3: White Tophat blend (bright text enhancement)
    if TOPHAT_BLEND > 0:
        try:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.addWeighted(l, 1.0, th, TOPHAT_BLEND, 0)
            bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        except cv2.error:
            pass

    # Step 4: Gamma correction
    if GAMMA and GAMMA != 1.0:
        table = (np.linspace(0, 1, 256) ** (1.0 / GAMMA) * 255).astype(np.uint8)
        bgr = cv2.LUT(bgr, table)

    # Step 5: Unsharp mask for text definition
    if SHARPEN_ENABLE:
        work = bgr.copy()
        if DENOISE_BEFORE:
            work = cv2.bilateralFilter(work, DENOISE_PARAMS["d"],
                                       DENOISE_PARAMS["sigmaColor"],
                                       DENOISE_PARAMS["sigmaSpace"])
        for _ in range(SHARPEN_PASSES):
            blur = cv2.GaussianBlur(work, (0, 0), sigmaX=SHARPEN_BLUR_SIGMA)
            work = cv2.addWeighted(work, SHARPEN_ORIG_WEIGHT, blur, SHARPEN_BLUR_WEIGHT, 0)
        bgr = work

    # Step 6: Auto-contrast for readability
    if AUTO_CONTRAST:
        bgr = auto_contrast_rgb(bgr, 2, 98)

    return bgr


# --- TEST BLOCK ---
if __name__ == "__main__":
    import os
    from k1_frames_extractor import FrameStreamer

    demo_video = os.path.join("Inputs", "Original_video.mp4")
    out_dir = os.path.join("k_OUTPUTS", "k2_demo_outputs")
    os.makedirs(out_dir, exist_ok=True)

    log.info("[K2] Demo: Enhancing first few frames for OCR clarity...")

    try:
        with FrameStreamer(demo_video) as fs:
            for idx, ts, frame in fs.frames(seconds_per_frame=2.0, max_frames=3):
                enhanced = enhance_for_ocr(frame)
                out_path = os.path.join(out_dir, f"frame_{idx:03d}_enhanced.jpg")
                cv2.imwrite(out_path, enhanced)
                log.info(f"  Saved: {out_path}")
    except Exception as e:
        log.error(f"[K2] Demo failed: {e}")
