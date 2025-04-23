#!/usr/bin/env python3
import os
import cv2
import numpy as np
from PIL import Image

def load_gray(path):
    """
    Load any image as a true single‑channel grayscale.
    If it's 3‑channel with identical channels, collapse to one.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load: {path}")

    # If it's 3‑channel but R=G=B, collapse to one
    if img.ndim == 3 and img.shape[2] == 3:
        b, g, r = cv2.split(img)
        if np.allclose(r, g) and np.allclose(r, b):
            img = r
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img

def enhance_gray_with_dolp(gray, dolp,
                           alpha=0.7,
                           blur_size=7,
                           clahe_clip=2.0,
                           clahe_tiles=(8,8)):
    """
    Enhance a single‑channel grayscale image using DoLP to drive CLAHE + brightness.
    Returns a single‑channel uint8 result in [0–255].

    Params:
    - alpha:     (0–1) how strongly high‑DoLP areas use CLAHE over raw gray
    - blur_size: kernel size for smoothing DoLP
    - clahe_clip:The clipLimit parameter for CLAHE
    - clahe_tiles:The tileGridSize parameter for CLAHE
    """
    # 1) normalize & smooth DoLP
    d = dolp.astype(np.float32)
    if d.max() > 1.0:
        d /= 255.0
    d = cv2.GaussianBlur(d, (blur_size, blur_size), 0)
    d = np.clip(d, 0.0, 1.0)

    # 2) CLAHE on the raw gray
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tiles)
    gray_eq = clahe.apply(gray)

    # 3) linear blend: raw→equalized based on DoLP
    out = (1.0 - alpha * d) * gray + (alpha * d) * gray_eq
    return np.clip(out, 0, 255).astype(np.uint8)

if __name__ == "__main__":

    ###### Change this to your own data ######
    # === USER CONFIG ===
    # base_dir: root folder containing rgb_xyz and dolp_xyz subdirs
    base_dir    = "/l/users/ahmed.aly/ipd/val/000008"
    # cam_folder: e.g. "cam1", "cam2", etc.
    cam_folder  = "cam3"

    for idx in range(6):

        ##### Parameters ####
        alpha=1.0  
        blur_size=7  
        clahe_clip=4.0  
        clahe_tiles=(4,4)  
        gamma=0.6  
        hist_lo=1  
        hist_hi=99

        # Build file paths (zero‑padded to 6 digits)
        gray_path = os.path.join(base_dir, f"rgb_{cam_folder}",  f"{idx:06d}.png")
        dolp_path = os.path.join(base_dir, f"dolp_{cam_folder}", f"{idx:06d}.png")

        # 1) Load inputs
        gray = load_gray(gray_path)
        dolp = cv2.imread(dolp_path, cv2.IMREAD_UNCHANGED)
        if dolp is None:
            raise FileNotFoundError(f"Could not load DoLP: {dolp_path}")

        # 2) DoLP‑driven enhancement
        enhanced = enhance_gray_with_dolp(
            gray, dolp,
            alpha=alpha,
            blur_size=blur_size,
            clahe_clip=clahe_clip,
            clahe_tiles=clahe_tiles
        )

        # 3) Gamma correction
        enh_f = enhanced.astype(np.float32) / 255.0
        enh_f = np.power(enh_f, gamma)
        enhanced = np.clip(enh_f * 255.0, 0, 255).astype(np.uint8)

        # 4) Histogram stretch
        lo, hi = np.percentile(enhanced, (hist_lo, hist_hi))
        stretched = (enhanced.astype(np.float32) - lo) * (255.0 / (hi - lo))
        enhanced = np.clip(stretched, 0, 255).astype(np.uint8)

        # 5) Prepare side‑by‑side comparison
        orig_rgb     = cv2.cvtColor(gray,     cv2.COLOR_GRAY2RGB)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        side_by_side = np.hstack((orig_rgb, enhanced_rgb))

        # 6) Save & display
        # out_combo = os.path.join(base_dir, "orig_vs_enhanced.png")
        # Image.fromarray(side_by_side).save(out_combo)
        # print(f"Saved side‑by‑side comparison to: {out_combo}")

        # Show on screen (scaled down)
        h, w, _ = side_by_side.shape
        disp = cv2.resize(side_by_side, (w//3, h//3))
        cv2.imshow("Original | Enhanced", cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
