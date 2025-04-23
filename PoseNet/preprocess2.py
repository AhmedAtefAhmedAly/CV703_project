import numpy as np
import cv2

def enhance_rgb_with_polarization(rgb, dolp, aolp, alpha=0.5, beta=0.5):
    """
    Enhance the RGB image by using DoLP and AoLP.

    Parameters:
    - rgb: RGB image (H, W, 3) in [0, 255]
    - dolp: Degree of Linear Polarization image (H, W) in [0, 1]
    - aolp: Angle of Linear Polarization image (H, W) in [0, 1] or radians
    - alpha: weight for DoLP-based enhancement
    - beta: weight for AoLP-based enhancement

    Returns:
    - enhanced_rgb: enhanced RGB image
    """

    # Normalize RGB to [0, 1]
    rgb_norm = rgb.astype(np.float32) / 255.0

    # Expand DoLP to 3 channels
    dolp_3c = np.stack([dolp]*3, axis=-1)

    # Normalize AoLP to [0, 1] if in radians
    if aolp.max() > 1.0:
        aolp = aolp / np.pi  # Map [0, π] → [0, 1]

    # Create AoLP color map (optional but helps visual enhancement)
    aolp_colored = cv2.applyColorMap((aolp*255).astype(np.uint8), cv2.COLORMAP_HSV)
    aolp_colored = aolp_colored.astype(np.float32) / 255.0

    # Weighted fusion
    enhanced = rgb_norm + alpha * dolp_3c + beta * aolp_colored
    enhanced = np.clip(enhanced, 0, 1)

    # Convert back to [0, 255]
    enhanced_rgb = (enhanced * 255).astype(np.uint8)
    return enhanced_rgb



def enhance_rgb_with_dolp(rgb, dolp, k=0.35, blur_size=5):
    """
    Enhance an RGB image by boosting its brightness based on the DoLP map.

    Parameters:
    - rgb:    uint8 RGB image, shape (H, W, 3), values in [0,255]
    - dolp:   float DoLP map, shape (H, W), normalized to [0,1]
    - k:      float, strength of the boost (e.g. 0.3–1.0)
    - blur_size: int, kernel size for smoothing DoLP to avoid noise artifacts.

    Returns:
    - enhanced_rgb: uint8 RGB image, same shape as input
    """

    # 1. Smooth DoLP to reduce noise
    dolp_smooth = cv2.GaussianBlur(dolp.astype(np.float32), (blur_size, blur_size), 0)

    # 2. Convert RGB → HSV
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

    # 3. Boost V channel by DoLP
    #    hsv[...,2] is the V channel in [0,255]
    hsv[..., 2] += k * dolp_smooth * 255
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)

    # 4. Back to RGB
    enhanced_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return enhanced_rgb

# === Example usage ===
if __name__ == "__main__":
    num = 5
    rgb = cv2.imread(f"/l/users/bahey.tharwat/bpc_baseline_old/datasets/ipd_val/val/000008/rgb_cam3/00000{num}.png")  # BGR format
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    dolp = cv2.imread(f"/l/users/bahey.tharwat/bpc_baseline_old/datasets/ipd_val/val/000008/dolp_cam3/00000{num}.png", cv2.IMREAD_GRAYSCALE) / 255.0
    aolp = cv2.imread(f"/l/users/bahey.tharwat/bpc_baseline_old/datasets/ipd_val/val/000008/aolp_cam3/00000{num}.png", cv2.IMREAD_GRAYSCALE) / 255.0

    # enhanced = enhance_rgb_with_polarization(rgb, dolp, aolp, alpha=0.8, beta=0.5)
    enhanced = enhance_rgb_with_dolp(rgb, dolp)

    print(dolp.shape)
    enhanced = cv2.resize(dolp, None, fx=0.5, fy=0.5)
    # print(enhanced.shape)

    # Show or save
    # cv2.imshow("Enhanced", cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
    cv2.imshow("Enhanced", enhanced)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
