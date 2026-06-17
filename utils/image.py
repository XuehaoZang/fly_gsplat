from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import cv2

def crop_image(im: np.ndarray, cx: float, cy: float, crop_size: int = 160) -> Tuple[np.ndarray, int, int]:
    """
    Crop image around a given centre (cx, cy), e.g. mask centroid.
    Returns the cropped image and the crop window's top-left corner (x_min, y_min),
    which the caller uses to update camera intrinsics via CameraConfig.apply_crop.
    """
    h_orig, w_orig = im.shape[:2]

    # 1. Calculate top-left corner of the crop window, centred on (cx, cy)
    half_size = crop_size // 2
    x_min = int(round(cx)) - half_size
    y_min = int(round(cy)) - half_size

    # 2. Constrain window within image boundaries
    x_min = max(0, min(x_min, w_orig - crop_size))
    y_min = max(0, min(y_min, h_orig - crop_size))

    # 3. Execute crop
    cropped_im = im[y_min:y_min + crop_size, x_min:x_min + crop_size]

    return cropped_im, x_min, y_min

def gray_to_rgba(gray_path: Path, rgba_path: Path) -> bool:
    """grayscale -> RGBA PNG"""
    img = cv2.imread(str(gray_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    # RGB = 255, Alpha = 255
    rgba[mask > 0, 0:3] = 255
    rgba[mask > 0, 3] = 255
    
    cv2.imwrite(str(rgba_path), rgba)
    return True

def binarize_mask(im: np.ndarray, threshold: int = 1) -> np.ndarray:
    """
    Convert input image to a binary mask, isolating target areas.
    """
    # 1. Convert to grayscale if image is colored (H, W, 3)
    if len(im.shape) == 3:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        gray = im
        
    # 2. Thresholding: pixels > threshold become 255 (white)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    return binary

def dilate_mask(im: np.ndarray, kernel_size: int = 3, iterations: int = 2) -> np.ndarray:
    """
    Apply morphological dilation to expand the binary mask boundaries.
    """
    # 1. Generate elliptical kernel for smoother, natural edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 2. Execute dilation
    dilated = cv2.dilate(im, kernel, iterations=iterations)
    
    return dilated
