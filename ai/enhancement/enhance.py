"""Surveillance image enhancement algorithms for low-light, underexposed, and degraded footage."""
from __future__ import annotations

import cv2
import numpy as np


def clahe_enhance(image: np.ndarray, clip_limit: float = 2.5,
                  tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Enhance low-light surveillance frames using CLAHE in LAB color space.
    
    Applies adaptive histogram equalization only on the Lightness (L) channel,
    preventing unnatural color shifts or chromatic noise amplification.
    """
    if image is None or image.size == 0:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l = clahe.apply(l)
    merged = cv2.merge((enhanced_l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def gamma_correct(image: np.ndarray, gamma: float = 1.4) -> np.ndarray:
    """Apply non-linear power-law gamma correction to lift dark shadows."""
    if image is None or image.size == 0 or gamma <= 0:
        return image

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def unsharp_mask(image: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
    """Sharpen blurred surveillance frames using unsharp masking."""
    if image is None or image.size == 0:
        return image

    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def auto_enhance(image: np.ndarray, mode: str = "low_light") -> np.ndarray:
    """Automated enhancement pipeline for surveillance footage.
    
    Modes:
      'low_light': CLAHE + gamma correction (for night/shadow scenes)
      'sharpen': unsharp masking for deblurring
      'all': full low-light enhancement and edge sharpening
    """
    if mode == "low_light":
        return gamma_correct(clahe_enhance(image, clip_limit=2.5), gamma=1.3)
    elif mode == "sharpen":
        return unsharp_mask(image, sigma=1.0, strength=1.2)
    elif mode == "all":
        enhanced = gamma_correct(clahe_enhance(image, clip_limit=2.0), gamma=1.2)
        return unsharp_mask(enhanced, sigma=0.8, strength=1.0)
    return image
