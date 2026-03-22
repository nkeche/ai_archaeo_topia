import logging
from pathlib import Path
import numpy as np
import cv2
from osgeo import gdal

logger = logging.getLogger(__name__)


def read_image_gray_any(image_path: str | Path) -> np.ndarray:
    """Read an image as grayscale using OpenCV first, then GDAL fallback.

    Returns:
        2D uint8 grayscale image.

    Raises:
        ValueError: If the image cannot be read by any backend.
    """
    image_path = str(image_path)

    # 1) Standard OpenCV path
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None and img.size > 0:
            return img
    except Exception as exc:
        logger.warning("OpenCV grayscale read failed for %s: %s", image_path, exc)

    # 2) OpenCV + GDAL path
    try:
        img = cv2.imread(image_path, cv2.IMREAD_LOAD_GDAL | cv2.IMREAD_GRAYSCALE)
        if img is not None and img.size > 0:
            return img
    except Exception as exc:
        logger.warning("OpenCV GDAL grayscale read failed for %s: %s", image_path, exc)

    # 3) Native GDAL path
    try:
        ds = gdal.Open(image_path, gdal.GA_ReadOnly)
        if ds is None:
            raise ValueError("gdal.Open returned None")

        arr = ds.ReadAsArray()
        if arr is None:
            raise ValueError("ReadAsArray returned None")

        # GDAL returns:
        # - 2D array for single-band
        # - 3D array shaped (bands, rows, cols) for multi-band
        if arr.ndim == 2:
            gray = arr
        elif arr.ndim == 3:
            if arr.shape[0] >= 3:
                rgb = np.transpose(arr[:3], (1, 2, 0))
                if rgb.dtype != np.uint8:
                    rgb = normalize_to_uint8(rgb)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            else:
                gray = arr[0]
        else:
            raise ValueError(f"Unsupported GDAL array shape: {arr.shape}")

        if gray.dtype != np.uint8:
            gray = normalize_to_uint8(gray)

        return gray

    except Exception as exc:
        raise ValueError(f"Could not read image with OpenCV or GDAL: {exc}") from exc


def read_image_color_any(image_path: str | Path) -> np.ndarray:
    """Read an image as RGB using OpenCV first, then GDAL fallback.

    Returns:
        3D uint8 RGB image.

    Raises:
        ValueError: If the image cannot be read by any backend.
    """
    image_path = str(image_path)

    # 1) Standard OpenCV path
    try:
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is not None and bgr.size > 0:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        logger.warning("OpenCV color read failed for %s: %s", image_path, exc)

    # 2) OpenCV + GDAL path
    try:
        bgr = cv2.imread(image_path, cv2.IMREAD_LOAD_GDAL | cv2.IMREAD_COLOR)
        if bgr is not None and bgr.size > 0:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        logger.warning("OpenCV GDAL color read failed for %s: %s", image_path, exc)

    # 3) Native GDAL path
    try:
        ds = gdal.Open(image_path, gdal.GA_ReadOnly)
        if ds is None:
            raise ValueError("gdal.Open returned None")

        arr = ds.ReadAsArray()
        if arr is None:
            raise ValueError("ReadAsArray returned None")

        if arr.ndim == 2:
            gray = normalize_to_uint8(arr) if arr.dtype != np.uint8 else arr
            return np.stack([gray, gray, gray], axis=-1)

        if arr.ndim == 3:
            bands, _, _ = arr.shape
            if bands >= 3:
                rgb = np.transpose(arr[:3], (1, 2, 0))
                if rgb.dtype != np.uint8:
                    rgb = normalize_to_uint8(rgb)
                return rgb
            gray = arr[0]
            if gray.dtype != np.uint8:
                gray = normalize_to_uint8(gray)
            return np.stack([gray, gray, gray], axis=-1)

        raise ValueError(f"Unsupported GDAL array shape: {arr.shape}")

    except Exception as exc:
        raise ValueError(f"Could not read image with OpenCV or GDAL: {exc}") from exc


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize numeric raster data to uint8 for CV processing."""
    arr = np.asarray(arr)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    min_val = float(arr[finite_mask].min())
    max_val = float(arr[finite_mask].max())

    if max_val <= min_val:
        return np.zeros(arr.shape, dtype=np.uint8)

    scaled = (arr.astype(np.float32) - min_val) / (max_val - min_val)
    scaled = np.clip(scaled * 255.0, 0, 255)
    return scaled.astype(np.uint8)