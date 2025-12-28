# finger_detector.py
# (ahora es un detector genérico por color)

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np


@dataclass
class MarkerDetection:
    bbox: Tuple[int, int, int, int]          # x, y, w, h
    center: Tuple[int, int]                  # cx, cy
    contour: np.ndarray
    area: float


class ColorMarkerDetector:
    """
    Detector de un marcador/cinta de color (en HSV) para tracking.
    Soporta: red, green, blue, yellow.
    """

    # Rangos HSV típicos (OpenCV H: 0..180)
    COLOR_RANGES: Dict[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = {
        "red": [
            ((0, 90, 80), (10, 255, 255)),
            ((170, 90, 80), (180, 255, 255)),
        ],
        "green": [
            ((35, 80, 80), (85, 255, 255)),
        ],
        "blue": [
            ((90, 80, 80), (130, 255, 255)),
        ],
        "yellow": [
            ((20, 80, 80), (35, 255, 255)),
        ],
    }

    def __init__(
        self,
        color: str,
        min_area: float = 900.0,
        blur_ksize: int = 5,
        morph_ksize: int = 5,
        open_iters: int = 2,
        close_iters: int = 2,
    ):
        if color not in self.COLOR_RANGES:
            raise ValueError(f"Color '{color}' no soportado. Usa: {list(self.COLOR_RANGES.keys())}")

        self.color = color
        self.min_area = min_area
        self.blur_ksize = blur_ksize
        self.kernel = np.ones((morph_ksize, morph_ksize), np.uint8)
        self.open_iters = open_iters
        self.close_iters = close_iters

        self.ranges = self.COLOR_RANGES[color]

    def _build_mask(self, hsv: np.ndarray) -> np.ndarray:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (low, high) in self.ranges:
            low_np = np.array(low, dtype=np.uint8)
            high_np = np.array(high, dtype=np.uint8)
            mask |= cv2.inRange(hsv, low_np, high_np)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=self.open_iters)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=self.close_iters)
        return mask

    def detect(self, frame_bgr: np.ndarray) -> Tuple[Optional[MarkerDetection], np.ndarray]:
        """
        Devuelve:
          - MarkerDetection o None
          - mask (uint8) para debug
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None, np.zeros((1, 1), dtype=np.uint8)

        blurred = cv2.GaussianBlur(frame_bgr, (self.blur_ksize, self.blur_ksize), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = self._build_mask(hsv)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask

        best = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(best))
        if area < self.min_area:
            return None, mask

        x, y, w, h = cv2.boundingRect(best)
        cx = x + w // 2
        cy = y + h // 2

        det = MarkerDetection(
            bbox=(x, y, w, h),
            center=(cx, cy),
            contour=best,
            area=area
        )
        return det, mask
