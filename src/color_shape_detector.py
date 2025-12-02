# color_shape_detector.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np


# ----------------------------
# Configuración de colores HSV
# ----------------------------

# Rangos típicos en HSV para BGR->HSV en OpenCV
# Ajusta si la iluminación de tu entorno lo requiere.
COLOR_RANGES: Dict[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = {
    "red": [
        ((0, 80, 80), (10, 255, 255)),    # Rojo bajo
        ((170, 80, 80), (180, 255, 255))  # Rojo alto (wrap)
    ],
    "green": [
        ((35, 80, 80), (85, 255, 255))
    ],
    "blue": [
        ((90, 80, 80), (130, 255, 255))
    ],
    "yellow": [
        ((20, 80, 80), (35, 255, 255))
    ],
}

KERNEL = np.ones((5, 5), np.uint8)


@dataclass
class DetectedPattern:
    color: str
    shape: str
    label: str
    area: float
    center: Tuple[int, int]
    contour: np.ndarray


# ----------------------------
# Clasificación de formas
# ----------------------------

def _classify_shape(contour: np.ndarray) -> Optional[str]:
    """
    Clasifica la forma de un contorno en:
    - "triangle"
    - "square"
    - "circle"
    - "line"
    Devuelve None si no se reconoce.
    """
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return None

    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    vertices = len(approx)
    area = cv2.contourArea(contour)

    if area <= 0:
        return None

    x, y, w, h = cv2.boundingRect(approx)
    if w == 0 or h == 0:
        return None

    aspect_ratio = w / float(h)
    aspect_ratio = max(aspect_ratio, 1.0 / aspect_ratio)  # >= 1

    circularity = 4.0 * np.pi * area / (peri * peri)

    # Triángulo
    if vertices == 3:
        return "triangle"

    # Cuadrado / rectángulo / línea
    if vertices == 4:
        # Si es muy alargado -> línea
        if aspect_ratio > 4.0:
            return "line"
        else:
            return "square"

    # Formas con más vértices -> círculo o línea muy curva
    if vertices > 4:
        if circularity > 0.7:
            return "circle"
        if aspect_ratio > 4.0:
            return "line"

    return None


# ----------------------------
# Detección color + forma
# ----------------------------

def _build_color_mask(hsv: np.ndarray, color_name: str) -> np.ndarray:
    """
    Construye la máscara binaria para un color dado
    a partir de los rangos definidos en COLOR_RANGES.
    """
    ranges = COLOR_RANGES[color_name]
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for (low, high) in ranges:
        low_np = np.array(low, dtype=np.uint8)
        high_np = np.array(high, dtype=np.uint8)
        mask |= cv2.inRange(hsv, low_np, high_np)

    # Limpieza morfológica
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    return mask


def detect_color_shape(
    frame_bgr: np.ndarray,
    min_area: float = 1000.0
) -> Optional[DetectedPattern]:
    """
    Detecta el patrón dominante (color + forma) en la imagen.
    Devuelve un DetectedPattern o None si no se detecta nada fiable.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    # Suavizado para reducir ruido
    blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    candidates: List[DetectedPattern] = []

    for color_name in COLOR_RANGES.keys():
        mask = _build_color_mask(hsv, color_name)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            shape = _classify_shape(cnt)
            if shape is None:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            label = f"{color_name}_{shape}"
            candidates.append(
                DetectedPattern(
                    color=color_name,
                    shape=shape,
                    label=label,
                    area=area,
                    center=(cx, cy),
                    contour=cnt
                )
            )

    if not candidates:
        return None

    # Elegimos el patrón con mayor área (el principal en la escena)
    best = max(candidates, key=lambda p: p.area)
    return best


# ----------------------------
# Función auxiliar para dibujar
# ----------------------------

def draw_detected_pattern(
    frame_bgr: np.ndarray,
    pattern: Optional[DetectedPattern]
) -> np.ndarray:
    """
    Dibuja contorno, centro y etiqueta del patrón detectado.
    Si pattern es None, devuelve la imagen original.
    """
    if frame_bgr is None or pattern is None:
        return frame_bgr

    out = frame_bgr.copy()

    cv2.drawContours(out, [pattern.contour], -1, (0, 255, 0), 2)
    cx, cy = pattern.center
    cv2.circle(out, (cx, cy), 5, (255, 255, 255), -1)

    text = pattern.label
    cv2.putText(
        out,
        text,
        (cx - 60, cy - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        3,
        cv2.LINE_AA
    )
    cv2.putText(
        out,
        text,
        (cx - 60, cy - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return out
