# color_shape_detector.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np



# Configuración de colores HSV

# Diccionario que define los rangos HSV para cada color a detectar.
COLOR_RANGES: Dict[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = {
    "red": [
        ((0, 80, 80), (10, 255, 255)),    # Rojo en el extremo bajo del canal H
        ((170, 80, 80), (180, 255, 255))  # Rojo en el extremo alto
    ],
    "green": [
        ((35, 80, 80), (85, 255, 255))    # Verde en HSV
    ],
    "blue": [
        ((90, 80, 80), (130, 255, 255))   # Azul en HSV
    ],
    "yellow": [
        ((20, 80, 80), (35, 255, 255))    # Amarillo en HSV
    ],
}

# Kernel estructurante usado para operaciones morfológicas (apertura y cierre)
KERNEL = np.ones((5, 5), np.uint8)


# Estructura de datos de salida
@dataclass
class DetectedPattern:
    # Color detectado (red, green, blue, yellow)
    color: str
    # Forma detectada (circle, triangle, square, line)
    shape: str
    # Etiqueta combinada color + forma (ej: "red_circle")
    label: str
    # Área del contorno detectado (en píxeles)
    area: float
    # Centroide del objeto detectado (cx, cy)
    center: Tuple[int, int]
    # Contorno completo del objeto
    contour: np.ndarray


# Clasificación de formas
def _classify_shape(contour: np.ndarray) -> Optional[str]:
    """
    Clasifica la forma geométrica de un contorno.
    Devuelve:
    - "triangle"
    - "square"
    - "circle"
    - "line"
    o None si la forma no es fiable.
    """

    # Cálculo del perímetro del contorno
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return None

    # Aproximación poligonal del contorno
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    vertices = len(approx)

    # Área del contorno
    area = cv2.contourArea(contour)
    if area <= 0:
        return None

    # Bounding box alineada con los ejes
    x, y, w, h = cv2.boundingRect(approx)
    if w == 0 or h == 0:
        return None

    # Relación de aspecto para distinguir entre líneas y cuadrados
    aspect_ratio = w / float(h)
    aspect_ratio = max(aspect_ratio, 1.0 / aspect_ratio)

    # Medida de circularidad (≈1 para círculos)
    circularity = 4.0 * np.pi * area / (peri * peri)

    # Clasificación según número de vértices y geometría

    # Triángulo
    if vertices == 3:
        return "triangle"

    # Cuadrado / rectángulo / línea
    if vertices == 4:
        # Si la relación de aspecto es muy grande, se considera una línea
        if aspect_ratio > 4.0:
            return "line"
        else:
            return "square"

    # Más de 4 vértices: círculo o línea curva
    if vertices > 4:
        # Alta circularidad indica círculo
        if circularity > 0.7:
            return "circle"
        # Muy alargado indica línea
        if aspect_ratio > 4.0:
            return "line"

    # Forma no reconocida
    return None


# Detección color + forma
def _build_color_mask(hsv: np.ndarray, color_name: str) -> np.ndarray:
    """
    Construye una máscara binaria para un color específico
    usando los rangos HSV definidos en COLOR_RANGES.
    """

    # Rangos HSV del color seleccionado
    ranges = COLOR_RANGES[color_name]

    # Inicializamos la máscara a negro
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    # Aplicamos cada rango HSV y acumulamos resultados
    for (low, high) in ranges:
        low_np = np.array(low, dtype=np.uint8)
        high_np = np.array(high, dtype=np.uint8)
        mask |= cv2.inRange(hsv, low_np, high_np)

    # Limpieza morfológica:
    # APERTURA elimina pequeños ruidos aislados
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=2)
    # CIERRE rellena pequeños huecos dentro del objeto
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)

    return mask


def detect_color_shape(
    frame_bgr: np.ndarray,
    min_area: float = 1000.0
) -> Optional[DetectedPattern]:
    """
    Detecta el patrón dominante (color + forma) en un frame.
    Devuelve un DetectedPattern si encuentra un objeto fiable,
    o None si no hay detecciones válidas.
    """

    # Comprobación de frame válido
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    # Suavizado para reducir ruido antes de segmentar
    blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)

    # Conversión de BGR a HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Lista de candidatos detectados
    candidates: List[DetectedPattern] = []

    # Iteramos sobre todos los colores definidos
    for color_name in COLOR_RANGES.keys():
        # Construimos la máscara binaria del color
        mask = _build_color_mask(hsv, color_name)

        # Buscamos contornos en la máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            # Filtrado por área mínima para eliminar ruido
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            # Clasificación de la forma
            shape = _classify_shape(cnt)
            if shape is None:
                continue

            # Cálculo del centroide usando momentos
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Etiqueta final color + forma
            label = f"{color_name}_{shape}"

            # Guardamos el patrón detectado
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

    # Si no hay candidatos, no se detecta ningún patrón
    if not candidates:
        return None

    # Seleccionamos el patrón con mayor área
    # (asumimos que es el objeto principal en escena)
    best = max(candidates, key=lambda p: p.area)
    return best


# Función auxiliar para dibujar
def draw_detected_pattern(
    frame_bgr: np.ndarray,
    pattern: Optional[DetectedPattern]
) -> np.ndarray:
    """
    Dibuja el contorno, centro y etiqueta del patrón detectado.
    Si pattern es None, devuelve la imagen original sin cambios.
    """

    if frame_bgr is None or pattern is None:
        return frame_bgr

    out = frame_bgr.copy()

    # Dibujo del contorno del objeto
    cv2.drawContours(out, [pattern.contour], -1, (0, 255, 0), 2)

    # Dibujo del centroide
    cx, cy = pattern.center
    cv2.circle(out, (cx, cy), 5, (255, 255, 255), -1)

    # Dibujo de la etiqueta (color + forma)
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