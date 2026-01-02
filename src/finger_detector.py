# finger_detector.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np


# Estructura de salida
@dataclass
class MarkerDetection:
    # Bounding box del marcador detectado: (x, y, width, height)
    bbox: Tuple[int, int, int, int]      

    # Centro del bounding box
    center: Tuple[int, int]              

    # Contorno completo del marcado
    contour: np.ndarray

    # Área del contorno en píxeles
    area: float


class ColorMarkerDetector:
    """
    Detector de un marcador o cinta de color usando segmentación en HSV.

    Este detector:
    - Segmenta un color concreto en el espacio HSV.
    - Limpia la máscara con morfología.
    - Extrae contornos y selecciona el más grande.
    - Devuelve su bounding box, centro y contorno para tracking.

    """

    # Rangos HSV para cada color
    COLOR_RANGES: Dict[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = {
        "red": [
            ((0, 90, 80), (10, 255, 255)),    # Rojo bajo
            ((170, 90, 80), (180, 255, 255)), # Rojo alto
        ],
        "green": [
            ((35, 80, 80), (85, 255, 255)),   # Verde 
        ],
        "blue": [
            ((90, 80, 80), (130, 255, 255)),  # Azul 
        ],
        "yellow": [
            ((20, 80, 80), (35, 255, 255)),   # Amarillo
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
        # Verificación de color soportado
        if color not in self.COLOR_RANGES:
            raise ValueError(f"Color '{color}' no soportado. Usa: {list(self.COLOR_RANGES.keys())}")

        # Color a detectar
        self.color = color

        # Área mínima para considerar una detección válida (evita ruido o pequeñas manchas)
        self.min_area = min_area

        # Tamaño del kernel del blur (reduce ruido antes de segmentar)
        self.blur_ksize = blur_ksize

        # Kernel morfológico usado en APERTURA y CIERRE
        self.kernel = np.ones((morph_ksize, morph_ksize), np.uint8)

        # Número de iteraciones morfológicas
        self.open_iters = open_iters
        self.close_iters = close_iters

        # Rangos HSV correspondientes al color seleccionado
        self.ranges = self.COLOR_RANGES[color]

    def _build_mask(self, hsv: np.ndarray) -> np.ndarray:
        """
        Construye la máscara binaria del color seleccionado a partir
        de la imagen HSV.
        """
        # Inicializa la máscara a negro
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        # Aplica todos los rangos HSV del color
        for (low, high) in self.ranges:
            low_np = np.array(low, dtype=np.uint8)
            high_np = np.array(high, dtype=np.uint8)
            mask |= cv2.inRange(hsv, low_np, high_np)

        # APERTURA elimina ruido aislado
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=self.open_iters)

        # CIERRE rellena huecos dentro del marcador
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=self.close_iters)

        return mask

    def detect(self, frame_bgr: np.ndarray) -> Tuple[Optional[MarkerDetection], np.ndarray]:
        """
        Detecta el marcador de color en un frame.

        Devuelve:
          - MarkerDetection si se detecta un marcador válido
          - None si no hay detección fiable
          - mask (uint8) para debug/visualización
        """

        # Validación del frame
        if frame_bgr is None or frame_bgr.size == 0:
            return None, np.zeros((1, 1), dtype=np.uint8)

        # Suavizado previo para reducir ruido
        blurred = cv2.GaussianBlur(frame_bgr, (self.blur_ksize, self.blur_ksize), 0)

        # Conversión a HSV para segmentación por color
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Construcción de la máscara binaria del color
        mask = self._build_mask(hsv)

        # Extracción de contornos en la máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask

        # Selecciona el contorno con mayor área (se asume que es el marcador principal)
        best = max(contours, key=cv2.contourArea)

        # Cálculo del área del contorno
        area = float(cv2.contourArea(best))
        if area < self.min_area:
            # Contorno demasiado pequeño asumimos que es ruido
            return None, mask

        # Bounding box del contorno
        x, y, w, h = cv2.boundingRect(best)

        # Centro del bounding box
        cx = x + w // 2
        cy = y + h // 2

        # Construcción del resultado final
        det = MarkerDetection(
            bbox=(x, y, w, h),
            center=(cx, cy),
            contour=best,
            area=area
        )

        return det, mask
