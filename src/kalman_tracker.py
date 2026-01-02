# kalman_tracker.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2



# Configuración del filtro Kalman
@dataclass
class KalmanConfig:
    # Paso temporal del modelo (se asume 1 frame = 1 unidad de tiempo)
    dt: float = 1.0

    # Ruido del proceso:
    # indica cuánto puede desviarse el movimiento real del modelo ideal
    process_noise: float = 1e-2

    # Ruido de medición:
    # indica cuánta confianza se tiene en la medición del detector
    measurement_noise: float = 1e-1

    # Covarianza inicial del error del estado
    error_cov_post: float = 1.0


class Kalman2DTracker:
    """
    Filtro de Kalman 2D para tracking de un punto.

    Estado interno del filtro:
        [x, y, vx, vy]
        x, y  -> posición estimada
        vx, vy -> velocidad estimada

    Medición externa:
        [x, y]
        (solo se mide posición, la velocidad se infiere)
    """

    def __init__(self, cfg: KalmanConfig | None = None):
        # Configuración del filtro (parámetros de ruido, dt, etc.)
        self.cfg = cfg or KalmanConfig()

        # Filtro Kalman de OpenCV:
        # 4 variables de estado, 2 variables de medición
        self.kf = cv2.KalmanFilter(4, 2)

        # Inicializa todas las matrices del filtro
        self._init_filter()

        # Indica si el filtro ya fue inicializado con una medición real
        self.initialized = False

    def _init_filter(self):
        # Paso temporal del modelo
        dt = self.cfg.dt

        # Matriz de transición de estados
        # Se asume movimiento a velocidad constante
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1,  0],
             [0, 0, 0,  1]], dtype=np.float32
        )

        # Matriz de observación (medición)
        # Indica qué partes del estado se miden directamente:
        # se mide x e y, pero no las velocidades
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32
        )


        # Covarianza del ruido del proceso
        # Modela la incertidumbre del modelo dinámico
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * self.cfg.process_noise

        # Covarianza del ruido de medición
        # Modela el ruido del detector (mediciones imprecisas)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.cfg.measurement_noise

        # Covarianza inicial del error posterior
        # Representa la incertidumbre inicial del estado estimado
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * self.cfg.error_cov_post

    def reset(self):
        # Reinicia completamente el filtro Kalman
        # Se pierde toda la información previa
        self._init_filter()
        self.initialized = False

    def initialize(self, x: float, y: float):
        # Inicializa el estado del filtro con una medición real
        # Se asume velocidad inicial nula
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.initialized = True

    def predict(self) -> Tuple[float, float]:
        # Predicción del siguiente estado usando el modelo dinámico
        # Se usa cuando no hay medición disponible
        pred = self.kf.predict()
        return float(pred[0]), float(pred[1])

    def correct(self, x: float, y: float) -> Tuple[float, float]:
        # Corrección del estado usando una medición real (x, y)
        # Combina la predicción del modelo con la observación
        meas = np.array([[x], [y]], dtype=np.float32)
        est = self.kf.correct(meas)
        return float(est[0]), float(est[1])
 