# kalman_tracker.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2


@dataclass
class KalmanConfig:
    dt: float = 1.0
    process_noise: float = 1e-2
    measurement_noise: float = 1e-1
    error_cov_post: float = 1.0


class Kalman2DTracker:
    """
    Estado: [x, y, vx, vy]
    Medición: [x, y]
    """

    def __init__(self, cfg: KalmanConfig | None = None):
        self.cfg = cfg or KalmanConfig()
        self.kf = cv2.KalmanFilter(4, 2)
        self._init_filter()
        self.initialized = False

    def _init_filter(self):
        dt = self.cfg.dt

        # Matriz de transición
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1,  0],
             [0, 0, 0,  1]], dtype=np.float32
        )

        # Matriz de observación
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32
        )

        # Ruido del proceso
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * self.cfg.process_noise

        # Ruido de medición
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.cfg.measurement_noise

        # Covarianza posterior inicial
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * self.cfg.error_cov_post

    def reset(self):
        self._init_filter()
        self.initialized = False

    def initialize(self, x: float, y: float):
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.initialized = True

    def predict(self) -> Tuple[float, float]:
        pred = self.kf.predict()
        return float(pred[0]), float(pred[1])

    def correct(self, x: float, y: float) -> Tuple[float, float]:
        meas = np.array([[x], [y]], dtype=np.float32)
        est = self.kf.correct(meas)
        return float(est[0]), float(est[1])
