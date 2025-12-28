# snake_game.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import random
import math


@dataclass
class SnakeConfig:
    max_length: int = 80
    segment_step: float = 12.0      # distancia mínima entre segmentos
    eat_radius: float = 25.0        # si la cabeza se acerca a la comida, la “come”
    food_margin: int = 60           # margen para no aparecer pegado al borde


class SnakeGame:
    def __init__(self, width: int, height: int, cfg: SnakeConfig | None = None):
        self.width = width
        self.height = height
        self.cfg = cfg or SnakeConfig()
        self.reset()

    def reset(self):
        self.points: List[Tuple[float, float]] = []
        self.score = 0
        self._target_length = 12  # longitud inicial
        self.food = self._spawn_food()

    def _spawn_food(self) -> Tuple[int, int]:
        m = self.cfg.food_margin
        x = random.randint(m, max(m+1, self.width - m))
        y = random.randint(m, max(m+1, self.height - m))
        return (x, y)

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def update(self, head_xy: Tuple[float, float]):
        """
        head_xy: (x,y) de la cabeza (controlada por el dedo)
        """
        hx, hy = head_xy

        # Añadir punto si está suficientemente lejos del último (para suavizar)
        if not self.points:
            self.points.append((hx, hy))
        else:
            if self._dist(self.points[-1], (hx, hy)) >= self.cfg.segment_step:
                self.points.append((hx, hy))

        # Mantener longitud objetivo
        while len(self.points) > self._target_length:
            self.points.pop(0)

        # Comer comida
        if self._dist((hx, hy), self.food) <= self.cfg.eat_radius:
            self.score += 1
            self._target_length = min(self.cfg.max_length, self._target_length + 6)
            self.food = self._spawn_food()

    def get_segments(self) -> List[Tuple[int, int]]:
        return [(int(x), int(y)) for (x, y) in self.points]

    def get_head(self) -> Tuple[int, int]:
        if not self.points:
            return (0, 0)
        return (int(self.points[-1][0]), int(self.points[-1][1]))
