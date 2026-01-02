# snake_game.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import random
import math


@dataclass
class SnakeConfig:
    # Parámetros principales del juego
    max_length: int = 80
    segment_step: float = 12.0      # distancia mínima entre puntos para “añadir” un nuevo segmento (suaviza el cuerpo)
    eat_radius: float = 25.0        # radio de “captura” para considerar que la comida fue comida
    food_margin: int = 60           # margen para que la comida no aparezca pegada al borde


class SnakeGame:
    def __init__(self, width: int, height: int, cfg: SnakeConfig | None = None):
        # Dimensiones del tablero (en píxeles)
        self.width = width
        self.height = height
        # Configuración del snake (si no se pasa, usa defaults)
        self.cfg = cfg or SnakeConfig()
        # Estado inicial
        self.reset()

    def reset(self):
        # Lista de puntos (x,y) que representan el cuerpo (la cabeza es el último)
        self.points: List[Tuple[float, float]] = []
        # Puntuación del jugador
        self.score = 0
        # Longitud objetivo del cuerpo (se recorta la cola para mantenerla)
        self._target_length = 8  # longitud inicial
        # Generar la primera comida
        self.food = self._spawn_food()

    def _spawn_food(self) -> Tuple[int, int]:
        # Genera comida aleatoria dentro del tablero respetando un margen
        m = self.cfg.food_margin
        x = random.randint(m, max(m+1, self.width - m))
        y = random.randint(m, max(m+1, self.height - m))
        return (x, y)

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        # Distancia entre dos puntos
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def update(self, head_xy: Tuple[float, float]):
        """
        head_xy: (x,y) de la cabeza (controlada por el dedo)
        """
        hx, hy = head_xy

        # Construcción del cuerpo: solo añade un nuevo punto si la cabeza se movió lo suficiente.
        if not self.points:
            self.points.append((hx, hy))
        else:
            if self._dist(self.points[-1], (hx, hy)) >= self.cfg.segment_step:
                self.points.append((hx, hy))

        # Mantener longitud: si hay más puntos que la longitud objetivo, se recorta la cola.
        while len(self.points) > self._target_length:
            self.points.pop(0)

        # Comer comida: si la cabeza entra en un radio, aumenta score, crece y respawnea comida.
        if self._dist((hx, hy), self.food) <= self.cfg.eat_radius:
            self.score += 1
            self._target_length = min(self.cfg.max_length, self._target_length + 8)
            self.food = self._spawn_food()

    def get_segments(self) -> List[Tuple[int, int]]:
        # Devuelve el cuerpo como enteros para dibujar
        return [(int(x), int(y)) for (x, y) in self.points]

    def get_head(self) -> Tuple[int, int]:
        # Devuelve la cabeza (último punto). Si no hay puntos, devuelve (0,0).
        if not self.points:
            return (0, 0)
        return (int(self.points[-1][0]), int(self.points[-1][1]))
