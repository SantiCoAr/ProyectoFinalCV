# password_decoder.py

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional
import time


class DecoderState(Enum):
    WAIT_STABLE = auto()   # esperando un símbolo estable para aceptar
    WAIT_RELEASE = auto()  # ya aceptó uno; espera a que se retire (None) para permitir el siguiente


@dataclass
class DecoderConfig:
    stable_frames_required: int = 15   # cuántos frames iguales para considerar estable
    release_frames_required: int = 10   # cuántos frames None para considerar "se retiró"
    cooldown_seconds: float = 1.5    # tiempo mínimo entre aceptaciones (extra seguridad)


class PasswordDecoder:
    """
    Decodificador automático:
    - Acepta un símbolo cuando es estable N frames.
    - Después exige "release" (detectar None M frames) para evitar contar repetido.
    - Al llegar a 4 símbolos, valida contra la contraseña y resetea el intento.
    """

    def __init__(self, password: List[str], config: DecoderConfig | None = None):
        self.password = password
        self.cfg = config or DecoderConfig()
        self.reset()

    def reset(self) -> None:
        self.state = DecoderState.WAIT_STABLE
        self.entered: List[str] = []

        self._stable_label: Optional[str] = None
        self._stable_count: int = 0

        self._none_count: int = 0
        self._last_accept_time: float = 0.0

        self.last_result: Optional[bool] = None   # True / False cuando termina una secuencia
        self.last_result_time: float = 0.0

    def _update_stability(self, label: Optional[str]) -> None:
        if label is None:
            self._stable_label = None
            self._stable_count = 0
            return

        if label == self._stable_label:
            self._stable_count += 1
        else:
            self._stable_label = label
            self._stable_count = 1

    def _update_release(self, label: Optional[str]) -> None:
        if label is None:
            self._none_count += 1
        else:
            self._none_count = 0

    def can_accept_now(self) -> bool:
        now = time.time()
        return (now - self._last_accept_time) >= self.cfg.cooldown_seconds

    def update(self, detected_label: Optional[str]) -> None:
        """
        Llamar en cada frame con el label detectado (o None).
        Maneja internamente el progreso y decide cuándo aceptar.
        """
        now = time.time()

        # Limpia resultado mostrado pasado un rato
        if self.last_result is not None and (now - self.last_result_time) > 2.0:
            self.last_result = None

        if self.state == DecoderState.WAIT_STABLE:
            self._update_stability(detected_label)

            # Condición de aceptación: estable N frames y cooldown cumplido
            if (
                self._stable_label is not None
                and self._stable_count >= self.cfg.stable_frames_required
                and self.can_accept_now()
            ):
                # Aceptar símbolo
                self.entered.append(self._stable_label)
                self._last_accept_time = now

                # Pasar a estado de release para no contar repetido
                self.state = DecoderState.WAIT_RELEASE
                self._none_count = 0

                # Reset estabilidad para el próximo símbolo
                self._stable_label = None
                self._stable_count = 0

                # Si ya hay 4, validar
                if len(self.entered) >= len(self.password):
                    ok = (self.entered == self.password)
                    self.last_result = ok
                    self.last_result_time = now
                    # Resetea intento automáticamente tras evaluar
                    self.entered = []
                    self.state = DecoderState.WAIT_STABLE

        elif self.state == DecoderState.WAIT_RELEASE:
            # Espera a que el usuario quite el patrón (varios frames None)
            self._update_release(detected_label)
            if self._none_count >= self.cfg.release_frames_required:
                self.state = DecoderState.WAIT_STABLE
                self._none_count = 0

    # Helpers para UI
    def progress(self) -> int:
        return len(self.entered)

    def progress_str(self) -> str:
        return " - ".join(self.entered) if self.entered else "(vacio)"

    def stability_meter(self) -> str:
        # para mostrar cuánto falta para “aceptar” el actual
        if self.state != DecoderState.WAIT_STABLE or self._stable_label is None:
            return "-"
        return f"{self._stable_count}/{self.cfg.stable_frames_required} ({self._stable_label})"

    def state_str(self) -> str:
        if self.state == DecoderState.WAIT_STABLE:
            return "WAIT_STABLE"
        return f"WAIT_RELEASE {self._none_count}/{self.cfg.release_frames_required}"
