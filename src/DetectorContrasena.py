import time

SECUENCIA_CORRECTA = ['A', 'C', 'D', 'B']
TIEMPO_RESET = 5.0   # segundos sin patrón para borrar buffer


class DetectorContrasena:
    def __init__(self, secuencia_correcta, tiempo_reset=5.0):
        self.secuencia_correcta = secuencia_correcta
        self.tiempo_reset = tiempo_reset
        self.buffer = []
        self.unlocked = False
        self._ultimo_tiempo = time.time()

    def reset(self):
        self.buffer = []
        self.unlocked = False
        self._ultimo_tiempo = time.time()

    def update(self, patron_detectado):
        ahora = time.time()

        if ahora - self._ultimo_tiempo > self.tiempo_reset:
            self.buffer = []

        self._ultimo_tiempo = ahora

        if patron_detectado is None:
            return

        if patron_detectado not in self.secuencia_correcta:
            self.buffer = []
            return

        self.buffer.append(patron_detectado)

        if len(self.buffer) > len(self.secuencia_correcta):
            self.buffer.pop(0)

        if self.buffer == self.secuencia_correcta:
            self.unlocked = True
            self.buffer = []

    def esta_desbloqueado(self):
        return self.unlocked
