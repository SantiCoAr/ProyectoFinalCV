import cv2
import numpy as np
import time
 
SECUENCIA_CORRECTA = ['A', 'C', 'D', 'B']
TIEMPO_RESET = 5.0
 
 
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
 
 
def detect_pattern(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
 
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
 
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
 
    if len(contours) == 0:
        return None, thresh, None
 
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
 
    if area < 800:  # si no detecta nada, baja este número
        return None, thresh, None
 
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = w / (h + 1e-6)
 
    peri = cv2.arcLength(c, True)
    circularity = 4 * np.pi * area / (peri * peri + 1e-6)
 
    letra = None
 
    if h / (w + 1e-6) > 2.5:
        letra = 'A'
    elif w / (h + 1e-6) > 2.5:
        letra = 'B'
    elif circularity > 0.75:
        letra = 'C'
    elif 0.85 < aspect_ratio < 1.15:
        letra = 'D'
 
    return letra, thresh, (x, y, w, h)
 
 
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se puede abrir la cámara")
        return
 
    detector = DetectorContrasena(SECUENCIA_CORRECTA, tiempo_reset=TIEMPO_RESET)
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        patron, thresh, bbox = detect_pattern(frame)
        detector.update(patron)
 
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
        texto_patron = f"Patron: {patron}" if patron is not None else "Patron: ninguno"
        cv2.putText(frame, texto_patron, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
 
        if detector.esta_desbloqueado():
            cv2.putText(frame, "DESBLOQUEADO", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
 
        cv2.imshow("Seguridad", frame)
        cv2.imshow("Thresh", thresh)
 
        if cv2.waitKey(1) & 0xFF == 27:
            break
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()