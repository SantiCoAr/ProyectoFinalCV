import cv2
import numpy as np

    
def detect_pattern(frame):
    # Conversión a gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Umbralización (Otsu)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morfología para limpiar ruido
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Buscar contornos
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return None

    # Elegir el contorno más grande
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)

    # Si el objeto es demasiado pequeño → no hay patrón válido
    if area < 1000:
        return None

    # Bounding box para medidas geométricas
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = w / h

    # Perímetro para circularidad
    peri = cv2.arcLength(c, True)
    circularity = 4 * np.pi * area / (peri * peri + 1e-6)

    # --- CLASIFICACIÓN ---

    # A → línea vertical
    if h / (w + 1e-6) > 2.5:
        return 'A'

    # B → línea horizontal
    if w / (h + 1e-6) > 2.5:
        return 'B'

    # C → círculo
    if circularity > 0.75:
        return 'C'

    # D → cuadrado
    if 0.85 < aspect_ratio < 1.15:
        return 'D'

    return None
