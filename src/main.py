import cv2
from DetectorContrasena import *
from utils import *

def main():
    cap = cv2.VideoCapture(0)
    detector = DetectorContrasena(SECUENCIA_CORRECTA, tiempo_reset=TIEMPO_RESET)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        patron = detect_pattern(frame)
        detector.update(patron)

        texto_patron = f"Patron: {patron}" if patron is not None else "Patron: ninguno"
        cv2.putText(frame, texto_patron, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if detector.esta_desbloqueado():
            cv2.putText(frame, "DESBLOQUEADO", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            # aquí ya podrías salir del bucle o activar el bloque Snake
            # break

        cv2.imshow("Seguridad", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
