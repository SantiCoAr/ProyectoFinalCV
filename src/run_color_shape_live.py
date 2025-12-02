# run_color_shape_live.py

import cv2
from color_shape_detector import detect_color_shape, draw_detected_pattern


def main():
    cap = cv2.VideoCapture(0)  # Cambia el índice si usas otra cámara

    if not cap.isOpened():
        print("No se ha podido abrir la cámara.")
        return

    print("Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer frame de la cámara.")
            break

        pattern = detect_color_shape(frame)
        frame_vis = draw_detected_pattern(frame, pattern)

        if pattern is not None:
            text = f"Detectado: {pattern.label}"
        else:
            text = "Sin patrón"

        cv2.putText(
            frame_vis,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            3,
            cv2.LINE_AA
        )
        cv2.putText(
            frame_vis,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.imshow("Color + Forma (Live)", frame_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
