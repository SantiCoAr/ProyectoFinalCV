# test_password_sequence.py

import cv2
from color_shape_detector import detect_color_shape, draw_detected_pattern


class PatternPasswordSystem:
    def __init__(self, password_sequence):
        """
        password_sequence: lista de labels, ej:
        ["red_circle", "blue_triangle", "green_square", "yellow_line"]
        """
        self.password = password_sequence
        self.entered = []

    def reset(self):
        self.entered = []

    def add_observation(self, label: str) -> str:
        """
        Añade un patrón observado.
        Devuelve:
        - "INCOMPLETE"
        - "ACCESS_GRANTED"
        - "ACCESS_DENIED"
        """
        self.entered.append(label)

        if len(self.entered) < len(self.password):
            return "INCOMPLETE"

        # Tenemos ya tantos elementos como la contraseña
        if self.entered == self.password:
            result = "ACCESS_GRANTED"
        else:
            result = "ACCESS_DENIED"

        # Una vez comprobado, reseteamos para el siguiente intento
        self.reset()
        return result

    def get_entered_str(self) -> str:
        return " - ".join(self.entered)


def main():
    # Define aquí tu contraseña (orden de patrones)
    PASSWORD = [
        "red_circle",
        "blue_triangle",
        "green_square",
        "yellow_line",
    ]

    system = PatternPasswordSystem(PASSWORD)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se ha podido abrir la cámara.")
        return

    status_msg = "Pulsa 'c' para capturar, 'r' para reset, 'q' para salir."
    last_result_msg = ""

    print("Secuencia de contraseña:", " - ".join(PASSWORD))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer frame de la cámara.")
            break

        pattern = detect_color_shape(frame)
        frame_vis = draw_detected_pattern(frame, pattern)

        if pattern is not None:
            detected_text = f"Actual: {pattern.label}"
        else:
            detected_text = "Actual: (ninguno)"

        # Mensajes en pantalla
        cv2.putText(
            frame_vis,
            detected_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv2.LINE_AA
        )
        cv2.putText(
            frame_vis,
            detected_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame_vis,
            status_msg,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA
        )
        cv2.putText(
            frame_vis,
            status_msg,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        if last_result_msg:
            cv2.putText(
                frame_vis,
                last_result_msg,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                3,
                cv2.LINE_AA
            )
            cv2.putText(
                frame_vis,
                last_result_msg,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

        # Secuencia parcial introducida
        seq_text = "Secuencia parcial: " + system.get_entered_str()
        cv2.putText(
            frame_vis,
            seq_text,
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA
        )
        cv2.putText(
            frame_vis,
            seq_text,
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.imshow("Sistema de contraseña (color + forma)", frame_vis)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):
            system.reset()
            last_result_msg = "Secuencia reseteada."

        elif key == ord("c"):
            if pattern is None:
                last_result_msg = "No se ha detectado ningún patrón al capturar."
            else:
                result = system.add_observation(pattern.label)
                if result == "INCOMPLETE":
                    last_result_msg = f"Capturado: {pattern.label}"
                elif result == "ACCESS_GRANTED":
                    last_result_msg = "ACCESO CONCEDIDO ✅"
                elif result == "ACCESS_DENIED":
                    last_result_msg = "ACCESO DENEGADO ❌"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
