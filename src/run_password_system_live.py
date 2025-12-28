# run_password_system_live.py

import cv2
from color_shape_detector import detect_color_shape, draw_detected_pattern
from password_decoder import PasswordDecoder, DecoderConfig


PASSWORD = ["blue_square", "red_circle",  "green_square", "yellow_line"]


def draw_overlay(frame, decoder: PasswordDecoder, current_label):
    out = frame

    # Texto superior
    detected_txt = f"Actual: {current_label if current_label else '(ninguno)'}"
    cv2.putText(out, detected_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(out, detected_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

    # Progreso
    prog = decoder.progress()
    prog_txt = f"Progreso: {prog}/4"
    cv2.putText(out, prog_txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(out, prog_txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

    # Secuencia parcial
    seq_txt = f"Secuencia: {decoder.progress_str()}"
    cv2.putText(out, seq_txt, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(out, seq_txt, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    # Estado interno (útil para debug)
    st_txt = f"Estado: {decoder.state_str()} | Estabilidad: {decoder.stability_meter()}"
    cv2.putText(out, st_txt, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(out, st_txt, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    # Resultado final (si existe)
    if decoder.last_result is not None:
        res_txt = "ENHORABUENA CONTRASENA CORRECTA" if decoder.last_result else "INTENTALO OTRA VEZ CONTRASENA INCORRECTA "
        cv2.putText(out, res_txt, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, res_txt, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1, cv2.LINE_AA)

    # Ayuda
    help_txt = "'r': resetear |  'q': salir"
    cv2.putText(out, help_txt, (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(out, help_txt, (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    return out


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se ha podido abrir la cámara.")
        return

    # Ajustes finos: puedes tocar esto según tu cámara/FPS
    cfg = DecoderConfig(
        stable_frames_required=12,
        release_frames_required=8,
        cooldown_seconds=0.6
    )
    decoder = PasswordDecoder(PASSWORD, cfg)

    print("Password:", " - ".join(PASSWORD))
    print("IMPORTANTE: Retira el patrón (sin nada) entre símbolos para que el sistema acepte el siguiente.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pattern = detect_color_shape(frame)
        frame_vis = draw_detected_pattern(frame, pattern)

        current_label = pattern.label if pattern else None

        # Actualiza el decodificador con la detección del frame
        decoder.update(current_label)

        # Overlay
        frame_vis = draw_overlay(frame_vis, decoder, current_label)

        cv2.imshow("Password system (Color+Shape)", frame_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            decoder.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
