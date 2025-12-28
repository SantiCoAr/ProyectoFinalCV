# run_snake_tracker.py

import time
import cv2

from finger_detector import RedFingerDetector
from kalman_tracker import Kalman2DTracker, KalmanConfig
from snake_game import SnakeGame, SnakeConfig


def draw_bbox(frame, bbox, ok=True):
    x, y, w, h = bbox
    color = (0, 255, 0) if ok else (0, 255, 255)  # verde si hay medición, amarillo si es predicción
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def draw_snake(frame, snake: SnakeGame):
    # comida
    fx, fy = snake.food
    cv2.circle(frame, (fx, fy), 12, (255, 255, 255), -1)

    # serpiente
    segs = snake.get_segments()
    if len(segs) >= 2:
        for i in range(1, len(segs)):
            cv2.line(frame, segs[i - 1], segs[i], (255, 255, 255), 6)

    # cabeza
    hx, hy = snake.get_head()
    cv2.circle(frame, (hx, hy), 10, (0, 0, 0), -1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se ha podido abrir la cámara.")
        return

    # Leer un frame para saber tamaño
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer frame inicial.")
        return

    h, w = frame.shape[:2]

    # Detector de dedo (rojo)
    detector = RedFingerDetector(
        min_area=900.0,       # sube si te detecta rojos de fondo
        morph_ksize=5,
        open_iters=2,
        close_iters=2
    )

    # Kalman
    kf = Kalman2DTracker(KalmanConfig(
        dt=1.0,
        process_noise=1e-2,
        measurement_noise=1e-1,
        error_cov_post=1.0
    ))

    # Snake
    snake = SnakeGame(w, h, SnakeConfig(
        max_length=90,
        segment_step=12.0,
        eat_radius=28.0,
        food_margin=70
    ))

    # Para FPS
    last_t = time.time()
    fps = 0.0

    # Para manejar pérdida de detección
    missing_frames = 0
    missing_limit = 20  # si pierde el dedo muchos frames, se "congela" el juego (opcional)

    print("Controles: q salir | r reset snake | k reset kalman | m mostrar mascara on/off")

    show_mask = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt = now - last_t
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        last_t = now

        det, mask = detector.detect(frame)

        # Predicción siempre
        if not kf.initialized:
            # si no está inicializado, solo intentamos inicializar cuando haya detección
            pred_x, pred_y = (None, None)
        else:
            pred_x, pred_y = kf.predict()

        measured_ok = False
        tracked_x, tracked_y = None, None
        bbox_to_draw = None

        if det is not None:
            measured_ok = True
            missing_frames = 0

            mx, my = det.center

            if not kf.initialized:
                kf.initialize(mx, my)

            # corregir con medición
            tracked_x, tracked_y = kf.correct(mx, my)

            # bbox de medición
            bbox_to_draw = det.bbox

        else:
            missing_frames += 1
            # sin medición: usamos predicción si existe
            if kf.initialized and pred_x is not None:
                tracked_x, tracked_y = pred_x, pred_y

                # bbox sintético alrededor de la predicción (tamaño fijo)
                # esto mantiene el requisito de bbox incluso si hay un mini fallo de detección
                box_w, box_h = 70, 70
                x = int(tracked_x - box_w / 2)
                y = int(tracked_y - box_h / 2)
                x = max(0, min(w - box_w, x))
                y = max(0, min(h - box_h, y))
                bbox_to_draw = (x, y, box_w, box_h)

        # Actualizar snake si tenemos posición trackeada
        if tracked_x is not None and tracked_y is not None:
            # opcional: si se pierde demasiado, no actualices (para que no “vuele”)
            if missing_frames <= missing_limit:
                snake.update((tracked_x, tracked_y))

        # Dibujo
        vis = frame.copy()

        # bbox
        if bbox_to_draw is not None:
            draw_bbox(vis, bbox_to_draw, ok=measured_ok)

        # snake
        draw_snake(vis, snake)

        # textos
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        cv2.putText(vis, f"Score: {snake.score}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, f"Score: {snake.score}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        status = "MEASURED" if measured_ok else "PREDICT"
        cv2.putText(vis, f"Tracking: {status} | missing={missing_frames}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, f"Tracking: {status} | missing={missing_frames}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        cv2.putText(vis, "Pon cinta roja en el dedo. Evita objetos rojos de fondo.", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, "Pon cinta roja en el dedo. Evita objetos rojos de fondo.", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

        cv2.imshow("Snake Tracker (Red Finger)", vis)

        if show_mask:
            cv2.imshow("Mask (red)", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            snake.reset()
        elif key == ord("k"):
            kf.reset()
        elif key == ord("m"):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow("Mask (red)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
