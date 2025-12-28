# run_snake_tracker_2p_split.py

import time
import cv2

from finger_detector import ColorMarkerDetector
from kalman_tracker import Kalman2DTracker, KalmanConfig
from snake_game import SnakeGame, SnakeConfig


WIN_SCORE = 10  # gana el primero en llegar a esto


def draw_bbox(frame, bbox, offset_x=0, ok=True):
    x, y, w, h = bbox
    x += offset_x
    color = (0, 255, 0) if ok else (0, 255, 255)  # verde: medido, amarillo: predicción
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def draw_snake(frame, snake: SnakeGame, offset_x=0):
    # comida
    fx, fy = snake.food
    cv2.circle(frame, (fx + offset_x, fy), 12, (255, 255, 255), -1)

    # serpiente
    segs = snake.get_segments()
    if len(segs) >= 2:
        for i in range(1, len(segs)):
            x1, y1 = segs[i - 1]
            x2, y2 = segs[i]
            cv2.line(frame, (x1 + offset_x, y1), (x2 + offset_x, y2), (255, 255, 255), 6)

    # cabeza
    hx, hy = snake.get_head()
    cv2.circle(frame, (hx + offset_x, hy), 10, (0, 0, 0), -1)


def overlay_text(frame, text, x, y, scale=0.7):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def clamp_bbox_center(cx, cy, w, h):
    cx = max(0, min(w - 1, int(cx)))
    cy = max(0, min(h - 1, int(cy)))
    return cx, cy


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se ha podido abrir la cámara.")
        return

    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer frame inicial.")
        return

    H, W = frame.shape[:2]
    halfW = W // 2

    # -------------------------
    # Jugador 1 (izquierda): VERDE
    # Jugador 2 (derecha): ROJO
    # -------------------------
    det1 = ColorMarkerDetector("green", min_area=900.0)
    det2 = ColorMarkerDetector("red", min_area=900.0)

    kf1 = Kalman2DTracker(KalmanConfig(process_noise=1e-2, measurement_noise=1e-1))
    kf2 = Kalman2DTracker(KalmanConfig(process_noise=1e-2, measurement_noise=1e-1))

    snake_cfg = SnakeConfig(max_length=90, segment_step=12.0, eat_radius=28.0, food_margin=60)
    snake1 = SnakeGame(halfW, H, snake_cfg)
    snake2 = SnakeGame(halfW, H, snake_cfg)

    show_masks = False
    winner = None
    winner_time = 0.0

    # FPS
    last_t = time.time()
    fps = 0.0

    # pérdida detección (para evitar que “vuele”)
    miss1 = 0
    miss2 = 0
    miss_limit = 20

    print("2P Split-Screen")
    print("P1 izquierda: cinta VERDE | P2 derecha: cinta ROJA")
    print("Teclas: q salir | r reset partida | k reset kalman | m masks on/off")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt = now - last_t
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        last_t = now

        # Split
        left = frame[:, :halfW]
        right = frame[:, halfW:]

        # Detectar en cada mitad
        d1, m1 = det1.detect(left)
        d2, m2 = det2.detect(right)

        # ------- Tracking P1 -------
        tracked1 = None
        measured1 = False
        bbox1 = None

        if kf1.initialized:
            predx1, predy1 = kf1.predict()
        else:
            predx1, predy1 = None, None

        if d1 is not None:
            measured1 = True
            miss1 = 0
            mx, my = d1.center
            mx, my = clamp_bbox_center(mx, my, halfW, H)
            if not kf1.initialized:
                kf1.initialize(mx, my)
            x1, y1 = kf1.correct(mx, my)
            tracked1 = (x1, y1)
            bbox1 = d1.bbox
        else:
            miss1 += 1
            if kf1.initialized and predx1 is not None:
                tracked1 = (predx1, predy1)
                # bbox sintético
                bw, bh = 70, 70
                x = int(predx1 - bw / 2)
                y = int(predy1 - bh / 2)
                x = max(0, min(halfW - bw, x))
                y = max(0, min(H - bh, y))
                bbox1 = (x, y, bw, bh)

        # ------- Tracking P2 -------
        tracked2 = None
        measured2 = False
        bbox2 = None

        if kf2.initialized:
            predx2, predy2 = kf2.predict()
        else:
            predx2, predy2 = None, None

        if d2 is not None:
            measured2 = True
            miss2 = 0
            mx, my = d2.center
            mx, my = clamp_bbox_center(mx, my, halfW, H)
            if not kf2.initialized:
                kf2.initialize(mx, my)
            x2, y2 = kf2.correct(mx, my)
            tracked2 = (x2, y2)
            bbox2 = d2.bbox
        else:
            miss2 += 1
            if kf2.initialized and predx2 is not None:
                tracked2 = (predx2, predy2)
                bw, bh = 70, 70
                x = int(predx2 - bw / 2)
                y = int(predy2 - bh / 2)
                x = max(0, min(halfW - bw, x))
                y = max(0, min(H - bh, y))
                bbox2 = (x, y, bw, bh)

        # Actualizar snakes (si no están demasiado perdidos)
        if tracked1 is not None and miss1 <= miss_limit:
            snake1.update(tracked1)
        if tracked2 is not None and miss2 <= miss_limit:
            snake2.update(tracked2)

        # Chequear victoria
        if winner is None:
            if snake1.score >= WIN_SCORE:
                winner = "P1 (IZQUIERDA)"
                winner_time = now
            elif snake2.score >= WIN_SCORE:
                winner = "P2 (DERECHA)"
                winner_time = now
        else:
            # mantener mensaje 3s y luego reset automático (opcional)
            if (now - winner_time) > 3.0:
                winner = None
                snake1.reset()
                snake2.reset()
                kf1.reset()
                kf2.reset()
                miss1 = 0
                miss2 = 0

        # Dibujar separador
        vis = frame.copy()
        cv2.line(vis, (halfW, 0), (halfW, H), (255, 255, 255), 2)

        # Dibujar bbox + snake en cada mitad
        if bbox1 is not None:
            draw_bbox(vis, bbox1, offset_x=0, ok=measured1)
        if bbox2 is not None:
            draw_bbox(vis, bbox2, offset_x=halfW, ok=measured2)

        draw_snake(vis, snake1, offset_x=0)
        draw_snake(vis, snake2, offset_x=halfW)

        # Overlays
        overlay_text(vis, f"FPS: {fps:.1f}", 10, 30, 0.7)
        overlay_text(vis, f"P1 score: {snake1.score}/{WIN_SCORE}", 10, 60, 0.7)
        overlay_text(vis, f"P2 score: {snake2.score}/{WIN_SCORE}", halfW + 10, 60, 0.7)

        st1 = "MEASURED" if measured1 else f"PREDICT (miss={miss1})"
        st2 = "MEASURED" if measured2 else f"PREDICT (miss={miss2})"
        overlay_text(vis, f"P1: {st1}", 10, 90, 0.55)
        overlay_text(vis, f"P2: {st2}", halfW + 10, 90, 0.55)

        overlay_text(vis, "P1: VERDE (izq) | P2: ROJO (der)", 10, H - 15, 0.55)

        if winner is not None:
            overlay_text(vis, f"GANADOR: {winner}", halfW - 170, H // 2, 1.0)

        cv2.imshow("Snake 2P Split-Screen", vis)

        if show_masks:
            cv2.imshow("Mask P1 (green)", m1)
            cv2.imshow("Mask P2 (red)", m2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            snake1.reset()
            snake2.reset()
            winner = None
            miss1 = 0
            miss2 = 0
        elif key == ord("k"):
            kf1.reset()
            kf2.reset()
            miss1 = 0
            miss2 = 0
        elif key == ord("m"):
            show_masks = not show_masks
            if not show_masks:
                # cerrar ventanas máscara si estaban abiertas
                try:
                    cv2.destroyWindow("Mask P1 (green)")
                    cv2.destroyWindow("Mask P2 (red)")
                except:
                    pass

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
