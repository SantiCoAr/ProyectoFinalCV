# main.py
import time
import cv2

from color_shape_detector import detect_color_shape, draw_detected_pattern
from password_decoder import PasswordDecoder, DecoderConfig

from finger_detector import ColorMarkerDetector
from kalman_tracker import Kalman2DTracker, KalmanConfig
from snake_game import SnakeGame, SnakeConfig


# -----------------------
# Configuración
# -----------------------
PASSWORD = ["red_circle", "blue_triangle", "green_square", "yellow_line"]

# Tracker 2P split
P1_COLOR = "green"   # jugador izquierda
P2_COLOR = "red"     # jugador derecha
WIN_SCORE = 10


class AppState:
    LOCKED = 0
    UNLOCKED = 1


def overlay_text(frame, text, x, y, scale=0.7):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_progress_bar(frame, progress, total=4, x=10, y=95):
    blocks = ["■" if i < progress else "□" for i in range(total)]
    bar = " ".join(blocks)
    overlay_text(frame, bar, x, y, 0.9)


def draw_bbox(frame, bbox, offset_x=0, ok=True):
    x, y, w, h = bbox
    x += offset_x
    color = (0, 255, 0) if ok else (0, 255, 255)  # verde si medido, amarillo si predicción
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


def clamp_center(cx, cy, w, h):
    cx = max(0, min(w - 1, int(cx)))
    cy = max(0, min(h - 1, int(cy)))
    return cx, cy


def reset_unlocked_game(snake1, snake2, kf1, kf2):
    snake1.reset()
    snake2.reset()
    kf1.reset()
    kf2.reset()


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

    # -----------------------
    # 1) Módulos contraseña
    # -----------------------
    decoder_cfg = DecoderConfig(
        stable_frames_required=12,
        release_frames_required=8,
        cooldown_seconds=0.6
    )
    decoder = PasswordDecoder(PASSWORD, decoder_cfg)

    # -----------------------
    # 2) Módulos tracker 2P split
    # -----------------------
    det1 = ColorMarkerDetector(P1_COLOR, min_area=900.0)
    det2 = ColorMarkerDetector(P2_COLOR, min_area=900.0)

    kf1 = Kalman2DTracker(KalmanConfig(process_noise=1e-2, measurement_noise=1e-1))
    kf2 = Kalman2DTracker(KalmanConfig(process_noise=1e-2, measurement_noise=1e-1))

    snake_cfg = SnakeConfig(max_length=90, segment_step=12.0, eat_radius=28.0, food_margin=60)
    snake1 = SnakeGame(halfW, H, snake_cfg)
    snake2 = SnakeGame(halfW, H, snake_cfg)

    miss1, miss2 = 0, 0
    miss_limit = 20
    winner = None
    winner_time = 0.0

    show_masks = False
    state = AppState.LOCKED

    # FPS
    last_t = time.time()
    fps = 0.0

    print("MAIN unificado")
    print("Estado inicial: LOCKED (introducir contraseña)")
    print("Contraseña:", " - ".join(PASSWORD))
    print("Luego: UNLOCKED -> Snake 2P Split")
    print("Teclas: q salir | r reset (según modo) | m masks on/off | k reset kalman (solo UNLOCKED) | n nueva partida (solo cuando hay ganador)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt = now - last_t
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)  # para suavizar el numero de FPS que salen y que sea legible y evitar saltos irreales que se arreglan en microsegundos
        last_t = now

        # -----------------------
        # MODO 1: LOCKED (password)
        # -----------------------
        if state == AppState.LOCKED:
            pattern = detect_color_shape(frame)
            current_label = pattern.label if pattern else None

            decoder.update(current_label)

            vis = frame.copy()
            vis = draw_detected_pattern(vis, pattern)

            overlay_text(vis, f"FPS: {fps:.1f}", 10, 30, 0.7)
            overlay_text(vis, "MODO: LOCKED (PASSWORD)", 10, 60, 0.7)

            overlay_text(vis, f"Actual: {current_label if current_label else '(ninguno)'}", 10, 90, 0.65)
            overlay_text(vis, f"Progreso: {decoder.progress()}/4", 10, 120, 0.65)
            draw_progress_bar(vis, decoder.progress(), 4, 10, 155)

            overlay_text(vis, f"Secuencia: {decoder.progress_str()}", 10, 190, 0.55)
            overlay_text(vis, f"Estado: {decoder.state_str()} | Estab: {decoder.stability_meter()}",
                         10, 220, 0.5)

            overlay_text(vis, "Retira el patron entre simbolos. 'r' reset. 'q' salir.",
                         10, H - 15, 0.55)

            # Resultado de intento
            if decoder.last_result is not None:
                msg = "CONTRASENA CORRECTA ✅" if decoder.last_result else "CONTRASENA INCORRECTA ❌"
                overlay_text(vis, msg, 10, 260, 0.8)

                if decoder.last_result is True:
                    # transición a tracker
                    state = AppState.UNLOCKED
                    reset_unlocked_game(snake1, snake2, kf1, kf2)
                    miss1, miss2 = 0, 0
                    winner = None
                    print(">>> UNLOCKED: entrando en tracker 2P split")

            cv2.imshow("MAIN (Password + Tracker)", vis)

            if show_masks:
                # no hay máscaras de color aquí (solo para tracker), cerramos si estaban
                try:
                    cv2.destroyWindow("Mask P1")
                    cv2.destroyWindow("Mask P2")
                except:
                    pass

        # -----------------------
        # MODO 2: UNLOCKED (tracker 2P split)
        # -----------------------
        else:
            left = frame[:, :halfW]
            right = frame[:, halfW:]

            d1, m1 = det1.detect(left)
            d2, m2 = det2.detect(right)

            # P1
            tracked1, measured1, bbox1 = None, False, None
            if kf1.initialized:
                predx1, predy1 = kf1.predict()
            else:
                predx1, predy1 = None, None

            if d1 is not None:
                measured1 = True
                miss1 = 0
                mx, my = clamp_center(d1.center[0], d1.center[1], halfW, H)
                if not kf1.initialized:
                    kf1.initialize(mx, my)
                x1, y1 = kf1.correct(mx, my)
                tracked1 = (x1, y1)
                bbox1 = d1.bbox
            else:
                miss1 += 1
                if kf1.initialized and predx1 is not None:
                    tracked1 = (predx1, predy1)
                    bw, bh = 70, 70
                    x = int(predx1 - bw / 2)
                    y = int(predy1 - bh / 2)
                    x = max(0, min(halfW - bw, x))
                    y = max(0, min(H - bh, y))
                    bbox1 = (x, y, bw, bh)

            # P2
            tracked2, measured2, bbox2 = None, False, None
            if kf2.initialized:
                predx2, predy2 = kf2.predict()
            else:
                predx2, predy2 = None, None

            if d2 is not None:
                measured2 = True
                miss2 = 0
                mx, my = clamp_center(d2.center[0], d2.center[1], halfW, H)
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

            # Actualizar snakes SOLO si no hay ganador (freeze al ganar)
            if winner is None:
                if tracked1 is not None and miss1 <= miss_limit:
                    snake1.update(tracked1)
                if tracked2 is not None and miss2 <= miss_limit:
                    snake2.update(tracked2)

            # Victoria (sin auto-reset)
            if winner is None:
                if snake1.score >= WIN_SCORE:
                    winner = "P1 (IZQUIERDA)"
                    winner_time = now
                elif snake2.score >= WIN_SCORE:
                    winner = "P2 (DERECHA)"
                    winner_time = now

            # Dibujo
            vis = frame.copy()
            cv2.line(vis, (halfW, 0), (halfW, H), (255, 255, 255), 2)

            if bbox1 is not None:
                draw_bbox(vis, bbox1, offset_x=0, ok=measured1)
            if bbox2 is not None:
                draw_bbox(vis, bbox2, offset_x=halfW, ok=measured2)

            draw_snake(vis, snake1, offset_x=0)
            draw_snake(vis, snake2, offset_x=halfW)

            overlay_text(vis, f"FPS: {fps:.1f}", 10, 30, 0.7)
            overlay_text(vis, "MODO: UNLOCKED (TRACKER 2P SPLIT)", 10, 60, 0.7)
            overlay_text(vis, f"P1 ({P1_COLOR}) score: {snake1.score}/{WIN_SCORE}", 10, 90, 0.65)
            overlay_text(vis, f"P2 ({P2_COLOR}) score: {snake2.score}/{WIN_SCORE}", halfW + 10, 90, 0.65)

            st1 = "MEASURED" if measured1 else f"PREDICT (miss={miss1})"
            st2 = "MEASURED" if measured2 else f"PREDICT (miss={miss2})"
            overlay_text(vis, f"P1: {st1}", 10, 120, 0.55)
            overlay_text(vis, f"P2: {st2}", halfW + 10, 120, 0.55)

            if winner is None:
                overlay_text(vis, "Teclas: r reset partida | k reset kalman | m masks | q salir",
                             10, H - 15, 0.55)
            else:
                overlay_text(vis, "Teclas: n nueva partida | q salir",
                             10, H - 15, 0.55)

            if winner is not None:
                overlay_text(vis, f"GANADOR: {winner}", halfW - 220, H // 2 - 20, 1.0)
                overlay_text(vis, "Pulsa 'n' para nueva partida o 'q' para salir", halfW - 320, H // 2 + 30, 0.7)

            cv2.imshow("MAIN (Password + Tracker)", vis)

            if show_masks:
                cv2.imshow("Mask P1", m1)
                cv2.imshow("Mask P2", m2)
            else:
                try:
                    cv2.destroyWindow("Mask P1")
                    cv2.destroyWindow("Mask P2")
                except:
                    pass

        # -----------------------
        # Teclado global
        # -----------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if state == AppState.LOCKED:
            if key == ord("r"):
                decoder.reset()
            if key == ord("m"):
                show_masks = False

        else:
            # Si hay ganador, solo permitimos nueva partida con 'n' (y salir con 'q')
            if winner is not None:
                if key == ord("n"):
                    reset_unlocked_game(snake1, snake2, kf1, kf2)
                    miss1, miss2 = 0, 0
                    winner = None
                # opcional: ignoramos r/k/m mientras hay ganador para evitar estados raros
            else:
                if key == ord("r"):
                    snake1.reset()
                    snake2.reset()
                    winner = None
                    miss1, miss2 = 0, 0
                elif key == ord("k"):
                    kf1.reset()
                    kf2.reset()
                    miss1, miss2 = 0, 0
                elif key == ord("m"):
                    show_masks = not show_masks

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
