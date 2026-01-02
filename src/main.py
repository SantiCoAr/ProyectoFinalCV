# main.py
import time
import cv2

from color_shape_detector import detect_color_shape, draw_detected_pattern
from password_decoder import PasswordDecoder, DecoderConfig

from finger_detector import ColorMarkerDetector
from kalman_tracker import Kalman2DTracker, KalmanConfig
from snake_game import SnakeGame, SnakeConfig


# Configuración

PASSWORD = ["red_circle", "blue_triangle", "green_square", "yellow_line"]

# Tracker 2P split: dos jugadores, cada uno con una cinta de color distinta
P1_COLOR = "green"   # jugador izquierda
P2_COLOR = "red"     # jugador derecha

WIN_SCORE = 10



# Estados de la app
# LOCKED: fase contraseña
# UNLOCKED: fase juego snake 2 jugadores (TRACKER)
class AppState:
    LOCKED = 0
    UNLOCKED = 1


# Utilidad: texto con borde para que se lea bien en cualquier fondo
def overlay_text(frame, text, x, y, scale=0.7):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)



# Utilidad: dibuja la bounding box del marcador detectado/predicho
def draw_bbox(frame, bbox, offset_x=0, ok=True):
    x, y, w, h = bbox
    x += offset_x

    # Verde si el detector midió (detección real), amarillo si viene de predicción Kalman
    color = (0, 255, 0) if ok else (0, 255, 255) 

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


# Utilidad: dibuja la snake (comida, cuerpo y cabeza)
def draw_snake(frame, snake: SnakeGame, offset_x=0):
    # comida
    fx, fy = snake.food
    cv2.circle(frame, (fx + offset_x, fy), 12, (255, 255, 255), -1)

    # serpiente
    segs = snake.get_segments()
    # dibuja línea entre puntos consecutivos para simular cuerpo continuo
    if len(segs) >= 2:
        for i in range(1, len(segs)):
            x1, y1 = segs[i - 1]
            x2, y2 = segs[i]
            cv2.line(frame, (x1 + offset_x, y1), (x2 + offset_x, y2), (255, 255, 255), 6)

    # cabeza (el último punto del cuerpo es la cabeza)
    hx, hy = snake.get_head()
    cv2.circle(frame, (hx + offset_x, hy), 10, (0, 0, 0), -1)


def clamp_center(cx, cy, w, h):
    # asegura que cx esté entre [0, w-1] y cy entre [0, h-1]
    cx = max(0, min(w - 1, int(cx)))
    cy = max(0, min(h - 1, int(cy)))
    return cx, cy


# Utilidad: resetea el juego UNLOCKED (snakes + Kalman)
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

    # Dimensiones del frame completo
    H, W = frame.shape[:2]
    halfW = W // 2

    # 1) Módulos contraseña
    decoder_cfg = DecoderConfig(
        stable_frames_required=20,   # acepta símbolo si se mantiene estable N frames
        release_frames_required=12,  # exige ver None M frames para poder aceptar el siguiente (evita repetición)
        cooldown_seconds=1.5         # seguridad extra de tiempo entre aceptaciones
    )
    # Decodificador que compara secuencia introducida con PASSWORD
    decoder = PasswordDecoder(PASSWORD, decoder_cfg)

    # 2) Módulos tracker 2P split
    det1 = ColorMarkerDetector(P1_COLOR, min_area=900.0) # izquierda
    det2 = ColorMarkerDetector(P2_COLOR, min_area=900.0) # derecha

    # Kalman por jugador
    kf1 = Kalman2DTracker(KalmanConfig(process_noise=1e-2, measurement_noise=1e-1))
    kf2 = Kalman2DTracker(KalmanConfig(process_noise=1e-2, measurement_noise=1e-1))

    # Configuración de snake
    snake_cfg = SnakeConfig(max_length=90, segment_step=12.0, eat_radius=28.0, food_margin=60)

    # Cada snake vive en su “tablero” de tamaño halfW x H
    snake1 = SnakeGame(halfW, H, snake_cfg)
    snake2 = SnakeGame(halfW, H, snake_cfg)

    # Contadores de frames perdidos (si no se detecta el marcador)
    miss1, miss2 = 0, 0
    # Umbral: si se pierde el marcador demasiados frames, se deja de actualizar snake (para evitar saltos raros)
    miss_limit = 20

    winner = None
    winner_time = 0.0
    show_masks = False

    # Estado inicial
    state = AppState.LOCKED

    # FPS (estimación)
    # last_t guarda el timestamp del frame anterior
    last_t = time.time()
    # fps se “suaviza” para que el texto en pantalla no fluctúe bruscamente
    fps = 0.0

    # Mensajes de consola
    print("MAIN unificado")
    print("Estado inicial: LOCKED (introducir contraseña)")
    print("Contraseña:", " - ".join(PASSWORD))
    print("Luego: UNLOCKED -> Snake 2P Split")
    print("Teclas: q salir | r reset (según modo) | m masks on/off | k reset kalman (solo UNLOCKED) | n nueva partida (solo cuando hay ganador)")


    # Loop principal: procesa cámara frame a frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Timestamp actual para FPS y lógicas temporales
        now = time.time()

        # dt = tiempo transcurrido entre el frame actual y el anterior
        dt = now - last_t

        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)  # para suavizar el numero de FPS que salen y que sea legible y evitar saltos irreales que se arreglan en microsegundos
        last_t = now

        # MODO 1: LOCKED (password)
        if state == AppState.LOCKED:
            # Detecta un patrón dominante en el frame completo: color+forma
            pattern = detect_color_shape(frame)
            current_label = pattern.label if pattern else None

            # Actualiza el decodificador con el label detectado (o None si no hay patrón)
            # El decoder decide cuándo aceptar (estabilidad + release + cooldown)
            decoder.update(current_label)

            # Visualización: copia frame y dibuja el patrón detectado (contorno + etiqueta)
            vis = frame.copy()
            vis = draw_detected_pattern(vis, pattern)

            overlay_text(vis, f"FPS: {fps:.1f}", 10, 30, 0.7)
            overlay_text(vis, "MODO: LOCKED (PASSWORD)", 10, 60, 0.7)

            # Muestra cuál es el símbolo actual detectado
            overlay_text(vis, f"Actual: {current_label if current_label else '(ninguno)'}", 10, 90, 0.65)

            # Progreso: cuántos símbolos aceptó ya el decoder (0..4)
            overlay_text(vis, f"Progreso: {decoder.progress()}/4", 10, 120, 0.65)

            # Secuencia ya aceptada
            overlay_text(vis, f"Secuencia: {decoder.progress_str()}", 10, 190, 0.55)

            # Estado interno del decoder
            overlay_text(vis, f"Estado: {decoder.state_str()} | Estab: {decoder.stability_meter()}",
                         10, 220, 0.5)

            overlay_text(vis, "'r' reset. 'q' salir.",
                         10, H - 15, 0.55)

            # Resultado de intento:
            # decoder.last_result se setea cuando se introducen 4 símbolos (True/False)
            if decoder.last_result is not None:
                msg = "CONTRASENA CORRECTA" if decoder.last_result else "CONTRASENA INCORRECTA"
                overlay_text(vis, msg, 10, 260, 0.8)

                # Si la contraseña es correcta: transición a modo UNLOCKED
                if decoder.last_result is True:
                    state = AppState.UNLOCKED

                    # resetea el estado del juego/tracker para empezar “limpio”
                    reset_unlocked_game(snake1, snake2, kf1, kf2)
                    miss1, miss2 = 0, 0
                    winner = None
                    print(">>> UNLOCKED: entrando en tracker 2P split")

            # Muestra ventana principal
            cv2.imshow("MAIN (Password + Tracker)", vis)

            # En modo LOCKED no se usan masks (solo existen en tracker). Si estaban, las cerramos.
            if show_masks:
                # no hay máscaras de color aquí (solo para tracker), cerramos si estaban
                try:
                    cv2.destroyWindow("Mask P1")
                    cv2.destroyWindow("Mask P2")
                except:
                    pass

        # MODO 2: UNLOCKED (tracker 2P split)
        else:
            # Divide el frame en dos regiones:
            left = frame[:, :halfW]
            right = frame[:, halfW:]

            # Detecta el marcador de color en cada mitad
            # d1/d2: MarkerDetection o None
            # m1/m2: máscara binaria (para debug)
            d1, m1 = det1.detect(left)
            d2, m2 = det2.detect(right)

            # P1 (Jugador izquierda)
            tracked1, measured1, bbox1 = None, False, None

            # Si el Kalman está inicializado, obtenemos predicción (sirve si no hay detección)
            if kf1.initialized:
                predx1, predy1 = kf1.predict()
            else:
                predx1, predy1 = None, None

            # Si hay detección real:
            if d1 is not None:
                measured1 = True
                miss1 = 0

                # Centro medido del detector
                mx, my = clamp_center(d1.center[0], d1.center[1], halfW, H)

                # Si el Kalman no se inicializó aún, se inicializa con primera medición
                if not kf1.initialized:
                    kf1.initialize(mx, my)

                # Corrección Kalman
                x1, y1 = kf1.correct(mx, my)

                # tracked1 es la posición final usada para mover la snake
                tracked1 = (x1, y1)

                # bbox real del detector
                bbox1 = d1.bbox

            # Si NO hay detección:
            else:
                miss1 += 1

                # Si Kalman está inicializado, usamos la predicción como “posición estimada”
                if kf1.initialized and predx1 is not None:
                    tracked1 = (predx1, predy1)

                    # Creamos bbox ficticia centrada en la predicción para visualizar “tracking por predicción”
                    bw, bh = 70, 70
                    x = int(predx1 - bw / 2)
                    y = int(predy1 - bh / 2)
                    x = max(0, min(halfW - bw, x))
                    y = max(0, min(H - bh, y))
                    bbox1 = (x, y, bw, bh)


            # P2 (Jugador derecha)
            tracked2, measured2, bbox2 = None, False, None

            # Predicción Kalman si está inicializado
            if kf2.initialized:
                predx2, predy2 = kf2.predict()
            else:
                predx2, predy2 = None, None

            # Si hay detección real:
            if d2 is not None:
                measured2 = True
                miss2 = 0

                # Centro detectado
                mx, my = clamp_center(d2.center[0], d2.center[1], halfW, H)

                # Inicializa Kalman con primera medición si hiciera falta
                if not kf2.initialized:
                    kf2.initialize(mx, my)

                # Corrección Kalman
                x2, y2 = kf2.correct(mx, my)
                tracked2 = (x2, y2)

                # bbox real del detector
                bbox2 = d2.bbox

            # Si NO hay detección:
            else:
                miss2 += 1

                # Si hay predicción válida, se usa como tracking
                if kf2.initialized and predx2 is not None:
                    tracked2 = (predx2, predy2)

                    # bbox ficticia basada en predicción
                    bw, bh = 70, 70
                    x = int(predx2 - bw / 2)
                    y = int(predy2 - bh / 2)
                    x = max(0, min(halfW - bw, x))
                    y = max(0, min(H - bh, y))
                    bbox2 = (x, y, bw, bh)

            # Actualizar snakes (solo si no hay ganador)
            # Si hay ganador, se congela el juego para mostrar mensaje y esperar 'n' o 'q'
            if winner is None:
                # Actualiza snake solo si hay posición y no se perdió demasiados frames
                if tracked1 is not None and miss1 <= miss_limit:
                    snake1.update(tracked1)
                if tracked2 is not None and miss2 <= miss_limit:
                    snake2.update(tracked2)

            # Comprobar victoria
            # Se fija winner cuando alguno llega a WIN_SCORE
            if winner is None:
                if snake1.score >= WIN_SCORE:
                    winner = "P1 (IZQUIERDA)"
                    winner_time = now
                elif snake2.score >= WIN_SCORE:
                    winner = "P2 (DERECHA)"
                    winner_time = now

            # Dibujo / UI
            vis = frame.copy()

            # Línea separadora entre tableros
            cv2.line(vis, (halfW, 0), (halfW, H), (255, 255, 255), 2)

            # Bounding boxes (si existen). Para P2 se suma offset_x=halfW
            if bbox1 is not None:
                draw_bbox(vis, bbox1, offset_x=0, ok=measured1)
            if bbox2 is not None:
                draw_bbox(vis, bbox2, offset_x=halfW, ok=measured2)

            # Dibuja snakes en cada mitad
            draw_snake(vis, snake1, offset_x=0)
            draw_snake(vis, snake2, offset_x=halfW)

            # Overlay de estado
            overlay_text(vis, f"FPS: {fps:.1f}", 10, 30, 0.7)
            overlay_text(vis, "MODO: UNLOCKED (TRACKER 2P SPLIT)", 10, 60, 0.7)

            # Score por jugador
            overlay_text(vis, f"P1 ({P1_COLOR}) score: {snake1.score}/{WIN_SCORE}", 10, 90, 0.65)
            overlay_text(vis, f"P2 ({P2_COLOR}) score: {snake2.score}/{WIN_SCORE}", halfW + 10, 90, 0.65)

            # Indicador de si se está midiendo o prediciendo
            st1 = "MEASURED" if measured1 else f"PREDICT"
            st2 = "MEASURED" if measured2 else f"PREDICT"
            overlay_text(vis, f"P1: {st1}", 10, 120, 0.55)
            overlay_text(vis, f"P2: {st2}", halfW + 10, 120, 0.55)

            # Texto de teclas dependiendo de si hay ganador o no
            if winner is None:
                overlay_text(vis, "'r' reset partida | 'k' reset kalman | 'm' masks | 'q' salir",
                             10, H - 15, 0.55)
            # Mensaje de ganador
            else:
                overlay_text(vis, f"GANADOR: {winner}", halfW - 220, H // 2 - 20, 1.0)
                overlay_text(vis, "Pulsa 'n' para nueva partida o 'q' para salir", halfW - 320, H // 2 + 30, 0.7)

            # Mostrar ventana principal
            cv2.imshow("MAIN (Password + Tracker)", vis)

            # Mostrar máscaras si el modo debug está activo
            if show_masks:
                cv2.imshow("Mask P1", m1)
                cv2.imshow("Mask P2", m2)
            else:
                # Si no queremos máscaras, cerramos las ventanas si estaban abiertas
                try:
                    cv2.destroyWindow("Mask P1")
                    cv2.destroyWindow("Mask P2")
                except:
                    pass

        # Teclado global
        key = cv2.waitKey(1) & 0xFF

        # 'q' siempre sale
        if key == ord("q"):
            break

        # Teclas en LOCKED
        if state == AppState.LOCKED:
            # 'r' resetea el intento de contraseña
            if key == ord("r"):
                decoder.reset()

            # 'm' aquí no se usa (máscaras solo existen en tracker), forzamos a False por consistencia
            if key == ord("m"):
                show_masks = False

        # Teclas en UNLOCKED
        else:
            if winner is not None:
                if key == ord("n"):
                    # reinicia snakes + kalman + contadores + winner
                    reset_unlocked_game(snake1, snake2, kf1, kf2)
                    miss1, miss2 = 0, 0
                    winner = None
            else:
                # 'r' resetea partida (snakes y contadores)
                if key == ord("r"):
                    snake1.reset()
                    snake2.reset()
                    winner = None
                    miss1, miss2 = 0, 0

                # 'k' resetea Kalman (si el tracker se fue a cualquier lado)
                elif key == ord("k"):
                    kf1.reset()
                    kf2.reset()
                    miss1, miss2 = 0, 0

                # 'm' alterna mostrar/ocultar máscaras de segmentación
                elif key == ord("m"):
                    show_masks = not show_masks

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
