from typing import List
import numpy as np
import cv2
import copy  
import glob

def show_image(window_tag, image_bgr):
    # Mostrar imagen en una ventana (debug/visualización)
    cv2.imshow("drawchessboard" + window_tag, image_bgr)
    cv2.waitKey()
    cv2.destroyAllWindows()

def load_images(file_list: List) -> List:
    # Cargar todas las imágenes desde una lista de rutas
    return [cv2.imread(file_path) for file_path in file_list]

def get_chessboard_points(board_shape, square_dx, square_dy):
    # Construir los puntos 3D del patrón (z = 0) en el sistema del tablero
    num_cols, num_rows = board_shape
    points_3d = []
    for r in range(num_rows):
        for c in range(num_cols):
            x = c * square_dx
            y = r * square_dy
            z = 0
            points_3d.append([float(x), float(y), float(z)])
    obj_points = np.asarray(points_3d, dtype=np.float32)
    return obj_points.reshape(-1, 1, 3)

def write_image(tag, image_bgr):
    # Guardar imagen en disco (debug/visualización)
    cv2.imwrite(f"drawchessboard_{tag}.jpg", image_bgr)

# --- Carga de imágenes
image_paths = [item for item in glob.glob("data/*.jpg")]
images_bgr = load_images(image_paths)

chess_pattern = (7, 9)  # (cols, rows) esquinas internas

# --- Detección de esquinas
images_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images_bgr]
found_corners = [cv2.findChessboardCorners(img_gray, chess_pattern) for img_gray in images_gray]

# Criterio de parada para cornerSubPix (máx iteraciones y precisión)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

# Refinar solo si se han encontrado
refined_imgpoints = []
for gray_img, detection in zip(images_gray, found_corners):
    was_found, corner_pts = detection
    if was_found:
        refined_pts = cv2.cornerSubPix(gray_img, corner_pts, (7, 9), (-1, -1), subpix_criteria)
        refined_imgpoints.append(refined_pts)
    else:
        # no añadimos nada si no se encuentra
        pass

# Dibujar (opcional)
drawn_images = []
for bgr_img, detection in zip(images_bgr, found_corners):
    was_found, corner_pts = detection
    drawn = bgr_img.copy()
    if was_found:
        cv2.drawChessboardCorners(drawn, chess_pattern, corner_pts, was_found)
    drawn_images.append(drawn)

print("Número de imágenes totales:", len(images_bgr))
print("Número de imágenes válidas:", len(refined_imgpoints))

# --- Puntos 3D (objpoints) alineados con imágenes válidas
board_object_points = get_chessboard_points(chess_pattern, 20, 20)

object_points_list = [board_object_points for _ in range(len(refined_imgpoints))]
image_points_list = refined_imgpoints  # ya son np.float32

# --- Calibración
init_camera_matrix = None
init_dist_coeffs = None

frame_size = (images_bgr[0].shape[1], images_bgr[0].shape[0])  # (width, height) CORRECTO

rms, camera_matrix, distortion_coeffs, rotation_vecs, translation_vecs = cv2.calibrateCamera(
    object_points_list,
    image_points_list,
    frame_size,
    init_camera_matrix,
    init_dist_coeffs,
    criteria=subpix_criteria
)

# Extrínsecas
extrinsic_mats = [
    np.hstack((cv2.Rodrigues(rvec)[0], tvec))
    for rvec, tvec in zip(rotation_vecs, translation_vecs)
]

print("Extrinsics (primeras):", extrinsic_mats[:2])
print("Intrinsics:\n", camera_matrix)
print("Distortion coefficients:\n", distortion_coeffs)
print("Root mean squared reprojection error:\n", rms)
