from typing import List
import numpy as np
import cv2
import copy  
import glob

def show_image(name,img):
    cv2.imshow("drawchessboard"+name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def load_images(filenames: List) -> List:
    return [cv2.imread(filename) for filename in filenames]

def get_chessboard_points(chessboard_shape, dx, dy):
    cols, rows = chessboard_shape
    vector = []
    for row in range(rows):
        for col in range(cols):
            x = col * dx
            y = row * dy
            z = 0
            vector.append([float(x), float(y), float(z)])
    objp = np.asarray(vector, dtype=np.float32)
    return objp.reshape(-1, 1, 3)

def write_image(name, img):
    cv2.imwrite(f"drawchessboard_{name}.jpg", img)

# --- Carga de imágenes
imgs_path = [item for item in glob.glob("data/*.jpg")]
imgs = load_images(imgs_path)

pattern_size = (7, 9)  # (cols, rows) esquinas internas

# --- Detección de esquinas
imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
corners = [cv2.findChessboardCorners(img_gray, pattern_size) for img_gray in imgs_gray]

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

# Refinar solo si se han encontrado
corners_refined = []
for img_gray, cor in zip(imgs_gray, corners):
    found, pts = cor
    if found:
        pts_refined = cv2.cornerSubPix(img_gray, pts, (7, 9), (-1, -1), criteria)
        corners_refined.append(pts_refined)
    else:
        # no añadimos nada si no se encuentra
        pass

# Dibujar (opcional)
imgs_draw = []
for img, cor in zip(imgs, corners):
    found, pts = cor
    img_draw = img.copy()
    if found:
        cv2.drawChessboardCorners(img_draw, pattern_size, pts, found)
    imgs_draw.append(img_draw)

print("Número de imágenes totales:", len(imgs))
print("Número de imágenes válidas:", len(corners_refined))

# --- Puntos 3D (objpoints) alineados con imágenes válidas
chessboard_points = get_chessboard_points(pattern_size, 20, 20)

objpoints = [chessboard_points for _ in range(len(corners_refined))]
imgpoints = corners_refined  # ya son np.float32

# --- Calibración
cameraMat_init = None
distcoef_init = None

image_size = (imgs[0].shape[1], imgs[0].shape[0])  # (width, height) CORRECTO

rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    image_size,
    cameraMat_init,
    distcoef_init,
    criteria=criteria
)

# Extrínsecas
extrinsics = [
    np.hstack((cv2.Rodrigues(rvec)[0], tvec))
    for rvec, tvec in zip(rvecs, tvecs)
]

print("Extrinsics (primeras):", extrinsics[:2])
print("Intrinsics:\n", intrinsics)
print("Distortion coefficients:\n", dist_coeffs)
print("Root mean squared reprojection error:\n", rms)
