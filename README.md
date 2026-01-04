# Proyecto Final – Visión por Ordenador  
## Sistema de Contraseña Visual + Tracker 2D tipo Snake (2 jugadores)

Este repositorio contiene el Proyecto Final de la asignatura de Visión por Computador.  
El sistema integra detección de patrones visuales, decodificación automática de una contraseña, tracking por color con Filtro de Kalman (2D) y un juego tipo Snake para dos jugadores.

---

## Descripción general del sistema

El sistema funciona en dos fases principales:

### Fase 1 – Contraseña visual (LOCKED)
El usuario debe introducir una contraseña compuesta por 4 símbolos visuales, detectados mediante cámara:

1. Círculo rojo  
2. Triángulo azul  
3. Cuadrado verde  
4. Línea amarilla  

Cada símbolo se acepta automáticamente cuando:
- es detectado de forma estable durante varios frames, y
- el usuario retira el patrón antes de introducir el siguiente.

Si la secuencia es correcta, el sistema se desbloquea.

---

### Fase 2 – Tracker 2D + Juego Snake (UNLOCKED)

Una vez desbloqueado:
- La pantalla se divide en dos mitades (dos jugadores).
- Cada jugador controla una serpiente usando un marcador de color frente a la cámara:
  - Jugador 1 (izquierda): verde
  - Jugador 2 (derecha): rojo
- Se utiliza un Filtro de Kalman 2D para suavizar el tracking y predecir posiciones cuando se pierde la detección.
- Los jugadores compiten por alcanzar una puntuación objetivo.

Cuando un jugador gana, se muestra un mensaje y se permite:
- empezar una nueva partida, o
- salir del programa.

---

## Estructura del repositorio

ProyectoFinalCV/
├── src/
│   ├── main.py                  # Script principal (contraseña + tracker + juego)
│   ├── color_shape_detector.py  # Detección de color y forma
│   ├── password_decoder.py      # Lógica de decodificación de la contraseña
│   ├── finger_detector.py       # Detector de marcador por color
│   ├── kalman_tracker.py        # Tracker 2D con Filtro de Kalman
│   ├── snake_game.py            # Lógica del juego Snake
│   ├── calibration.py           # Calibración de cámara
│   └── __pycache__/
├── data/                        # Imágenes / datos de calibración
├── assets/                      # Recursos adicionales como el vídeo
├── report/                      # Informe
├── environment_win.yml          # Entorno Conda (Windows)
├── environment_unix.yml         # Entorno Conda (Linux)
├── environment_mac.yml          # Entorno Conda (macOS)
└── README.md                    

---

## Requisitos

- Python 3.9 o superior
- Conda / Miniconda
- Cámara web funcional
- Sistema operativo:
  - Windows
  - Linux
  - macOS

---

## Entorno virtual (Conda)

El proyecto proporciona archivos environment_*.yml con todas las dependencias necesarias.

### Crear el entorno

Selecciona el archivo según tu sistema operativo:

#### Windows
conda env create -f environment_win.yml

#### Linux / Unix
conda env create -f environment_unix.yml

#### macOS
conda env create -f environment_mac.yml

### Activar el entorno
conda activate <nombre_del_entorno>

(El nombre del entorno está definido dentro del archivo .yml)

### Actualizar el entorno (si se modifica el .yml)
conda env update -f environment_win.yml --prune
conda env update -f environment_unix.yml --prune
conda env update -f environment_mac.yml --prune

---

## Ejecución del proyecto

Desde la carpeta src/:
python main.py

---

## Controles

### Fase contraseña (LOCKED)
- r → resetear intento de contraseña
- q → salir

### Fase juego (UNLOCKED)
- r → resetear partida
- k → resetear Kalman
- m → mostrar / ocultar máscaras (debug)
- n → nueva partida (solo cuando hay ganador)
- q → salir

---

## Algoritmos utilizados

- Segmentación por color en espacio HSV
- Clasificación geométrica de contornos
- Decodificación robusta de secuencias visuales
- Filtro de Kalman 2D para tracking
- Procesamiento en tiempo real con OpenCV

---

## Informe

El informe del proyecto se ha realizado en LaTeX e incluye:
- Descripción del sistema
- Metodología y algoritmos
- Diagrama de bloques
- Secuenciación de imágenes
- Implementación
- Resultados experimentales
- Conclusiones

---

## Autores

- Santiago Córdoba Artieda
- Gonzalo García Martínez-Echevarría

Proyecto desarrollado para la asignatura de Visión por Ordenador.

---

## Licencia

Uso académico / educativo.
