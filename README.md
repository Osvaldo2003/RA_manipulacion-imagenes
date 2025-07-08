# RA- Aplicación de manipulación de imagenes

UNIVERSIDAD POLITECNICA DE CHIAPAS


MULTIMEDIA Y DISEÑO DIGITAL 

8° A

DOCENTE:ELIAS BELTRAN NATURI

ALUMNO: LUIS OSVALDO PÉREZ ÁNGEL - 221258


## Descripción

Esta aplicación permite realizar diversas operaciones de procesamiento de imágenes de manera interactiva, como conversión de espacios de color, ajuste de brillo y contraste, detección de rostros, detección de contornos, entre otras transformaciones. Los usuarios pueden aplicar filtros personalizados y ver los resultados en tiempo real. La interfaz, desarrollada con Gradio, facilita la carga y visualización de imágenes, así como su modificación a través de controles dinámicos e intuitivos. Además, permite la descarga de las imágenes procesadas de forma rápida y eficiente, lo que hace que el flujo de trabajo sea sencillo y accesible para todos los usuarios.


## Funcionalidades
- **Carga de imágenes**: Selecciona una imagen de la lista para procesarla.
- **Conversión de color**: Convierte entre diferentes espacios de color (RGB, BGR, HSV, etc.).
- **Ajuste de brillo y contraste**: Modifica el brillo y contraste de la imagen.
- **Aplicación de filtros**: Aplica filtros como mediana, Gaussiano y bilateral.
- **Eliminación de fondo**: Utiliza segmentación para eliminar fondos de la imagen.
- **Detección de rostros**: Usa Haar Cascades para detectar rostros en las imágenes.
- **Detección de contornos**: Detecta contornos utilizando la detección de bordes de Canny.

## Requisitos
Para ejecutar este proyecto, asegúrate de tener las siguientes librerías instaladas:

- Python
- OpenCV
- Gradio
- NumPy

Instalar las dependencias:

```bash
pip install -r requirements.txt
