# RA - Aplicación de Manipulación de Imágenes

**UNIVERSIDAD POLITECNICA DE CHIAPAS**  
**MULTIMEDIA Y DISEÑO DIGITAL**  
**8° A**  

**DOCENTE:** ELÍAS BELTRÁN NATURI  
**ALUMNO:** LUIS OSVALDO PÉREZ ÁNGEL - 221258

## Descripción

Esta aplicación permite realizar diversas operaciones de procesamiento de imágenes de manera interactiva. A través de una interfaz desarrollada con **Gradio**, los usuarios pueden cargar imágenes y aplicar transformaciones en tiempo real, como ajustes de brillo y contraste, rotación, redimensionamiento, y más. También pueden aplicar filtros personalizados (como mediana, Gaussiano, y bilateral) y combinar imágenes con fondos seleccionados. Además, permite la descarga de las imágenes procesadas de forma rápida y eficiente, lo que facilita un flujo de trabajo sencillo y accesible para todos los usuarios.

### Características principales:
- **Interactividad**: Los usuarios pueden modificar la imagen a través de controles dinámicos.
- **Previsualización en tiempo real**: Ver los efectos de las modificaciones al instante.
- **Descarga de imágenes**: Permite descargar las imágenes procesadas como una imagen final con las modificaciones realizadas.

## Funcionalidades

- **Carga de imágenes**: Selecciona una imagen de una lista predefinida para cargarla y procesarla.
- **Transformaciones de imagen**:  
  - **Rotación**: Rota la imagen según el ángulo especificado.  
  - **Voltear**: Voltea la imagen de manera horizontal o vertical.  
  - **Redimensionar**: Cambia el tamaño de la imagen ajustando el porcentaje de su altura y ancho.
- **Ajustes de brillo y contraste**: Modifica el brillo y el contraste de la imagen para lograr efectos visuales.
- **Filtros personalizados**: Aplica filtros como:
  - **Mediana**: Filtro de mediana para suavizar la imagen.
  - **Gaussiano**: Filtro para desenfoque y suavizado.
  - **Bilateral**: Filtro que preserva los bordes mientras reduce el ruido.
  - **Otros filtros**: Como detección de bordes, enfocar, y realce de detalles.
- **Combina con fondo**: Permite agregar una imagen de fondo seleccionada y combinarla con la imagen procesada.
- **Previsualización**: Muestra la imagen original, transformada y combinada en tiempo real para que el usuario pueda visualizar los resultados antes de descargarla.
- **Descarga de imágenes**: Después de aplicar todas las transformaciones, puedes descargar la imagen final, que incluye las tres versiones (original, transformada, combinada con el fondo).

## Requisitos

Para ejecutar este proyecto, asegúrate de tener las siguientes librerías instaladas:

- **Python**
- **OpenCV**: Biblioteca para procesamiento de imágenes.
- **Gradio**: Herramienta para crear interfaces de usuario interactivas.
- **NumPy**: Librería para el manejo de arreglos y operaciones matemáticas.

### Instalación de dependencias

1. Clona este repositorio o descarga el código fuente.
2. Instala las dependencias con el siguiente comando:

```bash
pip install -r requirements.txt
