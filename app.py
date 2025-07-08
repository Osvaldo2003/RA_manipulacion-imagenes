import cv2
import gradio as gr
import numpy as np
import os
import time

# Crear carpeta "descargas" si no existe
if not os.path.exists("descargas"):
    os.makedirs("descargas")

# Cargar imágenes desde una carpeta preestablecida
image_folder = "images/"
image_files = ["luis.jpg", "LUIS2.jpg", "LUIS3.jpg", "LUIS4.jpg"]  # Lista de imágenes

def load_image(image_path):
    img = cv2.imread(image_folder + image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def process_image(image, rotation=0, flip=0, resize=(200, 200), brightness=0, contrast=1, filter_type="Mediana"):
    # Aplicar rotación
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows))

    # Voltear la imagen
    flipped = cv2.flip(rotated, flip)

    # Redimensionar la imagen
    resized = cv2.resize(flipped, resize)

    # Ajuste de brillo y contraste
    adjusted = cv2.convertScaleAbs(resized, alpha=contrast, beta=brightness)

    # Aplicación de filtros
    if filter_type == "Mediana":
        filtered = cv2.medianBlur(adjusted, 5)
    elif filter_type == "Gaussiano":
        filtered = cv2.GaussianBlur(adjusted, (5, 5), 0)
    elif filter_type == "Bilateral":
        filtered = cv2.bilateralFilter(adjusted, 9, 75, 75)
    elif filter_type == "Detección de bordes":
        filtered = cv2.Canny(adjusted, 100, 200)
    elif filter_type == "Enfocar":
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        filtered = cv2.filter2D(adjusted, -1, kernel)

    return filtered

# Función para mostrar la imagen cargada
def display_image(image_path=None, image=None):
    if image_path:
        img = load_image(image_path)
    elif image is not None:
        img = image
    return img

# Función para guardar la imagen procesada en la carpeta local con nombre único
def save_image(image, filename="processed_image.jpg"):
    # Generar un nombre único para cada descarga
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("descargas", f"{timestamp}_{filename}")
    
    # Guardar la imagen procesada
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return output_path

# Función para crear un collage de ambas imágenes
def create_collage(original_image, processed_image):
    # Redimensionar ambas imágenes al mismo tamaño
    processed_resized = cv2.resize(processed_image, (original_image.shape[1], original_image.shape[0]))
    # Apilar las imágenes en una fila (horizontalmente)
    collage = np.hstack((original_image, processed_resized))
    return collage

# Función para descargar ambas imágenes y generar un nombre único
def download_images(original_image, processed_image):
    # Crear collage de las imágenes
    collage = create_collage(original_image, processed_image)
    
    # Guardar la imagen combinada (collage) con un nombre único
    output_path = save_image(collage, "collage_imagenes.jpg")
    
    return output_path

# Interfaz de Gradio
with gr.Blocks() as demo:
    with gr.Row():
        # Sección de entrada para elegir la imagen
        image_input = gr.Dropdown(choices=image_files, label="Selecciona una imagen", interactive=True, value=None)
        image_display = gr.Image(label="Imagen Original", elem_id="image_display")

    with gr.Row():
        # Botón para cargar y mostrar la imagen seleccionada
        load_button = gr.Button("Cargar Imagen para Transformar")

    with gr.Row():
        # Controles para modificar la imagen
        rotation_slider = gr.Slider(-180, 180, step=1, label="Rotación de la imagen")
        flip_slider = gr.Slider(0, 1, step=1, label="Voltear la imagen (0: Horizontal, 1: Vertical)")
        resize_width = gr.Slider(50, 500, step=10, label="Redimensionar (Ancho)")
        resize_height = gr.Slider(50, 500, step=10, label="Redimensionar (Alto)")
        brightness_slider = gr.Slider(-100, 100, step=1, label="Brillo de la imagen")
        contrast_slider = gr.Slider(1, 3, step=0.1, label="Contraste de la imagen")
        filter_dropdown = gr.Dropdown(choices=["Imagen Original", "Mediana", "Gaussiano", "Bilateral", "Detección de bordes", "Enfocar"], label="Selecciona un filtro")

    with gr.Row():
        # Botón para aplicar transformaciones
        apply_button = gr.Button("Aplicar Transformaciones")

    with gr.Row():
        # Mostrar la imagen procesada
        processed_image = gr.Image(label="Imagen Procesada")

    with gr.Row():
        # Botón para descargar las imágenes (original + procesada)
        download_button = gr.Button("Descargar Imágenes")
        download_output = gr.File(label="Descargar Imágenes", visible=False)  # Usamos el componente gr.File para la descarga

    # Lógica para cargar la imagen y aplicar transformaciones
    def apply_transformations(image_path, rotation, flip, resize_width, resize_height, brightness, contrast, filter_type):
        image = load_image(image_path)
        
        # Si se selecciona "Imagen Original", no aplicar ninguna transformación
        if filter_type == "Imagen Original":
            transformed_image = image
        else:
            transformed_image = process_image(image, rotation, flip, (resize_width, resize_height), brightness, contrast, filter_type)
        
        return image, transformed_image  # Devolver tanto la imagen original como la procesada

    # Lógica para descarga de ambas imágenes (incluyendo collage)
    def download_file(original_image, processed_image):
        # Crear collage y devolver el archivo guardado
        saved_image_path = download_images(original_image, processed_image)
        return saved_image_path

    # Enlazar los eventos
    image_input.change(display_image, inputs=image_input, outputs=image_display)
    load_button.click(display_image, inputs=image_input, outputs=image_display)  # Cargar automáticamente la imagen cuando se presione el botón
    apply_button.click(apply_transformations, 
                       inputs=[image_input, rotation_slider, flip_slider, resize_width, resize_height, brightness_slider, contrast_slider, filter_dropdown], 
                       outputs=[image_display, processed_image])
    download_button.click(download_file, inputs=[image_display, processed_image], outputs=download_output)

# Iniciar la aplicación
demo.launch(server_name="127.0.0.1", server_port=7860)
