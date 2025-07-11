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
background_images = ["1.jpeg", "2.jpeg", "3.jpeg"]  # Imágenes para fondo

# Cargar los clasificadores Haar cascades pre-entrenados para la detección de objetos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def load_image(image_path):
    img = cv2.imread(image_folder + image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Función para mostrar la imagen cargada
def display_image(image_path=None, image=None):
    if image_path:
        img = load_image(image_path)
    elif image is not None:
        img = image
    return img

# Función para convertir entre espacios de color
def convert_color(image, color_space):
    if color_space == "BGR to RGB":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_space == "RGB to BGR":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_space == "RGB to HSV":
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif color_space == "HSV to RGB":
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    elif color_space == "RGB to LAB":
        return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    elif color_space == "LAB to RGB":
        return cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    return image

# Función para aplicar transformaciones a la imagen
def process_image(image, rotation=0, flip=0, resize_percentage_width=100, resize_percentage_height=100, brightness=0, contrast=1, filter_type="Mediana", gamma=1.0):
    # Aplicar rotación
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows))

    # Voltear la imagen
    flipped = cv2.flip(rotated, flip)

    # Redimensionar la imagen por porcentaje
    new_width = int(cols * (resize_percentage_width / 100))
    new_height = int(rows * (resize_percentage_height / 100))
    resized = cv2.resize(flipped, (new_width, new_height))

    # Ajuste de brillo y contraste
    adjusted = cv2.convertScaleAbs(resized, alpha=contrast, beta=brightness)

    # Corrección Gamma
    gamma_corrected = np.array(255 * (adjusted / 255) ** gamma, dtype='uint8')

    # Asegurarse de que los valores estén dentro del rango [0, 255]
    gamma_corrected = np.clip(gamma_corrected, 0, 255)

    # Inicialización de la variable filtered para prevenir UnboundLocalError
    filtered = gamma_corrected

    # Aplicación de filtros
    if filter_type == "Mediana":
        filtered = cv2.medianBlur(gamma_corrected, 5)
    elif filter_type == "Gaussiano":
        filtered = cv2.GaussianBlur(gamma_corrected, (5, 5), 0)
    elif filter_type == "Bilateral":
        filtered = cv2.bilateralFilter(gamma_corrected, 9, 75, 75)
    elif filter_type == "Detección de bordes":
        # Detección de bordes con Canny
        gray = cv2.cvtColor(gamma_corrected, cv2.COLOR_RGB2GRAY)
        filtered = cv2.Canny(gray, 100, 200)
        # Convertir la imagen a RGB para evitar problemas de canales al combinarla con el fondo
        filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
    elif filter_type == "Enfocar":
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        filtered = cv2.filter2D(gamma_corrected, -1, kernel)
    elif filter_type == "Media":
        filtered = cv2.blur(gamma_corrected, (5, 5))
    elif filter_type == "Sobel":
        grad_x = cv2.Sobel(gamma_corrected, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gamma_corrected, cv2.CV_64F, 0, 1, ksize=3)
        filtered = cv2.magnitude(grad_x, grad_y)
        filtered = cv2.convertScaleAbs(filtered)  # Convertir los valores a uint8
    elif filter_type == "Scharr":
        grad_x = cv2.Scharr(gamma_corrected, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(gamma_corrected, cv2.CV_64F, 0, 1)
        filtered = cv2.magnitude(grad_x, grad_y)
        filtered = cv2.convertScaleAbs(filtered)  # Convertir los valores a uint8
    elif filter_type == "Dilatación":
        kernel = np.ones((5, 5), np.uint8)
        filtered = cv2.dilate(gamma_corrected, kernel, iterations=1)
    elif filter_type == "Erosión":
        kernel = np.ones((5, 5), np.uint8)
        filtered = cv2.erode(gamma_corrected, kernel, iterations=1)
    elif filter_type == "Contornos":
        gray = cv2.cvtColor(gamma_corrected, cv2.COLOR_RGB2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = cv2.drawContours(gamma_corrected.copy(), contours, -1, (0, 255, 0), 3)
    elif filter_type == "Umbral":
        _, filtered = cv2.threshold(gamma_corrected, 127, 255, cv2.THRESH_BINARY)
    
    # Ecualización de Histograma
    if filter_type == "Ecualización":
        gray_image = cv2.cvtColor(gamma_corrected, cv2.COLOR_RGB2GRAY)
        filtered = cv2.equalizeHist(gray_image)

    # Mejoramiento de detalles (realce)
    if filter_type == "Realce":
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        filtered = cv2.filter2D(gamma_corrected, -1, kernel)

    return filtered

# Función para guardar la imagen procesada en la carpeta local con nombre único
def save_image(image, filename="processed_image.jpg"):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("descargas", f"{timestamp}_{filename}")
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return output_path

# Función para combinar la imagen procesada con la imagen de fondo seleccionada
def combine_with_background(processed_image, background_image_path):
    background = cv2.imread(image_folder + background_image_path)
    background_resized = cv2.resize(background, (processed_image.shape[1], processed_image.shape[0]))
    combined = cv2.addWeighted(processed_image, 1, background_resized, 0.5, 0)
    return combined

# Función para detectar objetos con Haar cascades
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

def detect_eyes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    eyes = eye_cascade.detectMultiScale(gray)
    for (x, y, w, h) in eyes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image

# Función para calcular detalles estadísticos de la imagen
def image_statistics(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    mean = np.mean(gray)
    std_dev = np.std(gray)
    return hist, mean, std_dev

# Función para realizar segmentación avanzada con GrabCut
def grabcut_segmentation(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convertir a BGR
    mask = np.zeros(img.shape[:2], np.uint8)  # Máscara inicial
    background = np.zeros_like(img, np.uint8)  # Fondo inicial
    foreground = np.zeros_like(img, np.uint8)  # Primer plano inicial
    
    # Rectángulo inicial para definir el área de interés
    rect = (10, 10, img.shape[1]-10, img.shape[0]-10)
    
    # El modelo debe ser inicializado, crea un modelo vacío
    model = np.zeros((1, 65), np.float64)
    
    # Aplicar GrabCut con inicialización del rectángulo
    cv2.grabCut(img, mask, rect, background, foreground, 5, cv2.GC_INIT_WITH_RECT)
    
    # Cambiar el valor de la máscara para mejorar la segmentación
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Función para descargar la imagen combinada en un solo collage
def download_images(original_image, processed_image, background_image):
    processed_resized = cv2.resize(processed_image, (original_image.shape[1], original_image.shape[0]))
    combined_image = combine_with_background(processed_image, background_image)
    
    # Crear collage final con las tres imágenes
    collage = np.hstack((original_image, processed_resized, combined_image))
    
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
        # Controles para modificar la imagen
        rotation_slider = gr.Slider(-180, 180, step=1, label="Rotación de la imagen")
        flip_slider = gr.Slider(0, 1, step=1, label="Voltear la imagen (0: Horizontal, 1: Vertical)")
        resize_percentage_width_slider = gr.Slider(10, 200, step=1, label="Redimensionar por porcentaje de Ancho (%)", value=100)
        resize_percentage_height_slider = gr.Slider(10, 200, step=1, label="Redimensionar por porcentaje de Altura (%)", value=100)
        brightness_slider = gr.Slider(-100, 100, step=1, label="Brillo de la imagen")
        contrast_slider = gr.Slider(1, 3, step=0.1, label="Contraste de la imagen")
        filter_dropdown = gr.Dropdown(choices=["Imagen Original", "Mediana", "Gaussiano", "Bilateral", "Detección de bordes", "Enfocar", "Media", "Sobel", "Scharr", "Dilatación", "Erosión", "Contornos", "Umbral", "Ecualización", "Realce", "GrabCut", "Detección de rostros", "Detección de ojos"], label="Selecciona un filtro")
        gamma_slider = gr.Slider(0.1, 3.0, step=0.1, label="Corrección Gamma")
        
        # Nuevo dropdown para selección de espacio de color
        color_space_dropdown = gr.Dropdown(choices=["Ninguno", "BGR to RGB", "RGB to BGR", "RGB to HSV", "HSV to RGB", "RGB to LAB", "LAB to RGB"], label="Selecciona un espacio de color", value="Ninguno")

    with gr.Row():
        # Mostrar la imagen transformada antes de combinarla
        transformed_image_display = gr.Image(label="Imagen Transformada")

    with gr.Row():
        # Mostrar la imagen combinada con fondo
        combined_image_display = gr.Image(label="Imagen Combinada")

    with gr.Row():
        # Campo para seleccionar la imagen de fondo (opcional) en la parte inferior
        background_dropdown = gr.Dropdown(choices=background_images, label="Selecciona una imagen de fondo", interactive=True, value=None)

    with gr.Row():
        # Botón para descargar las imágenes (original + procesada)
        download_button = gr.Button("Descargar Imágenes")
        download_output = gr.File(label="Descargar Imágenes", visible=False)

    # Lógica para cargar la imagen y aplicar transformaciones
    def apply_transformations(image_path, rotation, flip, resize_percentage_width, resize_percentage_height, brightness, contrast, filter_type, gamma, background_image, color_space):
        image = load_image(image_path)
        
        # Si se selecciona "Imagen Original", no aplicar ninguna transformación
        if filter_type != "Imagen Original":
            transformed_image = process_image(image, rotation, flip, resize_percentage_width, resize_percentage_height, brightness, contrast, filter_type, gamma)
        else:
            transformed_image = image
        
        # Aplicar la conversión de color automáticamente, si el usuario elige transformaciones que lo requieran
        if color_space != "Ninguno":
            transformed_image = convert_color(transformed_image, color_space) 
        
        # Si se selecciona un fondo, combinarlo con la imagen transformada
        if background_image:
            combined_image = combine_with_background(transformed_image, background_image)
        else:
            combined_image = transformed_image
        
        # Si se selecciona la segmentación GrabCut
        if filter_type == "GrabCut":
            transformed_image = grabcut_segmentation(transformed_image)
        
        # Si se selecciona la detección de objetos
        if filter_type == "Detección de rostros":
            transformed_image = detect_faces(transformed_image)
        
        if filter_type == "Detección de ojos":
            transformed_image = detect_eyes(transformed_image)
        
        return image, transformed_image, combined_image

    # Lógica para descarga de las imágenes (incluyendo collage)
    def download_file(original_image, processed_image, background_image):
        # Crear collage y devolver el archivo guardado
        saved_image_path = download_images(original_image, processed_image, background_image)
        return saved_image_path

    # Enlazar los eventos
    image_input.change(display_image, inputs=image_input, outputs=image_display)
    rotation_slider.change(apply_transformations, 
                           inputs=[image_input, rotation_slider, flip_slider, resize_percentage_width_slider, resize_percentage_height_slider, brightness_slider, contrast_slider, filter_dropdown, gamma_slider, background_dropdown, color_space_dropdown], 
                           outputs=[image_display, transformed_image_display, combined_image_display])
    flip_slider.change(apply_transformations, 
                       inputs=[image_input, rotation_slider, flip_slider, resize_percentage_width_slider, resize_percentage_height_slider, brightness_slider, contrast_slider, filter_dropdown, gamma_slider, background_dropdown, color_space_dropdown], 
                       outputs=[image_display, transformed_image_display, combined_image_display])
    resize_percentage_width_slider.change(apply_transformations, 
                                           inputs=[image_input, rotation_slider, flip_slider, resize_percentage_width_slider, resize_percentage_height_slider, brightness_slider, contrast_slider, filter_dropdown, gamma_slider, background_dropdown, color_space_dropdown], 
                                           outputs=[image_display, transformed_image_display, combined_image_display])
    resize_percentage_height_slider.change(apply_transformations, 
                                            inputs=[image_input, rotation_slider, flip_slider, resize_percentage_width_slider, resize_percentage_height_slider, brightness_slider, contrast_slider, filter_dropdown, gamma_slider, background_dropdown, color_space_dropdown], 
                                            outputs=[image_display, transformed_image_display, combined_image_display])
    brightness_slider.change(apply_transformations, 
                             inputs=[image_input, rotation_slider, flip_slider, resize_percentage_width_slider, resize_percentage_height_slider, brightness_slider, contrast_slider, filter_dropdown, gamma_slider, background_dropdown, color_space_dropdown], 
                             outputs=[image_display, transformed_image_display, combined_image_display])
    contrast_slider.change(apply_transformations, 
                           inputs=[image_input, rotation_slider, flip_slider, resize_percentage_width_slider, resize_percentage_height_slider, brightness_slider, contrast_slider, filter_dropdown, gamma_slider, background_dropdown, color_space_dropdown], 
                           outputs=[image_display, transformed_image_display, combined_image_display])
    filter_dropdown.change(apply_transformations, 
                           inputs=[image_input, rotation_slider, flip_slider, resize_percentage_width_slider, resize_percentage_height_slider, brightness_slider, contrast_slider, filter_dropdown, gamma_slider, background_dropdown, color_space_dropdown], 
                           outputs=[image_display, transformed_image_display, combined_image_display])
    gamma_slider.change(apply_transformations, 
                        inputs=[image_input, rotation_slider, flip_slider, resize_percentage_width_slider, resize_percentage_height_slider, brightness_slider, contrast_slider, filter_dropdown, gamma_slider, background_dropdown, color_space_dropdown], 
                        outputs=[image_display, transformed_image_display, combined_image_display])
    download_button.click(download_file, inputs=[image_display, transformed_image_display, background_dropdown], outputs=download_output)


demo.launch(server_name="127.0.0.1", server_port=7860)
