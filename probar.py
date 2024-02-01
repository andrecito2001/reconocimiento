from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Cargar el modelo entrenado
model = load_model('model')

# Función para cargar y preprocesar una imagen dada su ruta
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Función para realizar la inferencia y mostrar la categoría predicha
def predict_category(img_path):
    # Cargar y preprocesar la imagen
    img_array = load_and_preprocess_image(img_path)

    # Realizar la inferencia
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    # Mostrar resultados
    print(f'Predicted Category: {predicted_label}')

    # Mostrar la imagen
    img = Image.open(img_path)
    img = img.resize((224, 224))  # Ajustar tamaño para visualización
    img.show()

# Crear una interfaz gráfica simple para seleccionar una imagen
root = tk.Tk()
root.withdraw()  # Ocultar la ventana principal

# Permitir al usuario seleccionar una imagen
file_path = filedialog.askopenfilename(title="prueba", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

if file_path:
    # Realizar la predicción y mostrar la categoría
    predict_category(file_path)
else:
    print("No image selected.")
