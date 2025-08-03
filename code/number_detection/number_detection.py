import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense

# 🧠 Definir el modelo LeNet-5
def build_lenet5():
    model = Sequential([
        Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
        AveragePooling2D(pool_size=(2, 2)),
        Conv2D(16, kernel_size=(5, 5), activation='relu'),
        AveragePooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(120, activation='relu'),
        Dense(84, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

# 📷 Preprocesar imagen
def preprocess_image(image_path, show=False):
    img = Image.open(image_path).convert('L')  # Escala de grises
    img = img.resize((28, 28))
    img_array = np.array(img)

    # ⚫⚪ Binarización: negro si <128, blanco si ≥128
    img_array = np.where(img_array < 128, 0, 255)

    # 🔢 Normalizar a [0, 1]
    img_array = img_array / 255.0

    # 🌓 Invertir colores si el fondo es blanco
    img_array = 255 - img_array

    if show:
        plt.imshow(img_array, cmap='gray')
        plt.title("🖼 Imagen Binarizada (Negro/Blanco)")
        plt.axis('off')
        plt.show()

    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array



# 🔄 Cargar modelo y pesos
model = build_lenet5()
model.load_weights("lenet5_mnist.weights.h5")  # Asegúrate de tener este archivo

# 📁 Ruta de la imagen
image_path = "1.jpg"

# 🔍 Predecir
img = preprocess_image(image_path, show=True)  # Mostrar imagen preprocesada
pred = model.predict(img)
prediction = np.argmax(pred)

print(f"🧠 Predicción del dígito: {prediction}")
