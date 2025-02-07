import pickle
import pandas as pd
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

def predecirSigno(imagenCsv=None):
    # Abrir la imagen
    imagenCsv = pd.read_csv("ejemplo/image_1.csv",header=None)
    with open('model/modelo_clasificacion_lenguaje.pkl', 'rb') as modeloAr:
        modeloLenguaje = pickle.load(modeloAr)
    return np.argmax(modeloLenguaje.predict(imagenCsv))

register_heif_opener()

def convertirImagenAArreglo(image_path):
    # Abrir la imagen
    img = Image.open(image_path)

    # Convertirla a blanco y negro
    img = img.convert('L')

    # Redimensionar a 28x28
    img = img.resize((28, 28))

    # Convertir la imagen a un arreglo de Numpy
    img_matriz = np.array(img)

    # Aplanar la matriz 28x28 hacia un arreglo de dimensiones (764,)
    img_arreglo = img_matriz.flatten()

    # Imprimir el shape del arreglo (764,)
    #print(f"Dimensiones arreglo: {img_arreglo.shape}")

    return img_arreglo