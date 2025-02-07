import pickle
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from PIL import ImageOps

def predecirSigno(imagenCsv):
    # Abrir el archivo en el que se encuentra guardado el modelo de IA, y guardarlo en una variable
    with open('model/modelo_clasificacion_lenguaje.pkl', 'rb') as modeloAr:
        modeloLenguaje = pickle.load(modeloAr)
    
    # Usar el modelo para hacer una predicción
    return np.argmax(modeloLenguaje.predict(imagenCsv))

register_heif_opener()

def convertirImagenAArreglo(image_path):
    # Abrir la imagen
    img = Image.open(image_path)

    # Quite el tag de orientación EXIF, para que las fotos subidas desde el celular no se roten
    img = ImageOps.exif_transpose(img)

    # Convertirla a blanco y negro
    img = img.convert('L')

    # Redimensionar a 28x28
    img = img.resize((28, 28))

    # Convertir la imagen a un arreglo de Numpy
    img_matriz = np.array(img)

    # Aplanar la matriz 28x28 hacia un arreglo de dimensiones (784,)
    img_arreglo = img_matriz.flatten()

    return img_arreglo
