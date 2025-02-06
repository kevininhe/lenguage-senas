import pickle
import pandas as pd
import numpy as np

def predecirSigno(imagenCsv=None):
    imagenCsv = pd.read_csv("ejemplo/image_1.csv",header=None)
    with open('model/modelo_clasificacion_lenguaje.pkl', 'rb') as modeloAr:
        modeloLenguaje = pickle.load(modeloAr)
    return np.argmax(modeloLenguaje.predict(imagenCsv))