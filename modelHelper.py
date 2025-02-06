import pickle
import pandas as pd
import numpy as np

def predecirSigno(imagenCsv):
    with open('model/modelo_clasificacion_lenguaje.pkl', 'rb') as modeloAr:
        modeloLenguaje = pickle.load(modeloAr)
    return np.argmax(modeloLenguaje.predict(imagenCsv))
