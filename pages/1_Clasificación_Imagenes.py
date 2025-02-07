import streamlit as st
import pandas as pd
from modelHelper import predecirSigno, convertirImagenAArreglo
# Ya que el modelo retorna un número, mapee el número con la letra correspondiente
from data.dictSignVsNumber import signVsNumber

def deteccionLetra():
    if 'images' not in st.session_state:
        st.session_state['images'] = dict()

    st.write("## Detección letra en imagen - Lenguage de señas")

    # Si no hay ninguna imagen cargada, muestre este contenido
    if not st.session_state['images']:
        st.markdown("""
        A continuación puede cargar una imagen con un signo de lenguage de señas, y el modelo de IA construido tratará de deducir la letra correspondiente.
        """)

        uploaded_file = st.file_uploader("Sube una Imagen",  type = ['csv', 'jpg', 'png', 'HEIC'], accept_multiple_files=False)
    
        if st.button("Deducir letra"):
            placeholder = st.empty()
            with placeholder.container():
                st.write("Leyendo la imagen")
                name = uploaded_file.name
                # Guarde la imagen en una variable, la imagen se procesa luego de ser guardada
                st.session_state['images'][name] = uploaded_file

            placeholder.empty()
            st.rerun()

    # Si hay una imagen cargada, muestre este otro contenido
    else:
        if len(st.session_state['images'].keys()) > 0:
            # Itere en las imágenes guardadas
            for name, file in st.session_state['images'].items():
                imagen = None
                # Si es csv, cree el Dataframe de Pandas directamente desde el csv
                if "csv" in file.type:
                    imagen = pd.read_csv(file,header=None)
                # Si es una imagen, conviertala a un arreglo y luego a un dataframe de Pandas
                else:
                    imagen = convertirImagenAArreglo(file)
                    imagen = pd.DataFrame(imagen.reshape(1, 784))

                st.subheader("Análisis de letra completado!")
                st.markdown("La letra que se encuentra en la imagen es: **{}**".format(signVsNumber[predecirSigno(imagen)]))
                st.markdown("Y la imagen correspondiente es:")

                # Mostrar la imagen, convirtiendo el arreglo en matriz
                st.image(imagen.to_numpy(copy=True).reshape(28, 28),width=100)
                st.session_state['images'] = dict()
        if st.button("Reiniciar"):
            # La imagen se había guardado en una variable. Limpie la variable
            st.session_state['images'] = dict()
            st.rerun()

st.set_page_config(page_title="Detección letra", page_icon="🤞")
deteccionLetra()