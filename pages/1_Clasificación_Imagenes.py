import streamlit as st
import pandas as pd
from modelHelper import predecirSigno
from data.dictSignVsNumber import signVsNumber

def deteccionLetra():
    if 'images' not in st.session_state:
        st.session_state['images'] = dict()

    st.write("## Detección letra en imagen - Lenguage de señas")

    if not st.session_state['images']:
        st.markdown("""
        A continuación puede cargar una imagen con un signo de lenguage de señas, y el modelo de IA construido tratará de deducir la letra correspondiente.
        """)

        uploaded_file = st.file_uploader("Sube una Imagen",  type = ['csv'], accept_multiple_files=False)
    
        if st.button("Deducir letra"):
            placeholder = st.empty()
            with placeholder.container():
                st.write("Leyendo la imagen")
                name = uploaded_file.name
                st.session_state['images'][name] = uploaded_file

            placeholder.empty()
            st.rerun()
    else:
        if len(st.session_state['images'].keys()) > 0:
            for name, file in st.session_state['images'].items():
                imagen = pd.read_csv(file,header=None)
                st.subheader("Análisis de letra completado!")
                st.markdown("La letra que se encuentra en la imagen es: **{}**".format(signVsNumber[predecirSigno(imagen)]))
                st.markdown("Y la imagen correspondiente es:")
                # Mostrar la imagen
                st.image(imagen.to_numpy(copy=True).reshape(28, 28),width=100)
                st.session_state['images'] = dict()
        if st.button("Reiniciar"):
            st.session_state['images'] = dict()
            st.rerun()


st.set_page_config(page_title="Detección letra", page_icon="🤞")
deteccionLetra()