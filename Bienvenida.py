import streamlit as st

st.set_page_config(
    page_title="Bienvenida",
    page_icon="👋",
)

st.title("Detección de señas en imágenes a blanco y negro")

st.write(
    "En este aplicativo se utilizará un modelo de IA de detección de imágenes de lenguage de señas para establecer la letra que se está comunicando en la imagen. Las letras que se detectan son las siguientes:"
)

st.image(image="images/signLanguage.png")
st.markdown("Tomado de https://www.kaggle.com/datasets/datamunge/sign-language-mnist")
