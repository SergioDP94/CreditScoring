import numpy as np
import pickle
import streamlit as st


# Aplicación Web con Streamlit
st.title("Predicción de Riesgo de Crédito")
edad_input = st.number_input("Edad", min_value=18, max_value=70, value=30)
experiencia_input = st.number_input("Años de Experiencia", min_value=0, max_value=52, value=5)
ingresos_input = st.number_input("Ingresos en Soles", min_value=1025, value=3000)

if st.button("Predecir Riesgo"):
    with open("modelo_riesgo.pkl", "rb") as f:
        modelo_cargado = pickle.load(f)
    datos_usuario = np.array([[edad_input, experiencia_input, ingresos_input]])
    probabilidad_mora = modelo_cargado.predict_proba(datos_usuario)[:, 1][0]
    st.write(f"Probabilidad de impago: {probabilidad_mora:.2%}")
