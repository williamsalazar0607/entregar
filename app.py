%%writefile app.py
import streamlit as st
import pandas as pd
import joblib

# Configuración de la página
st.set_page_config(page_title="Predicción de Precio de Gasolina", layout="centered")

# Título y descripción
st.title("🚗 Predicción de Precio de Gasolina (USD/Galón)")
st.markdown("Esta aplicación utiliza un modelo de Regresión Lineal para estimar el precio de la gasolina basado en 7 variables clave.")

# Cargar el modelo guardado
@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_gasolina.pkl')

try:
    model = cargar_modelo()
    
    st.sidebar.header("Ingreso de Variables")
    
    # Crear inputs para cada una de las 7 variables con valores por defecto similares a los del dataset
    precio_petroleo = st.sidebar.slider("Precio del Petróleo (USD/Barril)", min_value=20.0, max_value=150.0, value=75.0, step=1.0)
    demanda = st.sidebar.number_input("Demanda (miles de barriles/día)", min_value=1000.0, max_value=20000.0, value=9500.0, step=50.0)
    pib = st.sidebar.slider("Crecimiento del PIB (%)", min_value=-10.0, max_value=15.0, value=3.5, step=0.1)
    impuestos = st.sidebar.slider("Impuestos a Combustibles (%)", min_value=0.0, max_value=50.0, value=19.0, step=0.5)
    
    eventos_geopoliticos = st.sidebar.selectbox("Eventos Geopolíticos Activos", options=[0, 1], format_func=lambda x: "Sí (1)" if x == 1 else "No (0)", index=1)
    
    tipo_cambio = st.sidebar.number_input("Tipo de Cambio (USD/COP)", min_value=1500.0, max_value=6000.0, value=4000.0, step=10.0)
    inflacion = st.sidebar.slider("Inflación (%)", min_value=-2.0, max_value=30.0, value=5.0, step=0.1)
    
    # Crear DataFrame con los datos de entrada en el orden correcto de columnas
    datos_entrada = pd.DataFrame({
        'Precio_Petroleo_USD_Barril': [precio_petroleo],
        'Demanda_miles_barriles_dia': [demanda],
        'PIB_Porcentaje': [pib],
        'Impuestos_Combustibles_Porcentaje': [impuestos],
        'Eventos_Geopoliticos': [eventos_geopoliticos],
        'Tipo_Cambio_USD_COP': [tipo_cambio],
        'Inflacion_Porcentaje': [inflacion]
    })
    
    # Mostrar datos ingresados en el cuerpo principal
    st.subheader("Datos de entrada seleccionados:")
    st.dataframe(datos_entrada)
    
    # Botón de predicción
    if st.button("⚡ Calcular Precio Predicho"):
        precio_predicho = model.predict(datos_entrada)[0]
        
        st.success(f"### 📊 El precio predicho es: **${precio_predicho:.2f} USD** por galón")
        
except FileNotFoundError:
    st.error("❌ No se encontró el archivo 'modelo_gasolina.pkl'. Asegúrate de ejecutar primero la celda de entrenamiento para generar el modelo.")
except Exception as e:
    st.error(f"Ocurrió un error inesperado: {e}")
