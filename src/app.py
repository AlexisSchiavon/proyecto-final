import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.graph_objects as go
import requests
import os
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Productos Bancarios",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🏦 Predictor de Productos Bancarios</h1>', unsafe_allow_html=True)

# URL del modelo
MODEL_URL = "https://www.dropbox.com/scl/fi/89xr0jlb76brekrxopxz5/modelo_xgb_optimizado_sin_sobreajuste.pkl?rlkey=te54yymfwqyrldy5kxu70nyds&st=zv2li6u7&dl=1"
MODEL_FILENAME = "modelo_xgb_optimizado_sin_sobreajuste_prueba1.pkl"

# Función para descargar modelo
@st.cache_data
def download_model():
    """Descarga el modelo si no existe"""
    if not os.path.exists(MODEL_FILENAME):
        st.info("⬇️ Descargando modelo por primera vez...")
        
        response = requests.get(MODEL_URL)
        response.raise_for_status()
        
        with open(MODEL_FILENAME, 'wb') as f:
            f.write(response.content)
        
        st.success("✅ Modelo descargado!")
    
    return MODEL_FILENAME

# Función para cargar modelo
@st.cache_resource
def load_model():
    """Carga el modelo y parcha errores de deserialización"""
    filename = download_model()

    # Hotfix para el error de _PredictScorer
    import sklearn.metrics._scorer
    if not hasattr(sklearn.metrics._scorer, '_PredictScorer'):
        class _PredictScorer:
            pass
        sklearn.metrics._scorer._PredictScorer = _PredictScorer

    with open(filename, 'rb') as f:
        modelo_completo = joblib.load(f)

    return modelo_completo

# Cargar modelo
modelo_completo = load_model()

# Resto del código igual...
st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.markdown("**📋 Información del Modelo:**")
st.markdown(f"- **Algoritmo:** XGBoost Multi-Output Optimizado")
st.markdown(f"- **Productos a predecir:** Tarjeta de Crédito, E-Cuenta, Depósito a Largo Plazo")
st.markdown(f"- **Score F1:** {modelo_completo.get('mejor_score', 'N/A')}")
st.markdown('</div>', unsafe_allow_html=True)

# Formulario de entrada
st.sidebar.header("📊 Datos del Cliente")

# Información demográfica
st.sidebar.subheader("Información Personal")
sexo = st.sidebar.selectbox("Sexo", ['Masculino', 'Femenino'])
age = st.sidebar.slider("Edad", 18, 100, 35)
antiguedad = st.sidebar.slider("Antigüedad (meses)", 0, 300, 60)
renta = st.sidebar.number_input("Renta mensual", 0, 200000, 30000, step=1000)

# Canal y segmento
st.sidebar.subheader("Información Comercial")
canal_entrada = st.sidebar.selectbox(
    "Canal de Entrada", 
    ['Online', 'Oficina', 'Asesores', 'Movil', 'Campañas']
)
segmento = st.sidebar.selectbox("Segmento", ['Particular', 'Universitario', 'VIP', 'Otros'])

# Productos actuales
st.sidebar.subheader("Productos Actuales")
col1, col2 = st.sidebar.columns(2)

with col1:
    cuenta_ahorros = st.checkbox("Cuenta Ahorros")
    cuenta_corriente = st.checkbox("Cuenta Corriente")
    credito_rapido = st.checkbox("Crédito Rápido")
    cuenta_nomina = st.checkbox("Cuenta Nómina")
    cuenta_joven = st.checkbox("Cuenta Joven")
    cuenta_adulto_mayor = st.checkbox("Cuenta Adulto Mayor")
    cuenta_apertura = st.checkbox("Cuenta Apertura")
    cuenta_pagos = st.checkbox("Cuenta Pagos")
    cuenta_debitos = st.checkbox("Cuenta Débitos")

with col2:
    hipotecas = st.checkbox("Hipotecas")
    ahorro_programado = st.checkbox("Ahorro Programado")
    prestamo_libre_inversion = st.checkbox("Préstamo Libre Inversión")
    credito_vivienda = st.checkbox("Crédito Vivienda")

# Botón de predicción
if st.sidebar.button("🔮 Realizar Predicción", type="primary"):

    scaler_mean = [4.94641014e+01, 1.16492858e+02, 1.46335041e+05]
    scaler_scale = [1.3294734e+01, 7.0587843e+01, 2.5989820e+05]

    def manual_scaling(values, mean, scale):
        return [(v - m) / s for v, m, s in zip(values, mean, scale)]

    
    # Mapeos para codificación
    sexo_mapping = {'Masculino': 0, 'Femenino': 1}
    canal_mapping = {
        '004': 0, '007': 1, '013': 2, 'DESCONOCIDO': 3, 'KAA': 4, 'KAB': 5, 'KAC': 6, 'KAD': 7, 'KAE': 8, 'KAF': 9, 'KAG': 10, 'KAH': 11, 'KAI': 12, 'KAJ': 13, 'KAK': 14, 'KAL': 15, 'KAM': 16, 'KAN': 17, 'KAO': 18, 'KAP': 19, 'KAQ': 20, 'KAR': 21, 'KAS': 22, 'Online': 23, 'KAU': 24, 'KAV': 25, 'KAW': 26, 'KAY': 27, 'KAZ': 28, 'KBB': 29, 'KBD': 30, 'KBE': 31, 'KBF': 32, 'KBG': 33, 'KBH': 34, 'KBJ': 35, 'KBL': 36, 'KBM': 37, 'KBO': 38, 'KBP': 39, 'KBQ': 40, 'KBR': 41, 'KBS': 42, 'KBU': 43, 'KBV': 44, 'KBW': 45, 'KBX': 46, 'KBY': 47, 'KBZ': 48, 'KCA': 49, 'KCB': 50, 'KCC': 51, 'KCD': 52, 'KCE': 53, 'KCF': 54, 'KCG': 55, 'KCH': 56, 'KCI': 57, 'KCJ': 58, 'KCK': 59, 'KCL': 60, 'KCM': 61, 'KCN': 62, 'KCO': 63, 'KCP': 64, 'KCQ': 65, 'KCU': 66, 'KCV': 67, 'KDC': 68, 'KDD': 69, 'KDE': 70, 'KDF': 71, 'KDG': 72, 'KDH': 73, 'KDM': 74, 'KDN': 75, 'KDO': 76, 'KDP': 77, 'KDQ': 78, 'KDR': 79, 'KDS': 80, 'KDT': 81, 'KDU': 82, 'KDV': 83, 'KDW': 84, 'KDX': 85, 'KDY': 86, 'KDZ': 87, 'KEA': 88, 'KEB': 89, 'KEC': 90, 'KED': 91, 'KEE': 92, 'KEF': 93, 'KEG': 94, 'KEH': 95, 'KEI': 96, 'KEJ': 97, 'KEK': 98, 'KEL': 99, 'KEM': 100, 'KEN': 101, 'KEO': 102, 'KES': 103, 'KEV': 104, 'KEW': 105, 'KEY': 106, 'KEZ': 107, 'Asesores': 108, 'KFB': 109, 'Oficina': 110, 'KFD': 111, 'KFF': 112, 'KFG': 113, 'KFH': 114, 'KFI': 115, 'KFJ': 116, 'KFK': 117, 'KFL': 118, 'KFM': 119, 'KFN': 120, 'KFP': 121, 'KFR': 122, 'KFS': 123, 'KFT': 124, 'KFU': 125, 'KGV': 126, 'KGW': 127, 'KGX': 128, 'KGY': 129, 'KHA': 130, 'KHC': 131, 'KHD': 132, 'KHE': 133, 'KHF': 134, 'KHK': 135, 'KHL': 136, 'Movil': 137, 'Campañas': 138, 'KHO': 139, 'KHQ': 140, 'RED': 141
    }
    segmento_mapping = {'Particular': 1, 'Universitario': 2, 'VIP': 0, 'Otros': 3}
    
    # Aplicar codificación
    sexo_n = sexo_mapping[sexo]
    canal_entrada_n = canal_mapping[canal_entrada]
    segmento_n = segmento_mapping[segmento]

    age_scaled, antiguedad_scaled, renta_scaled = manual_scaling(
        [age, antiguedad, renta], scaler_mean, scaler_scale
    )
    
    # Crear datos de entrada
    input_data = {
        'age': age_scaled,
        'antiguedad': antiguedad_scaled,
        'renta': renta_scaled,
        'cuenta_ahorros': int(cuenta_ahorros),
        'cuenta_corriente': int(cuenta_corriente),
        'credito_rapido': int(credito_rapido),
        'cuenta_nomina': int(cuenta_nomina),
        'cuenta_joven': int(cuenta_joven),
        'cuenta_adulto_mayor': int(cuenta_adulto_mayor),
        'cuenta_apertura': int(cuenta_apertura),
        'cuenta_pagos': int(cuenta_pagos),
        'cuenta_debitos': int(cuenta_debitos),
        'hipotecas': int(hipotecas),
        'ahorro_programado': int(ahorro_programado),
        'prestamo_libre_inversion': int(prestamo_libre_inversion),
        'credito_vivienda': int(credito_vivienda),
        'sexo_n': sexo_n,
        'canal_entrada_n': canal_entrada_n,
        'segmento_n': segmento_n
    }
    
    # Crear DataFrame
    df = pd.DataFrame([input_data])
    feature_columns = modelo_completo.get('features', list(input_data.keys()))
    df_model = df[feature_columns]
    
    # Realizar predicción
    modelo = modelo_completo['modelo']
    predictions = modelo.predict(df_model)[0]
    probabilities = modelo.predict_proba(df_model)
    
    # Extraer probabilidades
    prob_positive = []
    for i in range(len(predictions)):
        prob_positive.append(probabilities[i][0, 1])
    
    # Mostrar resultados
    st.header("📈 Resultados de la Predicción")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pred_class = "SÍ" if predictions[0] == 1 else "NO"
        st.metric("🏷️ Tarjeta de Crédito", pred_class, f"{prob_positive[0]:.2%}")
    
    with col2:
        pred_class = "SÍ" if predictions[1] == 1 else "NO"
        st.metric("💳 E-Cuenta", pred_class, f"{prob_positive[1]:.2%}")
    
    with col3:
        pred_class = "SÍ" if predictions[2] == 1 else "NO"
        st.metric("💰 Depósito Largo Plazo", pred_class, f"{prob_positive[2]:.2%}")
    
    # Recomendaciones
    st.header("💡 Recomendaciones")
    targets = ['Tarjeta Crédito', 'E-Cuenta', 'Depósito Largo Plazo']
    
    if sum(predictions) == 0:
        st.info("🔍 Cliente con baja probabilidad de contratación.")
    elif sum(predictions) == 1:
        producto = targets[predictions.tolist().index(1)]
        st.success(f"🎯 Recomienda: **{producto}**")
    else:
        productos = [targets[i] for i, p in enumerate(predictions) if p == 1]
        st.success(f"🎯 Recomienda: **{', '.join(productos)}**")
