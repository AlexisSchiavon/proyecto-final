import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-positive {
        color: #28a745;
        font-weight: bold;
    }
    .prediction-negative {
        color: #dc3545;
        font-weight: bold;
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

# Cargar modelo
with open('modelo_xgb_optimizado_sin_sobreajuste.pkl', 'rb') as f:
    modelo_completo = pickle.load(f)

# Información del modelo
st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.markdown("**Información del Modelo:**")
st.markdown(f"- **Algoritmo:** XGBoost Optimizado")
st.markdown(f"- **Productos a predecir:** Tarjeta de Crédito, E-Cuenta, Depósito a Largo Plazo")
st.markdown(f"- **Características:** {modelo_completo['datos_info']['n_features']} variables")
st.markdown(f"- **Score F1:** {modelo_completo['mejor_score']:.4f}")
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
canal_entrada = st.sidebar.selectbox("Canal de Entrada", ['Oficina', 'Online', 'Teléfono', 'Móvil', 'Cajero'])
segmento = st.sidebar.selectbox("Segmento", ['Particular', 'Universitario', 'VIP'])

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
    
    # Mapeos para codificación
    sexo_mapping = {'Masculino': 1, 'Femenino': 0}
    canal_mapping = {'Oficina': 0, 'Online': 1, 'Teléfono': 2, 'Móvil': 3, 'Cajero': 4}
    segmento_mapping = {'Particular': 0, 'Universitario': 1, 'VIP': 2}
    
    # Aplicar codificación
    sexo_n = sexo_mapping[sexo]
    canal_entrada_n = canal_mapping[canal_entrada]
    segmento_n = segmento_mapping[segmento]
    
    # Crear diccionario con datos de entrada
    input_data = {
        'age': age,
        'antiguedad': antiguedad,
        'renta': renta,
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
    
    # Ordenar columnas según el modelo
    feature_columns = modelo_completo['features']
    df_model = df[feature_columns]
    
    # Realizar predicción
    modelo = modelo_completo['modelo']
    predictions = modelo.predict(df_model)[0]
    probabilities = modelo.predict_proba(df_model)
    
    # Extraer probabilidades de la clase positiva
    prob_positive = []
    for i in range(len(predictions)):
        prob_positive.append(probabilities[i][0][1])
    
    # Mostrar resultados
    st.header("📈 Resultados de la Predicción")
    
    # Métricas principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pred_class = "SÍ" if predictions[0] == 1 else "NO"
        st.metric(
            label="🏷️ Tarjeta de Crédito",
            value=pred_class,
            delta=f"{prob_positive[0]:.2%} probabilidad"
        )
    
    with col2:
        pred_class = "SÍ" if predictions[1] == 1 else "NO"
        st.metric(
            label="💳 E-Cuenta",
            value=pred_class,
            delta=f"{prob_positive[1]:.2%} probabilidad"
        )
    
    with col3:
        pred_class = "SÍ" if predictions[2] == 1 else "NO"
        st.metric(
            label="💰 Depósito Largo Plazo",
            value=pred_class,
            delta=f"{prob_positive[2]:.2%} probabilidad"
        )
    
    # Visualizaciones
    targets = ['Tarjeta Crédito', 'E-Cuenta', 'Depósito Largo Plazo']
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
    
    # Gráfico de barras para predicciones
    fig_pred = go.Figure(data=[
        go.Bar(
            x=targets,
            y=predictions,
            marker_color=colors,
            text=[f'{"SÍ" if pred == 1 else "NO"}' for pred in predictions],
            textposition='auto',
        )
    ])
    
    fig_pred.update_layout(
        title="Predicciones de Productos Bancarios",
        xaxis_title="Productos",
        yaxis_title="Predicción (0=No, 1=Sí)",
        yaxis=dict(tickvals=[0, 1], ticktext=['NO', 'SÍ']),
        height=400
    )
    
    # Gráfico de probabilidades
    fig_prob = go.Figure()
    
    for i, (target, prob, color) in enumerate(zip(targets, prob_positive, colors)):
        fig_prob.add_trace(go.Bar(
            x=[target],
            y=[prob],
            name=target,
            marker_color=color,
            text=[f'{prob:.2%}'],
            textposition='auto',
        ))
    
    fig_prob.update_layout(
        title="Probabilidades de Contratación",
        xaxis_title="Productos",
        yaxis_title="Probabilidad",
        yaxis=dict(tickformat='.0%'),
        height=400,
        showlegend=False
    )
    
    # Mostrar gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig_pred, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig_prob, use_container_width=True)
    
    # Recomendaciones
    st.header("💡 Recomendaciones")
    
    if sum(predictions) == 0:
        st.info("🔍 Este cliente tiene baja probabilidad de contratar productos adicionales. Considera ofertas especiales o incentivos.")
    elif sum(predictions) == 1:
        producto_recomendado = targets[predictions.tolist().index(1)]
        st.success(f"🎯 Se recomienda enfocar la oferta en: **{producto_recomendado}**")
    else:
        productos_recomendados = [targets[i] for i, pred in enumerate(predictions) if pred == 1]
        st.success(f"🎯 Se recomienda ofrecer múltiples productos: **{', '.join(productos_recomendados)}**")
    
    # Mostrar datos de entrada
    with st.expander("📋 Ver datos de entrada"):
        st.write("**Datos originales:**")
        st.write(f"- Sexo: {sexo}")
        st.write(f"- Edad: {age}")
        st.write(f"- Antigüedad: {antiguedad} meses")
        st.write(f"- Renta: ${renta:,}")
        st.write(f"- Canal: {canal_entrada}")
        st.write(f"- Segmento: {segmento}")
        st.write("**Datos codificados:**")
        st.json(input_data)