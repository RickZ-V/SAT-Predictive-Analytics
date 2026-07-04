import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Configuración de página optimizada
st.set_page_config(page_title="SAT Lima - Predictive Analytics", layout="wide", page_icon="📊")

# BARRA LATERAL (Sidebar)
st.sidebar.title("Configuración General")
st.sidebar.markdown("**Curso:** Business Predictive Analytics — 1ASI0709")
st.sidebar.markdown("**Grupo:** Grupo 3")
st.sidebar.markdown("""
**Integrantes:**
* Santiago Rojas
* Abdiel Beraun
* Victor Temoche
* Orlando Mostacero
* Manuel Chávez
""")

st.title("🎯 Modelado Predictivo de Recaudación — SAT Lima")
st.caption("Trabajo Final BPA — Primer Trimestre 2026")
st.markdown("---")

tab_data, tab_train, tab_predict = st.tabs([
    "📂 1. Calidad & EDA", 
    "🏗️ 2. Pipeline & Modelado Base", 
    "🔮 3. Módulo de Predicción Final"
])

# Carga de datos con caché estricta para liberar RAM
@st.cache_data
def cargar_y_procesar_datos():
    df = pd.read_csv('Multas_Pagadas.csv')
    
    if 'Año pago' in df.columns:
        df.drop(columns=['Año pago'], inplace=True)
        
    cat_cols = ['Código falta', 'Descripción falta', 'Tipo formato', 'Nivel de gravedad']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.replace(r' +', ' ', regex=True)
            
    df['Años de rezago'] = 2026 - df['Año infracción']
    df['Es rezago'] = df['Años de rezago'] > 0
    df['Tiene descuento'] = df['Descuento'] > 0
    df['Porcentaje descuento'] = (df['Descuento'] / df['Monto emitido'] * 100).round(2)
    
    meses = {1: 'Enero', 2: 'Febrero', 3: 'Marzo'}
    df['Mes pago nombre'] = df['Mes pago'].map(meses)
    return df

try:
    df_clean = cargar_y_procesar_datos()
except Exception as e:
    st.error("⚠️ No se pudo cargar el dataset.")
    st.stop()

# =========================================================================
# PESTAÑA 1: CALIDAD & EDA
# =========================================================================
with tab_data:
    st.header("Análisis Exploratorio de Datos (EDA)")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Registros Totales", f"{len(df_clean):,}")
    m2.metric("Recaudación Total Q1", f"S/ {df_clean['Monto total pagado'].sum():,.2f}")
    m3.metric("Rezago Estructural", f"{df_clean['Es rezago'].mean():.1%}")
    m4.metric("Papeletas con Descuento", f"{df_clean['Tiene descuento'].mean():.1%}")
    
    st.markdown("### Vista Previa del Dataset Tras Limpieza")
    st.dataframe(df_clean.head(10), use_container_width=True)
    
    st.markdown("### Visualizaciones Clave del Negocio")
    col_graphs1, col_graphs2 = st.columns(2)
    
    with col_graphs1:
        st.subheader("Frecuencia por Nivel de Gravedad (L < M < G)")
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        sns.countplot(data=df_clean, x='Nivel de gravedad', order=['L', 'M', 'G'], palette='viridis', ax=ax1)
        st.pyplot(fig1)
        plt.close(fig1) # Cierra la figura para liberar memoria inmediatamente
        
    with col_graphs2:
        st.subheader("Relación: Monto Emitido vs Monto Total Pagado")
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        sns.scatterplot(data=df_clean, x='Monto emitido', y='Monto total pagado', alpha=0.4, color='steelblue', ax=ax2)
        ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        st.pyplot(fig2)
        plt.close(fig2)

# =========================================================================
# PESTAÑA 2: PIPELINE & MODELADO BASE
# =========================================================================
with tab_train:
    st.header("Construcción del Pipeline y Entrenamiento")
    st.write("Configura el split y valida las métricas del Random Forest.")
    
    test_percentage = st.slider("Porcentaje para el Test Set", 0.10, 0.40, 0.20, step=0.05)
    
    if st.button("🚀 Ejecutar Pipeline de Machine Learning"):
        with st.spinner("Entrenando de forma optimizada..."):
            
            cols_num = ['Monto emitido', 'Años de rezago', 'Descuento', 'Reincidencia pagada', 'Cant. multas con pagos']
            cols_nom = ['Tipo formato', 'Mes pago nombre']
            cols_ord = ['Nivel de gravedad']
            
            X = df_clean[cols_num + cols_nom + cols_ord].copy()
            y = df_clean['Monto total pagado'].copy()
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage, random_state=42)
            
            pipe_num = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())])
            pipe_nom = Pipeline([('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))])
            pipe_ord = Pipeline([('ordinal', OrdinalEncoder(categories=[['L', 'M', 'G']], handle_unknown='use_encoded_value', unknown_value=-1))])
            
            preprocesador = ColumnTransformer(transformers=[
                ('num', pipe_num, cols_num), ('nom', pipe_nom, cols_nom), ('ord', pipe_ord, cols_ord)
            ], remainder='drop')
            
            # OPTIMIZACIÓN DE MEMORIA: Reducimos n_estimators a 50 y quitamos n_jobs=-1 para que no consuma RAM en exceso
            modelo_ligero = Pipeline([
                ('prep', preprocesador),
                ('model', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42))
            ])
            
            modelo_ligero.fit(X_train, y_train)
            
            # Guardar el binario en el estado de la sesión de Streamlit en lugar de un archivo pkl pesado
            st.session_state['modelo_entrenado'] = modelo_ligero
            
            y_pred = modelo_ligero.predict(X_test)
            st.success("¡Pipeline entrenado con éxito!")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
            c2.metric("MAE (Soles)", f"S/ {mean_absolute_error(y_test, y_pred):,.2f}")
            c3.metric("RMSE (Soles)", f"S/ {mean_squared_error(y_test, y_pred)**0.5:,.2f}")
            c4.metric("MAPE", f"{mean_absolute_percentage_error(y_test, y_pred)*100:.2f}%")

# =========================================================================
# PESTAÑA 3: MÓDULO DE PREDICCIÓN FINAL
# =========================================================================
with tab_predict:
    st.header("Simulador de Recaudación del SAT en Tiempo Real")
    
    if 'modelo_entrenado' in st.session_state:
        st.success("✅ Modelo cargado en memoria interna y listo para simular.")
        
        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            monto_emitido = st.number_input("Monto Emitido Original (S/)", min_value=0.0, value=450.0, step=50.0)
            descuento = st.number_input("Descuento Aplicado (S/)", min_value=0.0, value=0.0, step=10.0)
        with col_in2:
            anios_rezago = st.slider("Años de Rezago (Antigüedad)", 0, 15, 1)
            reincidencia = st.number_input("Cargo por Reincidencia (S/)", min_value=0.0, value=0.0, step=10.0)
        with col_in3:
            cant_multas = st.number_input("Cant. multas con pagos", min_value=1, value=1, step=1)
            nivel_gravedad = st.selectbox("Nivel de Gravedad", options=['L', 'M', 'G'])
            tipo_formato = st.selectbox("Tipo de Formato", options=['Infraccion', 'Peatonal', 'Conductor'])
            mes_pago_nombre = st.selectbox("Mes de Pago Estimado", options=['Enero', 'Febrero', 'Marzo'])
            
        if st.button("🔮 Calcular Estimación de Pago"):
            input_data = pd.DataFrame([{
                'Monto emitido': monto_emitido, 'Años de rezago': anios_rezago, 'Descuento': descuento,
                'Reincidencia pagada': reincidencia, 'Cant. multas con pagos': cant_multas,
                'Tipo formato': tipo_formato, 'Mes pago nombre': mes_pago_nombre, 'Nivel de gravedad': nivel_gravedad
            }])
            
            prediccion_final = st.session_state['modelo_entrenado'].predict(input_data)[0]
            st.markdown("---")
            st.metric(label="Monto Final Estimado a Recaudar por el SAT", value=f"S/ {prediccion_final:,.2f}")
    else:
        st.warning("⚠️ El modelo no está inicializado. Por favor, ve a la **Pestaña 2** y haz clic en 'Ejecutar Pipeline de Machine Learning' para activarlo.")
