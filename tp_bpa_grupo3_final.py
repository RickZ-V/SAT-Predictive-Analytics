import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Librerías exactas de su nuevo Setup Global
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Configuración estética de la app
st.set_page_config(page_title="SAT Lima - Predictive Analytics v2", layout="wide", page_icon="📊")

# BARRA LATERAL (Sidebar)
st.sidebar.title("Configuración General")
st.sidebar.markdown("**Curso:** Business Predictive Analytics — 1ASI0709")
st.sidebar.markdown("**Grupo:** Grupo 3 (Actualizado)")
st.sidebar.markdown("""
**Integrantes:**
* Santiago Rojas
* Abdiel Beraun
* Victor Temoche
* Orlando Mostacero
* Manuel Chávez
""")

st.title("🎯 Modelado Predictivo de Recaudación por Papeletas — SAT Lima")
st.caption("Trabajo Final BPA — Versión Mejorada con PowerTransformer & ML Pipeline")
st.markdown("---")

tab_data, tab_train, tab_predict = st.tabs([
    "📂 1. Calidad & EDA", 
    "🏗️ 2. Pipeline & Modelado Optimizado", 
    "🔮 3. Módulo de Predicción Final"
])

# Carga de datos con caché estricta para liberar RAM
@st.cache_data
def cargar_y_procesar_datos():
    df = pd.read_csv('Multas_Pagadas.csv')
    
    # Capítulo 2.2 — Limpieza de datos constants
    if 'Año pago' in df.columns:
        df.drop(columns=['Año pago'], inplace=True)
        
    # Eliminación de espacios extras en strings
    cat_cols = ['Código falta', 'Descripción falta', 'Tipo formato', 'Nivel de gravedad']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.replace(r' +', ' ', regex=True)
            
    # Feature Engineering del Grupo 3
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
    st.error("⚠️ No se pudo cargar el dataset local. Verifica que el archivo CSV mantenga su nombre original.")
    st.stop()

# =========================================================================
# PESTAÑA 1: CALIDAD & EDA
# =========================================================================
with tab_data:
    st.header("Análisis Exploratorio de Datos (EDA)")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Registros Históricos", f"{len(df_clean):,}")
    m2.metric("Recaudación Total Q1", f"S/ {df_clean['Monto total pagado'].sum():,.2f}")
    m3.metric("Rezago de Cartera Estructural", f"{df_clean['Es rezago'].mean():.1%}")
    m4.metric("Aplicación de Pronto Pago", f"{df_clean['Tiene descuento'].mean():.1%}")
    
    st.markdown("### Muestra de Datos Procesados")
    st.dataframe(df_clean.head(10), use_container_width=True)
    
  st.markdown("### Visualizaciones Clave del Negocio")
    col_graphs1, col_graphs2 = st.columns(2)
    
    with col_graphs1:
        st.subheader("Volumen de Papeletas por Gravedad (Jerarquía L < M < G)")
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        sns.countplot(data=df_clean, x='Nivel de gravedad', order=['L', 'M', 'G'], palette='viridis', ax=ax1)
        ax1.set_ylabel('Cantidad de multas')
        st.pyplot(fig1)
        plt.close(fig1)
        
    with col_graphs2:
        st.subheader("Análisis de Dispersión: Monto Emitido vs Total Pagado")
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        sns.scatterplot(data=df_clean, x='Monto emitido', y='Monto total pagado', alpha=0.4, color='steelblue', ax=ax2)
      
        ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'S/ {x:,.0f}'))
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'S/ {x:,.0f}'))

        plt.setp(ax2.get_xticklabels(), rotation=15, ha="right")
        
        st.pyplot(fig2)
        plt.close(fig2)

# =========================================================================
# PESTAÑA 2: PIPELINE & MODELADO OPTIMIZADO (NUEVA LÓGICA)
# =========================================================================
with tab_train:
    st.header("Entrenamiento del Pipeline de Producción")
    st.write("Ajusta la partición de datos. El pipeline aplicará de forma secuencial las imputaciones, la reducción de sesgo mediante Yeo-Johnson y el escalado robusto.")
    
    test_percentage = st.slider("Tamaño del Set de Validación (Test)", 0.10, 0.40, 0.20, step=0.05)
    
    if st.button("🚀 Inicializar y Entrenar Pipeline v2"):
        with st.spinner("Ejecutando transformaciones estadísticas complejas..."):
            
            # Definición exacta de features de su nuevo entregable
            cols_num = ['Monto emitido', 'Años de rezago', 'Descuento', 'Reincidencia pagada', 'Cant. multas con pagos']
            cols_nom = ['Tipo formato', 'Mes pago nombre']
            cols_ord = ['Nivel de gravedad']
            TARGET = 'Monto total pagado'
            
            X = df_clean[cols_num + cols_nom + cols_ord].copy()
            y = df_clean[TARGET].copy()
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage, random_state=42)
            
            # NUEVO CAMBIO TÉCNICO: Integración del PowerTransformer (Yeo-Johnson) para eliminar sesgo
            pipe_num = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('yeo_johnson', PowerTransformer(method='yeo-johnson', standardize=False)),
                ('scaler', RobustScaler())
            ])
            
            pipe_nom = Pipeline(steps=[
                ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
            ])
            
            pipe_ord = Pipeline(steps=[
                ('ordinal', OrdinalEncoder(categories=[['L', 'M', 'G']], handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            
            preprocesador = ColumnTransformer(transformers=[
                ('num', pipe_num, cols_num), 
                ('nom', pipe_nom, cols_nom), 
                ('ord', pipe_ord, cols_ord)
            ], remainder='drop')
            
            # Hiperparámetros refinados basados en su sintonización con GridSearchCV
            modelo_final_opt = Pipeline([
                ('prep', preprocesador),
                ('model', RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=42))
            ])
            
            modelo_final_opt.fit(X_train, y_train)
            
            # Guardado en memoria volátil de sesión para protección de RAM
            st.session_state['modelo_v2'] = modelo_final_opt
            
            y_pred = modelo_final_opt.predict(X_test)
            st.success("¡Pipeline de Machine Learning v2 entrenado con éxito!")
            
            # Despliegue de métricas de rendimiento reales
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R² Score (Explicabilidad)", f"{r2_score(y_test, y_pred):.4f}")
            c2.metric("MAE (Error Medio Absoluto)", f"S/ {mean_absolute_error(y_test, y_pred):,.2f}")
            c3.metric("RMSE (Error Cuadrático)", f"S/ {mean_squared_error(y_test, y_pred)**0.5:,.2f}")
            c4.metric("MAPE (Porcentaje de Error)", f"{mean_absolute_percentage_error(y_test, y_pred)*100:.2f}%")
            
            # Diagnóstico Visual
            st.markdown("### Evaluación Diagnóstica del Modelo")
            fig_diag, ax_diag = plt.subplots(1, 2, figsize=(12, 4))
            
            ax_diag[0].scatter(y_test, y_pred, alpha=0.4, color='steelblue')
            lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            ax_diag[0].plot(lim, lim, 'r--', lw=1.5, label='Línea de Referencia Perfecta')
            ax_diag[0].set_title('Monto Real vs. Monto Predicho')
            ax_diag[0].set_xlabel('Real (S/)')
            ax_diag[0].set_ylabel('Predicho (S/)')
            ax_diag[0].legend()
            
            residuos = y_test - y_pred
            ax_diag[1].hist(residuos, bins=30, color='teal', edgecolor='white')
            ax_diag[1].axvline(0, color='red', ls='--')
            ax_diag[1].set_title('Distribución Esférica de Residuos')
            ax_diag[1].set_xlabel('Residuo (Soles)')
            
            st.pyplot(fig_diag)
            plt.close(fig_diag)

# =========================================================================
# PESTAÑA 3: MÓDULO DE PREDICCIÓN FINAL
# =========================================================================
with tab_predict:
    st.header("Simulador de Recaudación en Tiempo Real (Modelo v2)")
    
    if 'modelo_v2' in st.session_state:
        st.success("✅ Pipeline predictivo v2 con transformación de potencia activo.")
        
        st.markdown("### Parámetros de Consulta:")
        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            monto_emitido = st.number_input("Monto Emitido Original (S/)", min_value=0.0, value=450.0, step=50.0)
            descuento = st.number_input("Monto de Descuento (S/)", min_value=0.0, value=0.0, step=10.0)
        with col_in2:
            anios_rezago = st.slider("Años de Rezago de la Papeleta", 0, 15, 2)
            reincidencia = st.number_input("Cargo Adicional por Reincidencia (S/)", min_value=0.0, value=0.0, step=10.0)
        with col_in3:
            cant_multas = st.number_input("Cantidad de multas asociadas", min_value=1, value=1, step=1)
            nivel_gravedad = st.selectbox("Nivel de Gravedad", options=['L', 'M', 'G'])
            tipo_formato = st.selectbox("Tipo de Formato Registrado", options=['Papeleta de Tránsito', 'Papeleta Electrónica', 'Video Papeleta', 'Papeleta Cámara'])
            mes_pago_nombre = st.selectbox("Mes de Cobro Proyectado", options=['Enero', 'Febrero', 'Marzo'])
            
        if st.button("🔮 Ejecutar Inferencia Predictiva"):
            input_data = pd.DataFrame([{
                'Monto emitido': monto_emitido, 'Años de rezago': anios_rezago, 'Descuento': descuento,
                'Reincidencia pagada': reincidencia, 'Cant. multas con pagos': cant_multas,
                'Tipo formato': tipo_formato, 'Mes pago nombre': mes_pago_nombre, 'Nivel de gravedad': nivel_gravedad
            }])
            
            # El pipeline aplica automáticamente la transformación Yeo-Johnson al input antes de predecir
            prediccion_final = st.session_state['modelo_v2'].predict(input_data)[0]
            st.markdown("---")
            st.metric(label="Monto Final Estimado que Recaudará el SAT (S/)", value=f"S/ {prediccion_final:,.2f}")
    else:
        st.warning("⚠️ El nuevo pipeline predictivo no ha sido entrenado. Ve a la **Pestaña 2** e inicializa el entrenamiento para activar este módulo.")
