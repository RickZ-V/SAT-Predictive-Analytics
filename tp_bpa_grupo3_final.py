import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Componentes de Sklearn importados desde tu script definitivo
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Configuración de página
st.set_page_config(page_title="SAT Lima - Predictive Analytics v4", layout="wide", page_icon="📊")

# BARRA LATERAL (Sidebar)
st.sidebar.title("Configuración General")
st.sidebar.markdown("**Curso:** Business Predictive Analytics")
st.sidebar.markdown("**Grupo:** Grupo 3")
st.sidebar.markdown("""
**Integrantes:**
* Santiago Rojas
* Abdiel Beraun
* Victor Temoche
* Orlando Mostacero
* Manuel Chávez
""")

st.title("🎯 Modelado Predictivo de Recaudación por Papeletas — SAT Lima")
st.caption("Trabajo Final BPA — Versión Sintonizada y Mitigada contra Data Leakage (GridSearchCV)")
st.markdown("---")

tab_data, tab_train, tab_predict = st.tabs([
    "📂 1. Calidad & EDA", 
    "🔬 2. Benchmarking & Modelado Sintonizado", 
    "🔮 3. Módulo de Predicción Final"
])

# Carga y limpieza de datos estructurada
@st.cache_data
def cargar_y_procesar_datos():
    df = pd.read_csv('Multas_Pagadas.csv')
    
    if 'Año pago' in df.columns:
        df.drop(columns=['Año pago'], inplace=True)
        
    cat_cols = ['Código falta', 'Descripción falta', 'Tipo formato', 'Nivel de gravedad']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.replace(r' +', ' ', regex=True)
            
    # Ingeniería de variables del negocio
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
    st.error("⚠️ No se pudo cargar el dataset local. Verifica que el archivo CSV mantenga su nombre original en el repositorio.")
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
        sns.scatterplot(data=df_clean, x='Monto emitido', y='Monto total pagado', alpha=0.4, color='steelblue', ax=fig2.gca())
        
        ax2 = fig2.gca()
        ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'S/ {x:,.0f}'))
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'S/ {x:,.0f}'))
        plt.setp(ax2.get_xticklabels(), rotation=15, ha="right")
        
        st.pyplot(fig2)
        plt.close(fig2)

# =========================================================================
# PESTAÑA 2: PIPELINE & MODELADO SINTONIZADO (SIN LEAKAGE)
# =========================================================================
with tab_train:
    st.header("🔬 Modelización y Sintonización Fina de Algoritmos")
    st.write("Al ejecutar este bloque, el sistema aislará las muestras para evitar la fuga de datos, procesará el preprocesamiento por columnas y entrenará el Random Forest optimizado mediante la grilla de GridSearchCV.")
    
    test_percentage = st.slider("Tamaño del Set de Validación (Test Split)", 0.10, 0.40, 0.20, step=0.05)
    
    if st.button("🚀 Inicializar Pipeline v4 (Sin Leakage)"):
        with st.spinner("Entrenando modelos y calculando métricas honestas..."):
            
            # Definición de columnas basada en tu script de Colab
            cols_num = ['Monto emitido', 'Años de rezago', 'Descuento', 'Reincidencia pagada', 'Cant. multas con pagos']
            cols_nom = ['Tipo formato', 'Mes pago nombre']
            cols_ord = ['Nivel de gravedad']
            TARGET = 'Monto total pagado'
            
            X_B = df_clean[cols_num + cols_nom + cols_ord].copy()
            y_B = df_clean[TARGET].copy()
            
            # PASO ESENCIAL: Split inmediato para erradicar el Data Leakage
            X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_B, y_B, test_size=test_percentage, random_state=42)
            
            # Arquitectura del Preprocesador aislado
            pipe_num_B = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('yeo_johnson', PowerTransformer(method='yeo-johnson', standardize=False)),
                ('scaler', RobustScaler())
            ])
            pipe_nom_B = Pipeline([('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))])
            pipe_ord_B = Pipeline([('ordinal', OrdinalEncoder(categories=[['L', 'M', 'G']], handle_unknown='use_encoded_value', unknown_value=-1))])
            
            preprocesador_B = ColumnTransformer(transformers=[
                ('num', pipe_num_B, cols_num), ('nom', pipe_nom_B, cols_nom), ('ord', pipe_ord_B, cols_ord)
            ], remainder='drop')
            
            # 1. Benchmarking de Algoritmos Base en Cross-Validation
            modelos_dict = {
                'Regresión Lineal': LinearRegression(),
                'Ridge': Ridge(alpha=1.0),
                'Lasso': Lasso(alpha=1.0, max_iter=5000),
                'Árbol Decisión': DecisionTreeRegressor(max_depth=6, random_state=42),
                'Random Forest (Base)': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
            }
            
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_resultados = []
            
            for nombre, modelo in modelos_dict.items():
                pipe_eval = Pipeline([('prep', preprocesador_B), ('model', modelo)])
                r2_scores = cross_val_score(pipe_eval, X_train_B, y_train_B, cv=kf, scoring='r2')
                rmse_scores = np.sqrt(-cross_val_score(pipe_eval, X_train_B, y_train_B, cv=kf, scoring='neg_mean_squared_error'))
                cv_resultados.append({
                    'Modelo': nombre,
                    'R² Medio (CV)': r2_scores.mean(),
                    'RMSE Medio': rmse_scores.mean()
                })
                
            cv_df = pd.DataFrame(cv_resultados).sort_values('R² Medio (CV)', ascending=False)
            
            st.subheader("📊 Comparativa de Algoritmos Base (Validación Cruzada)")
            st.dataframe(cv_df.style.format({'R² Medio (CV)': '{:.4f}', 'RMSE Medio': 'S/ {:,.2f}'}), use_container_width=True)
            
            # 2. Carga del Modelo Ganador Optimizado por tu GridSearchCV (108 candidatos, 540 fits)
            st.markdown("---")
            st.subheader("🎯 Rendimiento Definitivo del Modelo Seleccionado (Random Forest Optimizado)")
            st.caption("Hiperparámetros óptimos aplicados: max_depth=12, n_estimators=300, min_samples_split=2, min_samples_leaf=1")
            
            mejor_modelo_rf = Pipeline([
                ('prep', preprocesador_B),
                ('model', RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=-1))
            ])
            
            # El ajuste se hace estrictamente con la data de entrenamiento aislada
            mejor_modelo_rf.fit(X_train_B, y_train_B)
            y_pred_B = mejor_modelo_rf.predict(X_test_B)
            
            # Despliegue de métricas exactas de tu consola de Colab
            c1, c2, c3 = st.columns(3)
            c1.metric("R² Final (Test Set)", f"{r2_score(y_test_B, y_pred_B):.4f}")
            c2.metric("MAE (Soles)", f"S/ {mean_absolute_error(y_test_B, y_pred_B):,.2f}")
            c3.metric("RMSE (Soles)", f"S/ {np.sqrt(mean_squared_error(y_test_B, y_pred_B)):,.2f}")
            
            # 3. Gráficos de diagnóstico trilineal solicitados
            st.markdown("### Diagnóstico Visual de Residuos")
            residuos_B = y_test_B.values - y_pred_B
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            
            # G1: Real vs Predicho
            axes[0].scatter(y_test_B, y_pred_B, alpha=0.4, s=15, color='steelblue')
            lim = [min(y_test_B.min(), y_pred_B.min()), max(y_test_B.max(), y_pred_B.max())]
            axes[0].plot(lim, lim, 'r--', lw=1.5, label='Predicción perfecta')
            axes[0].set_title('Real vs Predicho (Sin Leakage)')
            axes[0].set_xlabel('Monto real (S/)')
            axes[0].set_ylabel('Monto predicho (S/)')
            axes[0].legend()
            
            # G2: Residuos vs Predicho
            axes[1].scatter(y_pred_B, residuos_B, alpha=0.4, s=15, color='steelblue')
            axes[1].axhline(0, color='red', linestyle='--', lw=1.5)
            axes[1].set_title('Residuos vs Predicho')
            axes[1].set_xlabel('Monto predicho (S/)')
            axes[1].set_ylabel('Residuo')
            
            # G3: Distribución de residuos
            axes[2].hist(residuos_B, bins=40, color='steelblue', edgecolor='white')
            axes[2].axvline(0, color='red', linestyle='--', lw=1.5)
            axes[2].set_title('Distribución de Residuos')
            axes[2].set_xlabel('Residuo (S/)')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Guardar el modelo en sesión entrenado con el 100% de la data para inferencias del simulador
            modelo_produccion = Pipeline([
                ('prep', preprocesador_B),
                ('model', RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=-1))
            ])
            modelo_produccion.fit(X_B, y_B)
            st.session_state['modelo_v4_def'] = modelo_produccion
            st.success("🚀 Modelo final reentrenado con el 100% de los registros disponibles y guardado para inferencia.")

# =========================================================================
# PESTAÑA 3: MÓDULO DE PREDICCIÓN FINAL
# =========================================================================
with tab_predict:
    st.header("Simulador de Recaudación en Tiempo Real (Modelo Sintonizado)")
    
    if 'modelo_v4_def' in st.session_state:
        st.success("✅ Motor predictivo optimizado (Random Forest, n_estimators=300) cargado con éxito.")
        
        st.markdown("### Parámetros de Inferencia de la Papeleta:")
        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            monto_emitido = st.number_input("Monto Emitido Original (S/)", min_value=0.0, value=450.0, step=50.0)
            descuento = st.number_input("Monto de Descuento (S/)", min_value=0.0, value=0.0, step=10.0)
        with col_in2:
            anios_rezago = st.slider("Años de Rezago de la Papeleta", 0, 15, 2)
            reincidencia = st.number_input("Cargo por Reincidencia (S/)", min_value=0.0, value=0.0, step=10.0)
        with col_in3:
            cant_multas = st.number_input("Cantidad de multas agrupadas", min_value=1, value=1, step=1)
            nivel_gravedad = st.selectbox("Nivel de Gravedad", options=['L', 'M', 'G'])
            tipo_formato = st.selectbox("Tipo de Formato Registrado", options=['Papeleta de Tránsito', 'Papeleta Electrónica', 'Video Papeleta', 'Papeleta Cámara'])
            mes_pago_nombre = st.selectbox("Mes de Pago Estimado", options=['Enero', 'Febrero', 'Marzo'])
            
        if st.button("🔮 Calcular Estimación de Recaudación"):
            input_data = pd.DataFrame([{
                'Monto emitido': monto_emitido, 'Años de rezago': anios_rezago, 'Descuento': descuento,
                'Reincidencia pagada': reincidencia, 'Cant. multas con pagos': cant_multas,
                'Tipo formato': tipo_formato, 'Mes pago nombre': mes_pago_nombre, 'Nivel de gravedad': nivel_gravedad
            }])
            
            prediccion_final = st.session_state['modelo_v4_def'].predict(input_data)[0]
            st.markdown("---")
            st.metric(label="Monto Final Estimado que Recaudará el SAT", value=f"S/ {prediccion_final:,.2f}")
    else:
        st.warning("⚠️ Primero debes inicializar el entrenamiento en la **Pestaña 2** para calibrar el Pipeline sin leakage y activar este simulador.")
