# =======================================================
# App Streamlit Profesional: Forecasting de Demanda Semanal por SKU en Ferreterías
# Usa NeuralForecast (NHITS) para predicciones multi-SKU con exógenas/futuras
# Dashboard interactivo completo con KPIs, filtros, gráficos elegantes
# Incluye: Dashboard Principal, EDA Interactivo, Predicciones Dinámicas,
# Análisis de Sensibilidad, Gestión de Inventario, Cadenas de Markov, Monte Carlo,
# Análisis de la Cesta del Mercado (Market Basket Analysis)
# Ejecuta con: streamlit run app_demanda_ferreteria_bi.py
# =======================================================
# Requisitos: pip install streamlit pandas numpy matplotlib seaborn plotly neuralforecast networkx mlxtend

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import GMM, MQLoss, DistributionLoss
from neuralforecast.losses.pytorch import MAE
from utilsforecast.feature_engineering import future_exog_to_historic
import warnings
warnings.filterwarnings('ignore')

# Configuración página (estilo BI profesional)
st.set_page_config(page_title="Dashboard Ferreterías - Naren Castellón", layout="wide")
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stMetric {background-color: #003300; padding: 10px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
    h1 {color: #1e3a8a; text-align: center;}
    .sidebar .sidebar-content {background-color: #e0e7ff;}
</style>
""", unsafe_allow_html=True)


st.title("🔧 Forecasting de Demanda Semanal por SKU en Ferreterías")
st.markdown("**Redes Neuronales + Análisis Avanzado de Supply Chain** | Creado por Naren Castellón (@NarenCastellon)")

# Imagen header
st.sidebar.image("https://www.tunicaragua.com/images/stories/virtuemart/category/Banner-Categoría7.jpg", 
         caption="Pronóstico inteligente de demanda y optimización de inventario en ferreterías", )

# =======================================================
# Carga de datos y modelo base (cacheado)
# =======================================================
@st.cache_resource
def load_data_and_model():
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', '2025-12-31', freq='W-MON')
    skus = [
        'Martillo 16oz', 'Destornillador Phillips', 'Taladro Inalámbrico', 'Sierra Circular', 'Pintura Blanca 1Gal',
        'Cemento 50kg', 'Tornillos 2"', 'Clavos 3"', 'Cinta Métrica 5m', 'Nivel 1m',
        'Llave Ajustable', 'Brochas 2"', 'Lija Grano 120', 'Pegamento PVC', 'Tubos PVC 1/2"'
    ]

    index = pd.MultiIndex.from_product([skus, dates], names=['sku_id', 'ds'])
    df = pd.DataFrame(index=index).reset_index()

    df['price_unit'] = np.random.uniform(5.0, 150.0, len(df)).round(2)
    df['promotion_active'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
    df['season_factor'] = np.sin(2 * np.pi * df['ds'].dt.isocalendar().week / 52) + 1.5
    df['temperature_avg'] = np.random.normal(25, 8, len(df)).clip(10, 40)
    df['rain_days'] = np.random.poisson(3, len(df)).clip(0, 7)
    df['events_nearby'] = np.random.choice([0, 1, 2], len(df), p=[0.7, 0.2, 0.1])
    df['marketing_spend'] = np.random.normal(10000, 4000, len(df)).clip(3000, 25000)
    df['competitor_price'] = np.random.uniform(4.5, 145.0, len(df)).round(2)
    df['economic_index'] = np.random.normal(100, 10, len(df)).clip(80, 120)
    df['inflation_rate'] = np.random.normal(4, 1.5, len(df)).clip(2, 8)
    df['construction_index'] = np.random.normal(100, 15, len(df)).clip(70, 130)
    df['online_sales_rate'] = np.random.uniform(0.2, 0.6, len(df))
    df['store_visits'] = np.random.poisson(12000, len(df)).clip(5000, 25000)
    df['holiday_week'] = (df['ds'].dt.isocalendar().week.isin([52, 1, 25, 26])).astype(int)
    df['new_product_launch'] = np.random.choice([0, 1], len(df), p=[0.9, 0.1])
    df['lead_time_days'] = np.random.choice([14, 21, 30], len(df))
    df['current_stock'] = np.random.randint(100, 4000, len(df))
    df['supplier_delay_rate'] = np.random.uniform(0.05, 0.25, len(df))
    df['customer_reviews_score'] = np.random.uniform(3.5, 4.8, len(df)).round(1)
    df['distribution_channels'] = np.random.choice([1, 2, 3], len(df))
    df['previous_sku'] = df['sku_id']

    switch_mask = np.random.choice([0, 1], len(df), p=[0.6, 0.4])
    df.loc[switch_mask == 1, 'previous_sku'] = np.random.choice(skus, sum(switch_mask))

    df['demand_units'] = (np.random.poisson(500, len(df)) * df['season_factor'] * (1 + df['promotion_active']*0.4) * 
                          (1 + df['events_nearby']*0.15) * (df['construction_index']/100) * (1 - df['rain_days']/7 * 0.3) * 
                          (df['economic_index']/100) * (1 - df['inflation_rate']/100 * 0.2)).clip(20, 6000).astype(int)

    df['unique_id'] = df['sku_id']

    df_ts = df[['unique_id', 'ds', 'demand_units', 'price_unit', 'promotion_active', 'season_factor', 'temperature_avg', 'events_nearby', 'marketing_spend', 'competitor_price', 'economic_index', 'online_sales_rate', 'store_visits', 'holiday_week', 'lead_time_days', 'construction_index']].copy()
    df_ts = df_ts.rename(columns={'demand_units': 'y'})

    exog_cols = ['price_unit', 'promotion_active', 'season_factor', 'temperature_avg', 'events_nearby', 'marketing_spend', 'competitor_price', 'economic_index', 'online_sales_rate', 'store_visits', 'holiday_week', 'lead_time_days', 'construction_index']

    train_df = df_ts[df_ts['ds'] < '2024-01-01']

    horizon_base = 52

    models_nf = [NHITS(h= horizon_base, input_size= 52 *2,  
                       loss = DistributionLoss(distribution='Normal', level=[80, 90], return_params=True), 
                       valid_loss =  MQLoss(level = [80, 95]),
                       scaler_type = 'robust',
                       max_steps=500, 
                       n_freq_downsample=[4, 2, 1],
                       hist_exog_list = exog_cols,
                       futr_exog_list=exog_cols,
                       )]

    nf_base = NeuralForecast(models=models_nf, freq='W-MON')
    nf_base.fit(df=train_df)

    # Futr base para forecast inicial
    _, futr_base = future_exog_to_historic(df=train_df, freq= 'W-MON', features=exog_cols, h=horizon_base)

    forecast_base = nf_base.predict(futr_df=futr_base)

    return df, train_df, exog_cols, skus, nf_base, horizon_base, forecast_base

df, train_df, exog_cols, skus, nf_base, horizon_base, forecast_base = load_data_and_model()

# =======================================================
# Función para modelo dinámico
# =======================================================
@st.cache_resource
def get_nf_dynamic(horizon):
    models_nf = [NHITS(h=horizon, input_size= 52 *2,  
                       loss = DistributionLoss(distribution='Normal', level=[80, 90], return_params=True), 
                       valid_loss =  MQLoss(level = [80, 95]),
                       scaler_type = 'robust',
                       max_steps=500, 
                       n_freq_downsample=[4, 2, 1],
                       hist_exog_list = exog_cols,
                       futr_exog_list=exog_cols,
                       )]
    nf = NeuralForecast(models=models_nf, freq='W-MON')
    nf.fit(df=train_df)
    return nf

# =======================================================
# Sidebar navegación
# =======================================================
st.sidebar.title("📊 Navegación")




# 1. Obtener la sección de la URL (si existe), sino por defecto "Dashboard"
query_params = st.query_params
default_section = query_params.get("section", "Dashboard")

# 2. Definir la lista de secciones (asegúrate de que coincidan con tus 'elif')
menu_options = ["Dashboard Principal", "EDA Interactivo", "Predicciones", "Test Stress", "Análisis de Sensibilidad", 
                            "Gestión de Inventario", "Cadenas de Markov", "Monte Carlo", "Análisis de la Cesta del Mercado"]

# Si el valor en la URL no es válido, resetear a Dashboard
if default_section not in menu_options:
    default_section = "Dashboard Principal"

# 3. Crear el selector en el sidebar
section = st.sidebar.radio(
    "📍 Navegación",
    options=menu_options,
    index=menu_options.index(default_section)
)

# 4. Actualizar la URL cada vez que el usuario cambie de sección
st.query_params["section"] = section

# =======================================================
# Dashboard Principal - INTERACTIVO Y DINÁMICO
# =======================================================
if section == "Dashboard Principal":
    st.header("📈 Dashboard Ejecutivo: Estado de la Red de Ferreterías")

    # 1. CÁLCULO DE MÉTRICAS GLOBALES (Siempre visibles)
    total_demand = df['demand_units'].sum()
    monthly_agg = df.groupby(df['ds'].dt.to_period('M'))['demand_units'].sum()
    growth = ((monthly_agg.iloc[-1] / monthly_agg.iloc[-2]) - 1) * 100 if len(monthly_agg) > 1 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Demanda Total", f"{total_demand:,.0f} u", f"{growth:.1f}% vs Mes Ant.")
    with col2:
        st.metric("Demanda Promedio", f"{df['demand_units'].mean():.1f} u/sem")
    with col3:
        top_sku = df.groupby('sku_id')['demand_units'].sum().idxmax()
        st.metric("SKU Líder", top_sku)
    with col4:
        st.metric("SKUs Activos", len(skus))

    st.divider()

    # --- NUEVA SECCIÓN DE CONTROL INTERACTIVO ---
    st.subheader("🛠️ Controles de Análisis Dinámico")
    c_ctrl1, c_ctrl2 = st.columns([2, 1])
    
    with c_ctrl1:
        # Selector de SKU con opción "Todos"
        options = ["TODOS LOS SKUS"] + skus
        selection = st.selectbox("Seleccione SKU para análisis detallado:", options)
    
    with c_ctrl2:
        # Slider para Promedio Móvil
        window_size = st.slider("Ventana de Promedio Móvil (Semanas):", 2, 12, 4, 
                               help="Ajusta qué tan 'suave' quieres ver la línea de tendencia.")

    # Filtrado de datos basado en selección
    if selection == "TODOS LOS SKUS":
        plot_df = df.groupby('ds')['demand_units'].sum().reset_index()
        chart_title = f"Tendencia Global de Demanda (Promedio Móvil {window_size}s)"
    else:
        plot_df = df[df['sku_id'] == selection].groupby('ds')['demand_units'].sum().reset_index()
        chart_title = f"Tendencia para {selection} (Promedio Móvil {window_size}s)"

    # Cálculo del promedio móvil dinámico
    plot_df['MA'] = plot_df['demand_units'].rolling(window=window_size).mean()

    # --- FILA 2: GRÁFICO DE TENDENCIA DINÁMICO ---
    st.subheader(chart_title)
    fig_trend = go.Figure()
    
    # Línea de Demanda Real
    fig_trend.add_trace(go.Scatter(
        x=plot_df['ds'], y=plot_df['demand_units'], 
        name='Demanda Real', 
        line=dict(color='#cbd5e1', width=1.5, shape='spline')
    ))
    
    # Línea de Promedio Móvil
    fig_trend.add_trace(go.Scatter(
        x=plot_df['ds'], y=plot_df['MA'], 
        name=f'Tendencia (MA {window_size})', 
        line=dict(color='#1e3a8a', width=3.5)
    ))

    fig_trend.update_layout(
        template="plotly_white", 
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    

    st.divider()

    # --- FILA 3: PARETO Y CORRELACIÓN ---
    c_pareto, c_corr = st.columns([1, 1])

    with c_pareto:
        st.subheader("⚖️ Distribución de Pareto")
        pareto_df = df.groupby('sku_id')['demand_units'].sum().sort_values(ascending=False).reset_index()
        pareto_df['cum_pct'] = (pareto_df['demand_units'].cumsum() / pareto_df['demand_units'].sum()) * 100
        
        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Bar(x=pareto_df['sku_id'], y=pareto_df['demand_units'], name="Demanda", marker_color='#1e3a8a'))
        fig_pareto.add_trace(go.Scatter(x=pareto_df['sku_id'], y=pareto_df['cum_pct'], name="% Acumulado", 
                                        yaxis="y2", line=dict(color='#ef4444', width=2.5)))
        
        fig_pareto.update_layout(
            template="plotly_white",
            yaxis=dict(title="Unidades"),
            yaxis2=dict(title="% Acumulado", overlaying="y", side="right", range=[0, 105]),
            xaxis_tickangle=45,
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

    with c_corr:
        st.subheader("🔗 Correlaciones Exógenas")
        # Si hay un SKU seleccionado, correlacionar solo para ese SKU
        corr_df = df if selection == "TODOS LOS SKUS" else df[df['sku_id'] == selection]
        main_vars = ['demand_units', 'price_unit', 'promotion_active', 'construction_index', 'temperature_avg', 'marketing_spend']
        corr = corr_df[main_vars].corr()
        
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r', aspect="auto")
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

    # Insight Adaptativo
    if selection != "TODOS LOS SKUS":
        max_week = plot_df.loc[plot_df['demand_units'].idxmax()]['ds'].strftime('%Y-%W')
        st.info(f"**Insight para {selection}:** El pico histórico de demanda ocurrió en la semana **{max_week}**. La ventana de promedio móvil de **{window_size} semanas** sugiere que la variabilidad a corto plazo es {'alta' if window_size < 4 else 'estable'}.")


# =======================================================
# EDA Interactivo - VERSIÓN CON ANALÍTICA PRESCRIPTIVA
# =======================================================
elif section == "EDA Interactivo":
    st.header("🔍 Diagnóstico de Demanda y Recomendaciones Tácticas")

    if not skus:
        st.error("No se encontraron SKUs cargados.")
    else:
        selected_sku = st.selectbox("🎯 Seleccione el SKU a analizar:", skus)
        sku_df = df[df['sku_id'] == selected_sku].copy()
        
        if sku_df.empty:
            st.warning(f"No hay datos disponibles para el SKU: {selected_sku}")
        else:
            # --- TABS PARA ORGANIZACIÓN ---
            tab_time, tab_drivers, tab_dist = st.tabs([
                "📈 Evolución Temporal", 
                "🚦 Drivers de Demanda", 
                "📊 Distribución y Riesgo"
            ])

            # --- PESTAÑA 1: EVOLUCIÓN TEMPORAL ---
            with tab_time:
                st.subheader(f"Análisis de Ciclos y Tendencias: {selected_sku}")
                ma_window = st.slider("Ventana de Suavizado (Semanas):", 2, 12, 4)
                sku_df['MA'] = sku_df['demand_units'].rolling(window=ma_window).mean()
                
                fig_time = go.Figure()
                fig_time.add_trace(go.Scatter(x=sku_df['ds'], y=sku_df['demand_units'], 
                                             name="Venta Real", line=dict(color='#94a3b8', width=1.5)))
                fig_time.add_trace(go.Scatter(x=sku_df['ds'], y=sku_df['MA'], 
                                             name="Tendencia Limpia", line=dict(color='#1e3a8a', width=3.5)))
                fig_time.update_layout(template="plotly_white", hovermode="x unified", height=450)
                st.plotly_chart(fig_time, use_container_width=True)

                # --- INSIGHTS DE TOMA DE DECISIONES ---
                st.markdown("### 💡 Guía de Decisión (Temporalidad)")
                last_ma = sku_df['MA'].iloc[-1]
                prev_ma = sku_df['MA'].iloc[-ma_window-1]
                trend_status = "CRECIENTE" if last_ma > prev_ma else "DECRECIENTE"
                color_trend = "green" if trend_status == "CRECIENTE" else "red"

                st.info(f"""
                **1. Estado de la Tendencia:** La demanda actual es **:{color_trend}[{trend_status}]**. 
                * **Acción:** {'Aumentar órdenes de compra para evitar quiebres de stock.' if trend_status == "CRECIENTE" else 'Reducir el ritmo de reabastecimiento para evitar exceso de inventario y costos de almacenamiento.'}
                
                **2. Estacionalidad Detectada:** Observe los valles y picos. Si el patrón se repite anualmente, planifique sus compras con un **Lead Time** de anticipación de al menos 4 semanas antes del pico visualizado.
                """)

            # --- PESTAÑA 2: DRIVERS DE DEMANDA ---
            with tab_drivers:
                st.subheader("Análisis de Factores Críticos (Drivers)")
                col_d1, col_d2 = st.columns(2)
                
                with col_d1:
                    fig_price = px.scatter(sku_df, x='price_unit', y='demand_units', 
                                           trendline="ols", color='promotion_active',
                                           title="Sensibilidad al Precio (Elasticidad)",
                                           color_discrete_map={0: '#64748b', 1: '#ef4444'})
                    st.plotly_chart(fig_price, use_container_width=True)
                    
                with col_d2:
                    fig_ext = px.scatter(sku_df, x='construction_index', y='demand_units', 
                                         size='store_visits', color='temperature_avg',
                                         title="Impacto Entorno Económico y Clima",
                                         color_continuous_scale='Viridis')
                    st.plotly_chart(fig_ext, use_container_width=True)
                
                # --- INSIGHTS DE TOMA DE DECISIONES ---
                st.markdown("### 💡 Guía de Decisión (Drivers)")
                corr_price = sku_df['demand_units'].corr(sku_df['price_unit'])
                
                st.success(f"""
                **1. Estrategia de Precios:** La correlación precio-demanda es de **{corr_price:.2f}**.
                * **Decisión:** {'Alta Elasticidad: Las promociones de precio son MUY efectivas para mover este volumen.' if corr_price < -0.4 else 'Baja Elasticidad: El cliente compra este SKU por necesidad, no por precio. Evite descuentos agresivos; no dispararán la demanda significativamente.'}
                
                **2. Indicadores Externos:** Si las burbujas grandes (más visitas) coinciden con un alto Índice de Construcción, este SKU debe ser el protagonista de su publicidad en redes sociales cuando el sector construcción reporte crecimiento mensual.
                """)

            # --- PESTAÑA 3: DISTRIBUCIÓN Y RIESGO ---
            with tab_dist:
                st.subheader("Análisis de Riesgo de Abastecimiento")
                col_v1, col_v2 = st.columns([1, 2])
                
                with col_v1:
                    fig_hist = px.histogram(sku_df, x='demand_units', 
                                            title="Estabilidad de la Demanda",
                                            color_discrete_sequence=['#1e3a8a'])
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col_v2:
                    fig_box = px.box(df, x='sku_id', y='demand_units', color='promotion_active',
                                     title="Distribución y Valores Atípicos (Outliers)")
                    fig_box.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # --- INSIGHTS DE TOMA DE DECISIONES ---
                st.markdown("### 💡 Guía de Decisión (Riesgo)")
                cv = sku_df['demand_units'].std() / sku_df['demand_units'].mean()
                
                if cv < 0.3:
                    risk_msg = "BAJO (Producto Estable). Use modelos de reposición automáticos."
                    action_msg = "Mantenga un stock de seguridad mínimo (10-15%)."
                elif cv < 0.6:
                    risk_msg = "MEDIO (Producto Volátil). Requiere supervisión quincenal."
                    action_msg = "Aumente el stock de seguridad al 25% y vigile las promociones de la competencia."
                else:
                    risk_msg = "ALTO (Producto Intermitente/Errático). ¡Peligro de Stockout o Sobre-stock!"
                    action_msg = "No use promedios simples. Use el P95 de la simulación Monte Carlo para definir sus compras."

                st.warning(f"""
                **1. Nivel de Riesgo Operativo:** **{risk_msg}**
                * **Acción Recomendada:** {action_msg}
                
                **2. Gestión de Outliers:** Los puntos aislados en el gráfico de cajas representan ventas inusuales (proyectos especiales). No base su compra del próximo mes en estos picos, o terminará con capital inmovilizado.
                """)

# =======================================================
# Forecast
# =======================================================
elif section == "Predicciones":
    st.header("🔮 Pronóstico Automatizado de Demanda")
    st.markdown("Este módulo utiliza IA para proyectar la demanda futura basándose en el historial y patrones estacionales.")
    
    # 1. Selectores de alta precisión
    c_p1, c_p2 = st.columns([2, 1])
    with c_p1:
        selected_sku = st.selectbox("🎯 Seleccionar SKU para Forecast:", skus)
    with c_p2:
        # Horizonte extendido según tu requerimiento
        horizon_selected = st.select_slider("📅 Horizonte de Predicción (Semanas):", 
                                            options=[4, 12, 18, 24, 36, 48, 60, 72, 84], value=12)

    st.divider()

    # 2. Lógica de Predicción
    if st.button("🚀 Generar Forecast", use_container_width=True):
        with st.spinner(f"Sincronizando modelos neuronales para {selected_sku}..."):
            
            # Generar exógenas futuras usando la función técnica
            # Asumimos que train_df ya está preparado con unique_id, ds, y
            train1, futr_df1 = future_exog_to_historic(
                df = train_df,
                freq = 'W-MON',
                features = exog_cols,
                h = horizon_selected,
            )

            # Obtener modelo dinámico y predecir
            nf = get_nf_dynamic(horizon_selected)
            forecast_raw = nf.predict(futr_df=futr_df1)
            
            # Filtrar resultados para el SKU seleccionado
            f_sku = forecast_raw[forecast_raw['unique_id'] == selected_sku]
            h_sku = train_df[train_df['unique_id'] == selected_sku]

            # 3. Visualización de Línea de Vida Completa
            fig = go.Figure()
            
            # Histórico (Gris para contexto)
            fig.add_trace(go.Scatter(x=h_sku['ds'][-100:], y=h_sku['y'][-100:], 
                                     name='Demanda Histórica', line=dict(color='#94a3b8', width=1.1)))
            
            # Intervalo de Confianza (Sombreado de riesgo)
            fig.add_trace(go.Scatter(x=f_sku['ds'], y=f_sku['NHITS-hi-90'], mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=f_sku['ds'], y=f_sku['NHITS-lo-90'], mode='lines', line=dict(width=0), 
                                     fill='tonexty', fillcolor='rgba(37, 99, 235, 0.15)', name='Incertidumbre (90%)'))
            
            # Forecast Central (Azul para acción)
            fig.add_trace(go.Scatter(x=f_sku['ds'], y=f_sku['NHITS'], 
                                     name='Forecast IA', line=dict(color='#2563eb', width=1.5)))

            fig.update_layout(
                title=f"Proyección de Ventas: {selected_sku}",
                template="plotly_white",
                hovermode="x unified",
                yaxis=dict(title="Unidades", tickformat=".0f", autorange=True),
                xaxis=dict(title="Línea de Tiempo", rangeslider=dict(visible=True)),
                height=550,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

            # 4. Métricas de Impacto Financiero y Operativo
            # Comparamos el forecast contra el mismo periodo de tiempo hacia atrás
            m_avg_hist = h_sku['y'].tail(horizon_selected).mean()
            m_avg_pred = f_sku['NHITS'].mean()
            
            total_vol = f_sku['NHITS'].sum()
            net_change = total_vol - h_sku['y'].tail(horizon_selected).sum()
            growth_pct = ((m_avg_pred - m_avg_hist) / (m_avg_hist + 1e-6)) * 100

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Volumen Total", f"{total_vol:,.0f} u")
            c2.metric("Impacto Neto", f"{net_change:,.0f} u", delta=f"{net_change:,.0f}")
            c3.metric("Crecimiento", f"{growth_pct:.1f}%", delta=f"{growth_pct:.1f}%")
            c4.metric("Confianza", "90%", help="Basado en NHITS Distribution Loss")

            # 5. Insights y Toma de Decisiones
            st.divider()
            col_in1, col_in2 = st.columns(2)
            
            with col_in1:
                st.subheader("🌐 Diagnóstico de Red")
                st.write(f"""
                Para el SKU **{selected_sku}**, la IA proyecta una demanda acumulada de **{total_vol:,.0f} unidades**. 
                Comparado con el periodo anterior, esto representa una variación neta de **{net_change:,.0f} unidades**.
                
                **Uso sugerido:** Este volumen define la necesidad de espacio en bodega y la negociación de fletes por volumen.
                """)

            with col_in2:
                st.subheader("🎯 Acción Recomendada")
                if growth_pct > 7:
                    st.success(f"**ALERTA DE CRECIMIENTO:** Se proyecta un alza del {growth_pct:.1f}%.")
                    st.markdown(f"**Decisión:** Aumentar stock de seguridad en un **15%** y priorizar este SKU en el próximo ciclo de compras.")
                elif growth_pct < -7:
                    st.error(f"**ALERTA DE CONTRACCIÓN:** Caída estimada del {abs(growth_pct):.1f}%.")
                    st.markdown(f"**Decisión:** Reducir órdenes de reabastecimiento para evitar capital inmovilizado y posible obsolescencia.")
                else:
                    st.info("**DEMANDA ESTABLE:** Comportamiento orgánico sin desviaciones críticas.")
                    st.markdown(f"**Decisión:** Mantener el Plan de Abastecimiento Actual (BAU) y optimizar frecuencia de entrega.")

            with st.expander("📝 Ver Dataframe de Predicciones Detalladas"):
                st.dataframe(f_sku[['ds', 'NHITS', 'NHITS-lo-90', 'NHITS-hi-90']].rename(
                    columns={'NHITS': 'Predicción Central', 'NHITS-lo-90': 'Límite Inferior', 'NHITS-hi-90': 'Límite Superior'}
                ), use_container_width=True)


# =======================================================
# Test Stress
# =======================================================

# =======================================================
# Prueba de Stress - Escenarios Dinámicos
# =======================================================
elif section == "Test Stress":
    st.header("🔮 Simulación de Escenarios y Stress Test")
    st.info("Ajuste las variables externas para observar el impacto proyectado en la demanda futura.")

    # 1. Identificación automática de la columna de demanda (evita KeyError)
    col_y = 'y' if 'y' in train_df.columns else ('demand_kg' if 'demand_kg' in train_df.columns else train_df.select_dtypes(include=[np.number]).columns[0])

    # --- CONFIGURACIÓN INICIAL ---
    col_c1, col_c2 = st.columns([1, 1])
    with col_c1:
        selected_sku = st.selectbox("🎯 Seleccionar SKU para Stress Test:", skus)
    with col_c2:
        horizon_selected = st.slider("📅 Horizonte de Proyección (semanas):", 12, 104, 52)

    # --- CONFIGURACIÓN DE SHOCKS ---
    st.subheader("🛠️ Panel de Shocks Exógenos")
    st.markdown("Seleccione qué variables desea estresar y en qué magnitud (1.0 = Sin cambio).")
    
    selected_exogs = st.multiselect(
        "Variables a estresar:", 
        options=exog_cols,
        default=exog_cols[:2] if len(exog_cols) > 1 else exog_cols
    )

    shocks = {}
    if selected_exogs:
        # Mostramos los sliders de forma organizada en columnas
        cols_s = st.columns(len(selected_exogs))
        for i, exog in enumerate(selected_exogs):
            with cols_s[i]:
                shocks[exog] = st.slider(f"Shock {exog}", 0.5, 1.5, 1.0, key=f"shock_{exog}")
    else:
        st.warning("⚠️ Seleccione al menos una variable para aplicar un escenario de stress.")

    # --- EJECUCIÓN DEL MODELO ---
    if st.button("🚀 Ejecutar Simulación de Escenario"):
        with st.spinner("Calculando impacto dinámico..."):
            try:
                # A. Obtener modelo dinámico
                nf_dynamic = get_nf_dynamic(horizon_selected)
                
                # B. Construcción del futuro base (usando tu función auxiliar para consistencia)
                _, futr_df_dynamic = future_exog_to_historic(df=train_df, freq='W-MON', features=exog_cols, h=horizon_selected)
                
                # C. Aplicar los shocks seleccionados
                for exog_name, multiplier in shocks.items():
                    futr_df_dynamic[exog_name] *= multiplier

                # D. Predicción
                forecast_dynamic = nf_dynamic.predict(futr_df=futr_df_dynamic)
                
                # E. Filtrado para el SKU seleccionado
                forecast_sku = forecast_dynamic[forecast_dynamic['unique_id'] == selected_sku].copy()
                historical = train_df[train_df['unique_id'] == selected_sku].tail(52)

                # --- VISUALIZACIÓN ---
                tab_viz, tab_dec = st.tabs(["📉 Visualización del Impacto", "💡 Análisis de Decisiones"])

                with tab_viz:
                    fig = go.Figure()

                    # Intervalos de Confianza (Banda de Sombra)
                    if 'NHITS-hi-90' in forecast_sku.columns:
                        fig.add_trace(go.Scatter(
                            x=forecast_sku['ds'], y=forecast_sku['NHITS-hi-90'],
                            fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False
                        ))
                        fig.add_trace(go.Scatter(
                            x=forecast_sku['ds'], y=forecast_sku['NHITS-lo-90'],
                            fill='tonexty', mode='lines', fillcolor='rgba(255, 69, 0, 0.1)', 
                            line_color='rgba(0,0,0,0)', name='Rango de Incertidumbre (90%)'
                        ))

                    # Línea Histórica
                    fig.add_trace(go.Scatter(x=historical['ds'], y=historical[col_y], name='Histórico Real', line=dict(color='#475569')))
                    
                    # Línea Proyectada (Stress Test)
                    fig.add_trace(go.Scatter(x=forecast_sku['ds'], y=forecast_sku['NHITS'], 
                                           name='Predicción Estresada', line=dict(color='#ef4444', width=3)))

                    fig.update_layout(
                        title=f"Stress Test: {selected_sku} (Horizonte {horizon_selected} sem)",
                        template="plotly_white",
                        hovermode="x unified",
                        xaxis_title="Línea de Tiempo",
                        yaxis_title = f"{selected_sku}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    

                with tab_dec:
                    # Métricas Resumen
                    avg_pred = forecast_sku['NHITS'].mean()
                    total_pred = forecast_sku['NHITS'].sum()
                    last_real = historical[col_y].iloc[-1]
                    variacion = ((avg_pred / last_real) - 1) * 100 if last_real > 0 else 0

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Promedio Simulado", f"{avg_pred:,.0f} kg", delta=f"{variacion:.1f}% vs actual")
                    m2.metric("Volumen Total", f"{total_pred:,.0f} kg")
                    m3.metric("Pico de Demanda", f"{forecast_sku['NHITS'].max():,.0f} kg")

                    st.divider()
                    
                    # Lógica Consultiva
                    c_left, c_right = st.columns(2)
                    with c_left:
                        st.markdown("### 🛡️ Plan de Mitigación")
                        if variacion > 15:
                            st.error(f"**Alerta de Abastecimiento:** El escenario muestra un alza crítica. Se recomienda aumentar el cupo con proveedores en un **{variacion*0.8:.0f}%**.")
                        elif variacion < -15:
                            st.warning("**Riesgo de Sobre-stock:** El escenario sugiere una caída fuerte. Reduzca la frecuencia de pedidos para evitar mermas.")
                        else:
                            st.success("**Operación Estable:** Los shocks aplicados no desestabilizan drásticamente la operación actual.")

                    with c_right:
                        st.markdown("### 💰 Impacto Financiero")
                        st.write(f"Bajo este escenario de stress, el SKU **{selected_sku}** requerirá una gestión de flujo de caja para cubrir un total de **{total_pred:,.0f} kg** en el periodo seleccionado.")

                # Tabla detallada al final
                with st.expander("Ver Tabla de Datos Proyectados"):
                    st.dataframe(forecast_sku[['ds', 'NHITS']].rename(columns={'NHITS': 'Demanda Estresada'}).style.format(precision=0))

            except Exception as e:
                st.error(f"Ocurrió un error al procesar la simulación: {e}")
                st.info("Esto puede deberse a que el horizonte seleccionado es mayor al permitido por el modelo o faltan datos exógenos.")
    else:
        st.info("💡 Configure los shocks arriba y presione el botón para generar el escenario.")

# =======================================================
# Análisis de Sensibilidad - Escenarios de Estrés IA
# =======================================================
elif section == "Análisis de Sensibilidad":

    st.header("📊 Análisis de Sensibilidad y Elasticidad")
    st.markdown("Proyecta cómo cambia la demanda del SKU cuando manipulas variables clave.")

    # --- 1. CONTROLES EN COLUMNAS ---
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        selected_sku_sens = st.selectbox("🎯 Seleccionar SKU:", skus)
        variable_sens = st.selectbox("🧪 Variable a Testear:", exog_cols)
    
    with col_ctrl2:
        range_min = st.slider("📉 Límite Inferior", 0.5, 1.0, 0.8)
        range_max = st.slider("📈 Límite Superior", 1.0, 2.0, 1.2)

    # --- 2. CÁLCULO DE SENSIBILIDAD ---
    multipliers = np.linspace(range_min, range_max, 10)
    sensitivities = []
    base_mean = forecast_base[forecast_base['unique_id'] == selected_sku_sens]['NHITS'].mean()

    with st.spinner("Calculando elasticidad..."):
        for mult in multipliers:
            _, futr_sens = future_exog_to_historic(df=train_df, freq='W-MON', features=exog_cols, h = horizon_base)
            futr_sens[variable_sens] *= mult
            forecast_sens = nf_base.predict(futr_df=futr_sens)
            sku_mean = forecast_sens[forecast_sens['unique_id'] == selected_sku_sens]['NHITS'].mean()
            sensitivities.append(sku_mean)

    # --- 3. VISUALIZACIÓN ---
    fig_sens = px.line(x=multipliers, y=sensitivities, 
                       title=f"Curva de Respuesta: {selected_sku_sens} vs {variable_sens}",
                       markers=True, color_discrete_sequence=['#1e3a8a'])
    fig_sens.add_hline(y=base_mean, line_dash="dash", line_color="red", annotation_text="Base (1.0x)")
    fig_sens.update_layout(template="plotly_white", xaxis_title=f"Multiplicador de {variable_sens}", yaxis_title="Demanda (kg)")
    st.plotly_chart(fig_sens, use_container_width=True)

    # --- 4. MÉTRICAS DE IMPACTO ---
    st.subheader("⚖️ Impacto Comparativo")
    m1, m2, m3 = st.columns(3)
    val_min, val_max = sensitivities[0], sensitivities[-1]
    
    m1.metric(f"Escenario Mínimo ({range_min}x)", f"{val_min:,.0f}", f"{val_min - base_mean:+.0f}", delta_color="normal")
    m2.metric("Demanda Base (1.0x)", f"{base_mean:,.0f}")
    m3.metric(f"Escenario Máximo ({range_max}x)", f"{val_max:,.0f}", f"{val_max - base_mean:+.0f}", delta_color="normal")

    # --- 5. INSIGHTS DETALLADOS POR ESCENARIO ---
    st.divider()
    st.subheader(f"💡 Diagnóstico Estratégico: {selected_sku_sens}")
    
    col_down, col_up = st.columns(2)
    
    with col_down:
        st.markdown(f"#### 🔽 Al bajar {variable_sens} ({range_min}x)")
        diff_down = ((val_min - base_mean) / base_mean) * 100
        
        if diff_down < -5:
            st.error(f"**Impacto Negativo:** La demanda cae un **{abs(diff_down):.1f}%**.")
            st.write(f"Para el SKU **{selected_sku_sens}**, una reducción en esta variable es crítica. "
                     f"Se recomienda evitar este escenario a menos que se busque reducir inventario de forma agresiva.")
        elif diff_down > 5:
            st.success(f"**Impacto Positivo:** La demanda sube un **{diff_down:.1f}%**.")
            st.write(f"Este SKU presenta una relación inversa con {variable_sens}. "
                     f"Reducir la variable actúa como un catalizador de ventas. Úselo para liquidar stock excedente.")
        else:
            st.info("**Impacto Neutro:** El SKU muestra resiliencia a la baja. Los cambios menores no afectarán su rotación.")

    with col_up:
        st.markdown(f"#### 🔼 Al subir {variable_sens} ({range_max}x)")
        diff_up = ((val_max - base_mean) / base_mean) * 100
        
        if diff_up > 5:
            st.success(f"**Impacto Positivo:** La demanda sube un **{diff_up:.1f}%**.")
            st.write(f"El SKU **{selected_sku_sens}** es altamente reactivo al incremento de {variable_sens}. "
                     f"**Acción:** Asegure contratos de suministro para cubrir el pico de **{val_max - base_mean:,.0f} kg** adicionales.")
        elif diff_up < -5:
            st.error(f"**Impacto Negativo:** La demanda cae un **{abs(diff_up):.1f}%**.")
            st.write(f"Cuidado: Incrementar esta variable castiga el volumen de {selected_sku_sens}. "
                     f"Evalúe si el margen unitario compensa la pérdida de **{abs(val_max - base_mean):,.0f} kg** de mercado.")
        else:
            st.info("**Impacto Neutro:** El SKU es inelástico al alza. Puede subir la variable sin temor a perder volumen de ventas significativo.")

    # Resumen de Elasticidad
    st.info(f"**Resumen de Elasticidad:** El SKU {selected_sku_sens} tiene una sensibilidad total de **{abs(diff_up - diff_down):.1f}%** "
            f"dentro del rango seleccionado. Entre más alto el porcentaje, más peligroso es mover {variable_sens} sin un plan de contingencia.")


# =======================================================
# Gestión de Inventario - Inteligencia de Suministro
# =======================================================
elif section == "Gestión de Inventario":
    st.header("📦 Torre de Control de Inventarios (Supply Chain)")
    st.markdown("""
    Optimización de niveles de stock basada en la variabilidad de la demanda predicha por **NHITS** y tiempos de entrega (Lead Times).
    """)

    # --- 1. PROCESAMIENTO DE DATOS DE INVENTARIO ---
    forecast_base['sku_id'] = forecast_base['unique_id']
    inventory_summary = forecast_base.groupby('sku_id')['NHITS'].agg(['mean', 'std']).reset_index()
    inventory_summary = inventory_summary.rename(columns={'mean': 'demand_forecast_mean', 'std': 'demand_forecast_std'})

    lead_time_avg = df.groupby('sku_id')['lead_time_days'].mean().reset_index()
    
    # Obtenemos el último snapshot para stock actual y precio
    last_snapshot = df[df['ds'] == df['ds'].max()][['sku_id', 'demand_units', 'price_unit']]
    last_snapshot = last_snapshot.rename(columns={'demand_units': 'current_stock'})

    inventory_summary = inventory_summary.merge(lead_time_avg, on='sku_id').merge(last_snapshot, on='sku_id')

    # --- CÁLCULOS ESTADÍSTICOS ---
    # Stock de Seguridad (SS) y Punto de Reorden (ROP)
    inventory_summary['safety_stock'] = (1.65 * inventory_summary['demand_forecast_std'] * np.sqrt(inventory_summary['lead_time_days'] / 7)).round(0)
    inventory_summary['reorder_point'] = (inventory_summary['demand_forecast_mean'] * (inventory_summary['lead_time_days'] / 7) + inventory_summary['safety_stock']).round(0)
    
    # EOQ
    inventory_summary['eoq'] = np.sqrt((2 * (inventory_summary['demand_forecast_mean'] * 52) * 100) / (inventory_summary['price_unit'] * 0.20 + 1e-6)).round(0)
    
    # Índice de Salud y Valorización
    inventory_summary['health_idx'] = (inventory_summary['current_stock'] / (inventory_summary['reorder_point'] + 1e-6)).round(2)
    inventory_summary['valor_total_stock'] = inventory_summary['current_stock'] * inventory_summary['price_unit']

    # Lógica de Categorización y Acción
    def categorizar_inventario(row):
        if row['health_idx'] < 1.0:
            return 'Riesgo (Bajo Stock)', f"PEDIR {row['eoq']:.0f} u."
        if row['health_idx'] > 2.5:
            return 'Exceso (Inmovilizado)', "FRENAR COMPRAS"
        return 'Óptimo (Saludable)', "MANTENER"

    inventory_summary[['categoria_salud', 'accion_sugerida']] = inventory_summary.apply(
        lambda x: pd.Series(categorizar_inventario(x)), axis=1
    )

    # --- 2. INTERFAZ DE USUARIO ---
    tab_overview, tab_viz, tab_finance = st.tabs(["📋 Auditoría y Acciones", "📊 Visualización de Salud", "💰 Análisis de Capital"])

    with tab_overview:
        st.subheader("Matriz de Operaciones por SKU")
        
        # Columnas solicitadas incluyendo las estadísticas
        cols_mostrar = [
            'sku_id', 'demand_forecast_mean', 'demand_forecast_std', 'lead_time_days',
            'current_stock', 'safety_stock', 'reorder_point', 'eoq', 
            'health_idx', 'accion_sugerida', 'categoria_salud'
        ]

        def highlight_inventory(row):
            if row['categoria_salud'] == 'Riesgo (Bajo Stock)': return ['background-color: #B31626; color: white'] * len(row)
            if row['categoria_salud'] == 'Exceso (Inmovilizado)': return ['background-color: #FFDE21; color: black'] * len(row)
            return ['background-color: #00913F; color: white'] * len(row)

        st.dataframe(
            inventory_summary[cols_mostrar].style.apply(highlight_inventory, axis=1)
            .format({
                "demand_forecast_mean": "{:.2f}",
                "demand_forecast_std": "{:.2f}",
                "health_idx": "{:.2f}",
                "lead_time_days": "{:.1f}"
            }),
            use_container_width=True,
            column_config={"categoria_salud": None} 
        )

        # --- SECCIÓN DE INSIGHTS BASADOS EN ACCIONES ---
        st.divider()
        st.subheader("💡 Insights Estratégicos")
        
        c_ins1, c_ins2, c_ins3 = st.columns(3)
        
        # Conteo de acciones
        n_pedir = len(inventory_summary[inventory_summary['categoria_salud'] == 'Riesgo (Bajo Stock)'])
        n_frenar = len(inventory_summary[inventory_summary['categoria_salud'] == 'Exceso (Inmovilizado)'])
        
        with c_ins1:
            if n_pedir > 0:
                st.error(f"⚠️ **Prioridad de Abastecimiento**\n\nTienes {n_pedir} SKUs por debajo del punto de reorden. La variabilidad (Std Dev) sugiere que podrías enfrentar quiebres en menos de 7 días si no ejecutas los pedidos EOQ sugeridos.")
            else:
                st.success("✅ **Nivel de Servicio Seguro**\n\nNo hay riesgos de quiebre inmediatos para los SKUs actuales.")

        with c_ins2:
            if n_frenar > 0:
                st.warning(f"📉 **Optimización de Flujo**\n\nHay {n_frenar} SKUs en exceso. Se recomienda suspender órdenes de compra y evaluar promociones cruzadas (ver Análisis de Cesta) para liberar capital.")
            else:
                st.info("📦 **Eficiencia de Stock**\n\nTu inventario está bien balanceado; no se detecta capital ocioso significativo.")

        with c_ins3:
            avg_lead = inventory_summary['lead_time_days'].mean()
            st.write(f"⏱️ **Análisis de Lead Time**\n\nEl tiempo promedio de respuesta es de {avg_lead:.1f} días. Los SKUs con mayor desviación estándar requieren una revisión de proveedores para estabilizar el flujo.")

    with tab_viz:
        st.subheader("Estado Crítico: Stock vs Reorder Point")
        fig_inv = go.Figure()
        fig_inv.add_trace(go.Bar(x=inventory_summary['sku_id'], y=inventory_summary['current_stock'], name="Stock Actual", marker_color='#1e3a8a'))
        fig_inv.add_trace(go.Scatter(x=inventory_summary['sku_id'], y=inventory_summary['reorder_point'], name="Punto de Reorden (Gatillo)", mode='markers', marker=dict(color='#dc2626', size=12, symbol='x')))
        fig_inv.update_layout(template="plotly_white", xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig_inv, use_container_width=True)
        

    with tab_finance:
        st.subheader("Distribución de Capital por Categoría de Salud")
        col_pie, col_metrics = st.columns([2, 1])
        with col_pie:
            fig_pie = px.pie(inventory_summary, values='valor_total_stock', names='categoria_salud',
                             color='categoria_salud',
                             color_discrete_map={'Riesgo (Bajo Stock)': '#B31626', 'Exceso (Inmovilizado)': '#FFDE21', 'Óptimo (Saludable)': '#00913F'},
                             hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_metrics:
            total_money = inventory_summary['valor_total_stock'].sum()
            frozen_money = inventory_summary[inventory_summary['categoria_salud'] == 'Exceso (Inmovilizado)']['valor_total_stock'].sum()
            st.metric("Inversión Total", f"${total_money:,.0f}")
            st.metric("Capital Inmovilizado", f"${frozen_money:,.0f}", delta=f"{(frozen_money/total_money)*100:.1f}%", delta_color="inverse")

            

# =======================================================
# Análisis de Cadenas de Markov 
# =======================================================

elif section == "Cadenas de Markov":
    st.header("🔗 Inteligencia de Cadenas de Markov")
    st.markdown("""
    Este análisis revela la **probabilidad de transición**: la posibilidad de que un cliente que compró un SKU 
    vuelva a comprar el mismo (Lealtad) o cambie a otro (Switching/Canibalización).
    """)

    # --- 1. PROCESAMIENTO MATEMÁTICO ---
    transition_counts = pd.crosstab(df['previous_sku'], df['sku_id'])
    transition_counts = transition_counts.reindex(index=skus, columns=skus, fill_value=0)
    row_sums = transition_counts.sum(axis=1)
    transition_matrix = transition_counts.div(row_sums.replace(0, 1), axis=0).fillna(0)
    
    for s in skus:
        if row_sums[s] == 0:
            transition_matrix.loc[s, s] = 1.0

    # --- 2. INTERFAZ POR PESTAÑAS ---
    tab_math, tab_graph, tab_inv, tab_mkt, tab_plan = st.tabs([
        "🧮 Construcción", 
        "🕸️ Mapa de Flujo", 
        "📦 Inventario", 
        "🎯 Marketing", 
        "⚙️ Planificación Operativa"
    ])

    # --- TAB 1 & 2: MANTENIDAS SEGÚN TU SOLICITUD ---
    with tab_math:
        st.subheader("Laboratorio de Probabilidades")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.write("**1. Conteos Absolutos (Frecuencia)**")
            st.dataframe(transition_counts, use_container_width=True)
        with col_m2:
            st.write("**2. Sumas por Origen**")
            st.dataframe(row_sums.to_frame('Ventas Totales'), use_container_width=True)
        st.divider()
        st.write("**3. Matriz de Probabilidad de Transición final ($P$)**")
        st.dataframe(transition_matrix.round(3).style.background_gradient(cmap='Greens', axis=None), use_container_width=True)

    with tab_graph:
        st.subheader("🕸️ Mapa de Flujo Selectivo")
        
        # --- FILTRO DE ENFOQUE ---
        selected_skus_graph = st.multiselect(
            "Seleccionar SKUs para visualizar conexiones:",
            options=skus,
            default=skus[:5] 
        )

        if not selected_skus_graph:
            st.warning("Selecciona al menos un SKU para generar el diagrama.")
        else:
            import graphviz

            # Paleta de colores vibrante
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            color_map = {sku: colors[i % len(colors)] for i, sku in enumerate(skus)}
            
            # Configuración del Grafo Graphviz
            dot = graphviz.Digraph(comment='Markov Focused')
            dot.attr(
                layout='dot',        
                rankdir='LR',        
                bgcolor='transparent',
                overlap='false',
                splines='true'
            )
            
            # 1. Crear Nodos
            for sku in selected_skus_graph:
                display_name = sku.replace("SKU_", "Prod. ")
                dot.node(sku, display_name, 
                         shape='circle', style='filled', 
                         fillcolor='#262730', color=color_map.get(sku), # Fondo oscuro para el nodo
                         fontcolor='white', fontname='Helvetica-Bold', 
                         penwidth='3', width='1.2')

            # 2. Crear Aristas con valores en Blanco o Rojo
            for origin in selected_skus_graph:
                for destiny in selected_skus_graph:
                    prob = transition_matrix.loc[origin, destiny]
                    
                    if prob > 0.01:
                        edge_color = color_map.get(origin)
                        w = str(max(1, prob * 10)) 
                        
                        # --- LÓGICA DE COLOR DE NÚMEROS ---
                        # Rojo si la probabilidad es > 0.5 (Alerta de fuga/cambio fuerte), Blanco de lo contrario
                        text_color = "red" if prob > 0.5 else "white"
                        
                        if origin == destiny:
                            # Bucle de lealtad
                            dot.edge(origin, destiny, label=f" {prob:.2f}", 
                                     color=edge_color, 
                                     fontcolor=text_color, # Aplicamos el color aquí
                                     style='dashed', penwidth='2')
                        else:
                            # Flecha de cambio
                            dot.edge(origin, destiny, label=f" {prob:.2f}", 
                                     penwidth=w, color=edge_color, 
                                     fontcolor=text_color) # Aplicamos el color aquí

            st.graphviz_chart(dot, use_container_width=True)
            
            
            st.info(f"Leyenda: Los números en **rojo** indican transiciones dominantes (>50%), mientras que en **blanco** indican flujos regulares.")

    # --- NUEVAS PESTAÑAS SOLICITADAS ---

    with tab_inv:
        st.subheader("📦 Previsión de Demanda de Herramientas")
        st.markdown("Identificación de productos con mayor probabilidad de ser el 'Siguiente Paso' en la compra.")
        
        # Calcular demanda futura teórica (Estado Estacionario simple)
        # Probabilidad de que cualquier SKU sea el destino final
        future_demand = transition_matrix.mean(axis=0).sort_values(ascending=False)
        
        c_inv1, c_inv2 = st.columns([2, 1])
        with c_inv1:
            st.write("**Top SKUs con mayor atracción de demanda futura:**")
            st.bar_chart(future_demand.head(10))
        
        with c_inv2:
            st.info("### 💡 Insight de Inventario")
            top_attr = future_demand.index[0]
            st.write(f"""
            El SKU **{top_attr}** actúa como un 'Sumidero de Demanda'. 
            Muchos clientes que compran otros productos terminan migrando hacia este. 
            
            **Decisión:** Aumentar el stock de seguridad de **{top_attr}** en un **15%**, ya que la probabilidad de flujo entrante es alta.
            """)

    with tab_mkt:
        st.subheader("🎯 Estrategia de Marketing y Cross-Selling")
        
        # Quitar la diagonal para ver sustitución
        mask = np.eye(transition_matrix.shape[0], dtype=bool)
        substitution = transition_matrix.where(~mask).stack().sort_values(ascending=False)
        
        st.write("**Pares de productos con alta tendencia de sustitución/complemento:**")
        sub_df = substitution.head(5).reset_index()
        sub_df.columns = ['Origen', 'Sustituido por', 'Probabilidad']
        st.table(sub_df)

        st.warning(f"""
        **💡 Insight de Marketing:** Se detectó una fuerte migración de **{sub_df.iloc[0]['Origen']}** hacia **{sub_df.iloc[0]['Sustituido por']}**.
        
        **Decisión:** Diseñar un "Bundle" o promoción cruzada. Al cliente que compre el primer SKU, ofrecerle 
        un descuento inmediato en el segundo para acelerar una transición que ya es natural.
        """)
        

    with tab_plan:
        st.subheader("⚙️ Planificación Operativa y Rotación")
        
        # Calcular tasa de fuga (1 - Retención)
        retention = pd.Series(np.diag(transition_matrix), index=transition_matrix.index)
        churn_risk = (1 - retention).sort_values(ascending=False)
        
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            st.write("**SKUs con mayor rotación (Fuga hacia otros modelos):**")
            st.dataframe(churn_risk.to_frame('Probabilidad de Cambio').head(5))
            
        with c_p2:
            st.error("### 💡 Insight Operativo")
            high_rot = churn_risk.index[0]
            st.write(f"""
            El SKU **{high_rot}** tiene la mayor tasa de abandono (**{churn_risk[high_rot]*100:.1f}%**). 
            
            **Decisión:** Revisar la disponibilidad de las herramientas hacia las que el cliente huye. 
            Ajustar los turnos de logística de despacho para priorizar los productos de 'destino' de esta fuga.
            """)

    # --- 3. INSIGHTS GENERALES (MANTENIDOS) ---
    st.divider()
    st.subheader("💡 Insights Generales de Fidelidad")
    col_i1, col_i2 = st.columns(2)
    loyalty = pd.Series(np.diag(transition_matrix), index=transition_matrix.index)
    best_loyal = loyalty.idxmax()
    
    with col_i1:
        st.success(f"**Líder de Lealtad:** SKU **{best_loyal}**")
        st.write(f"Tasa de retención: **{loyalty[best_loyal]*100:.1f}%**.")
        
    with col_i2:
        mask = np.eye(transition_matrix.shape[0], dtype=bool)
        switches = transition_matrix.where(~mask).stack()
        if not switches.empty:
            top_switch = switches.idxmax()
            st.warning(f"**Mayor Riesgo de Fuga:** {top_switch[0]} → {top_switch[1]}")
            st.write(f"Probabilidad de cambio: **{switches[top_switch]*100:.1f}%**.")

# =======================================================
# Simulación Monte Carlo - 
# =======================================================
elif section == "Monte Carlo":
    st.header("🎲 Simulación Monte Carlo - Análisis de Riesgo")
    
    # Creamos las pestañas
    tab_config, tab_results = st.tabs(["⚙️ Configuración y Ejecución", "📊 Resultados por SKU"])

    with tab_config:
        st.markdown("Configure los parámetros para generar los escenarios de incertidumbre.")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            n_simulations = st.slider("Número de Simulaciones", 100, 2000, 500)
        with col_s2:
            horizon_mc = st.slider("Horizonte (semanas)", 4, 52, 12)

        if st.button("🚀 Iniciar Simulación"):
            sim_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Calculando escenarios..."):
                # A. Proyectar exógenas base
                _, futr_base_master = future_exog_to_historic(
                    df = train_df, freq = 'W-MON', features = exog_cols, h = horizon_mc
                )
                nf_mc = get_nf_dynamic(horizon_mc)

                # B. Bucle de Simulación
                for i in range(n_simulations):
                    current_prog = (i + 1) / n_simulations
                    progress_bar.progress(current_prog)
                    status_text.text(f"Simulando escenario {i+1} de {n_simulations}...")

                    futr_iteration = futr_base_master.copy()
                    for col in exog_cols:
                        noise = np.random.normal(1.0, 0.15, size=len(futr_iteration))
                        futr_iteration[col] = futr_iteration[col] * noise

                    forecast_mc = nf_mc.predict(futr_df=futr_iteration, verbose=False)
                    sim_results.append(forecast_mc.groupby('unique_id')['NHITS'].mean())

                # C. Guardar resultados en session_state para persistencia entre pestañas
                st.session_state['sim_df'] = pd.DataFrame(sim_results)
                st.session_state['horizon_mc'] = horizon_mc
                
                progress_bar.empty()
                status_text.success("✅ Simulación completada. Diríjase a la pestaña de Resultados.")

        # Botón de descarga (si existen resultados)
        if 'sim_df' in st.session_state:
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                st.session_state['sim_df'].to_excel(writer, sheet_name='Simulaciones')
                st.session_state['sim_df'].describe(percentiles=[0.05, 0.5, 0.95]).T.to_excel(writer, sheet_name='Resumen')
            
            st.download_button(
                label="📥 Descargar todos los escenarios (Excel)",
                data=buffer.getvalue(),
                file_name=f"montecarlo_full_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    with tab_results:
        if 'sim_df' in st.session_state:
            sim_df = st.session_state['sim_df']
            
            st.subheader("Análisis Individual por Producto")
            selected_sku = st.selectbox("Seleccione el SKU para ver el detalle:", options=sim_df.columns)

            # --- VISUALIZACIÓN DEL SKU SELECCIONADO ---
            col_graph, col_stats = st.columns([2, 1])
            
            p5 = sim_df[selected_sku].quantile(0.05)
            p50 = sim_df[selected_sku].quantile(0.5)
            p95 = sim_df[selected_sku].quantile(0.95)

            with col_graph:
                fig = px.histogram(
                    sim_df, x=selected_sku, nbins=30,
                    title=f"Distribución de Probabilidad: {selected_sku}",
                    color_discrete_sequence=['#1f77b4'], opacity=0.75
                )
                fig.add_vline(x=p5, line_dash="dash", line_color="#ef4444", annotation_text="P5")
                fig.add_vline(x=p50, line_dash="solid", line_color="#10b981", annotation_text="P50")
                fig.add_vline(x=p95, line_dash="dash", line_color="#3b82f6", annotation_text="P95")
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            with col_stats:
                st.write("### Métricas Clave")
                st.metric("Demanda Esperada (P50)", f"{p50:.0f} u")
                st.metric("Escenario Pesimista (P5)", f"{p5:.0f} u", delta=f"{((p5/p50)-1)*100:.1f}%", delta_color="inverse")
                st.metric("Escenario Optimista (P95)", f"{p95:.0f} u", delta=f"{((p95/p50)-1)*100:.1f}%")
                
                st.info(f"""
                **Insight:** Hay un 90% de probabilidad de que la demanda para **{selected_sku}** se mantenga entre **{p5:.0f}** y **{p95:.0f}** unidades.
                """)
        else:
            st.warning("⚠️ Primero debe ejecutar la simulación en la pestaña de 'Configuración y Ejecución'.")


# =======================================================
# Análisis de la Cesta del Mercado (Market Basket Analysis)
# =======================================================
elif section == "Análisis de la Cesta del Mercado":
    st.header("🛒 Market Basket Analysis (MBA)")
    
    st.markdown("""
    Identifique patrones de co-compra para optimizar el **layout de su tienda**, crear **kits de productos** y mejorar el **cross-selling**.
    """)

    # --- 1. GENERACIÓN DE TRANSACCIONES (SIMULACIÓN OPTIMIZADA) ---
    if 'basket_data' not in st.session_state:
        with st.spinner("Procesando historial de transacciones..."):
            transactions = []
            # Simulamos cestas basadas en la demanda histórica de los SKUs
            for week in df['ds'].unique():
                week_data = df[df['ds'] == week]
                probs = week_data['demand_units'].values
                probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(probs)) / len(probs)
                
                n_trans = np.random.randint(200, 400) # Cestas por semana
                for _ in range(n_trans):
                    n_items = np.random.choice([1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1])
                    basket = np.random.choice(skus, size=n_items, replace=False, p=probs)
                    transactions.append(basket.tolist())

            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            st.session_state['basket_data'] = pd.DataFrame(te_ary, columns=te.columns_)

    # --- 2. INTERFAZ POR PESTAÑAS ---
    tab_apriori, tab_rules, tab_network = st.tabs(["🔍 Configuración y Frecuencia", "📜 Reglas de Oro", "🕸️ Mapa de Asociación"])

    with tab_apriori:
        st.subheader("Configuración del Algoritmo Apriori")
        col_c1, col_c2 = st.columns([1, 2])
        
        with col_c1:
            min_support = st.slider("Soporte Mínimo (Frecuencia)", 0.01, 0.20, 0.05, 
                                    help="Porcentaje mínimo de transacciones en las que aparece el producto.")
            
            frequent_itemsets = apriori(st.session_state['basket_data'], min_support=min_support, use_colnames=True)
            frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
            
            st.metric("Itemsets Encontrados", len(frequent_itemsets))

        with col_c2:
            st.write("**Top Combinaciones más Comunes**")
            # Limpiamos los frozensets para visualización
            display_fi = frequent_itemsets.copy()
            display_fi['itemsets'] = display_fi['itemsets'].apply(lambda x: ', '.join(list(x)))
            st.dataframe(display_fi.sort_values('support', ascending=False).head(10), use_container_width=True)

    with tab_rules:
        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
            rules = rules.sort_values('lift', ascending=False)

            st.subheader("Reglas de Asociación (Cross-Selling)")
            
            # Limpieza para el usuario
            rules_view = rules.copy()
            rules_view['Origen'] = rules_view['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules_view['Destino'] = rules_view['consequents'].apply(lambda x: ', '.join(list(x)))
            
            st.dataframe(
                rules_view[['Origen', 'Destino', 'support', 'confidence', 'lift']].head(20)
                .style.background_gradient(subset=['lift'], cmap='OrRd'),
                use_container_width=True
            )

            

            st.info("""
            **Cómo interpretar:**
            * **Support:** Qué tan común es la regla.
            * **Confidence:** Probabilidad de comprar el 'Destino' si ya compró el 'Origen'.
            * **Lift:** Si es mayor a 1, los productos están asociados positivamente (no es coincidencia).
            """)
        else:
            st.warning("No se encontraron reglas. Intente bajar el Soporte Mínimo.")

    with tab_network:
        st.subheader("Mapa Visual de Conexiones (Grafo de Afinidad)")
        if len(frequent_itemsets) > 0 and 'rules' in locals():
            # Filtro para el grafo (solo las conexiones más fuertes para no saturar)
            min_lift_graph = st.slider("Filtrar por Lift mínimo (Fuerza de unión)", 1.0, 2.5, 1.2)
            rules_strong = rules[rules['lift'] > min_lift_graph]

            if not rules_strong.empty:
                G = nx.DiGraph()
                for _, row in rules_strong.iterrows():
                    ant = ', '.join(list(row['antecedents']))
                    cons = ', '.join(list(row['consequents']))
                    G.add_edge(ant, cons, weight=row['lift'])

                plt.figure(figsize=(12, 8))
                pos = nx.spring_layout(G, k=1.2, iterations=60)
                
                # Nodos con estilo coral
                nx.draw_networkx_nodes(G, pos, node_color='#ff8a80', node_size=2500, alpha=0.9)
                nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
                
                # Aristas (flechas) con grosor según Lift
                edges = G.edges(data=True)
                nx.draw_networkx_edges(G, pos, arrowsize=20, edge_color='#90a4ae', 
                                       width=[d['weight']*2 for (u,v,d) in edges], 
                                       connectionstyle='arc3,rad=0.1')
                
                plt.axis('off')
                st.pyplot(plt)
                st.caption("El grosor de la flecha indica la fuerza de la asociación (Lift).")
            else:
                st.info("No hay asociaciones lo suficientemente fuertes con el Lift seleccionado.")
        else:
            st.error("Debe generar itemsets frecuentes primero.")

    # --- 3. INSIGHTS ESTRATÉGICOS ---
    st.divider()
    st.subheader("💡 Recomendaciones de Negocio")
    
    col_ins1, col_ins2 = st.columns(2)
    with col_ins1:
        st.markdown("### 🏬 Tienda Física")
        st.write("""
        * **Layout Adyacente:** Coloque los productos con mayor 'Lift' en pasillos contiguos.
        * **Kits de Proyecto:** Agrupe itemsets de longitud 3 o más en un solo paquete con un ligero descuento (ej. Kit 'Pintado de Pared').
        """)
        
    with col_ins2:
        st.markdown("### 💻 E-Commerce")
        st.write("""
        * **Next-Best-Offer:** En el carrito, si el cliente tiene el 'Origen', dispare automáticamente la recomendación del 'Destino'.
        * **Bundling dinámico:** Muestre el mensaje 'Comprados juntos habitualmente' con las combinaciones de mayor 'Confidence'.
        """)


st.sidebar.markdown("---")
st.sidebar.markdown("**Dashboard Profesional - Naren Castellón** | Especialización en Forecasting & IA 2026")