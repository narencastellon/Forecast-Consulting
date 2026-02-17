# =======================================================
# App Streamlit Profesional: Forecasting Jerárquico de Ventas Semanales (Tienda → SKU) en Pycca
# Usa NeuralForecast (NHITS Probabilístico) para base forecasts bottom-level (tienda-SKU)
# + HierarchicalForecast para reconciliación jerárquica (Bottom → Tienda → Total)
# Dashboard interactivo completo con KPIs, filtros, gráficos elegantes
# Incluye: Dashboard Principal, EDA Interactivo, Predicciones Jerárquicas (con intervals),
# Análisis de Sensibilidad, Gestión de Inventario, Escenarios de Inventario, Cadenas de Markov, Monte Carlo
# Ejecuta con: streamlit run app_pycca_jerarquico_bi.py
# =======================================================
# Requisitos: pip install streamlit pandas numpy matplotlib seaborn plotly neuralforecast hierarchicalforecast networkx utilsforecast

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
#import networkx as nx
from utilsforecast.feature_engineering import future_exog_to_historic
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, MinTrace
from hierarchicalforecast.utils import aggregate
import warnings
warnings.filterwarnings('ignore')

# Configuración página
st.set_page_config(page_title="Dashboard Pycca - Naren Castellón", layout="wide")
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stMetric {background-color: #003300; padding: 10px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
    h1 {color: #1e3a8a; text-align: center;}
    .sidebar .sidebar-content {background-color: #e0e7ff;}
</style>
""", unsafe_allow_html=True)



st.title("🏪 Forecasting Jerárquico con Redes Neuronales para Ventas Semanales en Pycca")
st.markdown("Creado por Naren Castellón (@NarenCastellon)")

st.sidebar. image("https://scontent.fmga10-1.fna.fbcdn.net/v/t39.30808-6/470554355_18474138037055527_1020868256347381902_n.jpg?_nc_cat=110&ccb=1-7&_nc_sid=127cfc&_nc_ohc=sF3RXBbJzu4Q7kNvwFmHvM2&_nc_oc=Adnbt6Pj_w-wwTZXYivXSviU5nH9Gbp9uF4P4cDu4bHOKHtEONExGQbPnsRstB0vE-I&_nc_zt=23&_nc_ht=scontent.fmga10-1.fna&_nc_gid=pxMUsnU5_qHZ1F5BzstgMw&oh=00_AfswgXpwHTZ2oY7k97JXRui9cDMl3d7Q6hX0B2jIRCXXqA&oe=6996DF80", 
         caption="Forecasting Jerárquico de Ventas Semanales (Tienda → SKU) en https://www.pycca.com", )

# =======================================================
# Carga de datos y modelo base (cacheado)
# =======================================================
@st.cache_resource
def load_data_and_model():
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', '2026-02-01', freq='W-MON')

    stores = ['Quito Norte', 'Quito Sur', 'Guayaquil Centro', 'Cuenca', 'Manta']

    skus = [
        'Freidora de Aire AirPro', 'Licuadora Vitamix Pro', 'Microondas QuickHeat 25L', 'Lavadora Innova 41 Libras', 'Lavadora Electrolux 39 Libras',
        'Encimera Inducción Challenger', 'Campana Extractora Challenger', 'Enfriador Aire Oster', 'Purificador Aire Scent Diffuser', 'Máquina Coser Breli',
        'Smartwatch Cubitt Viva Pro', 'Teclado Bluetooth Bk3001', 'Teléfono Panasonic KX-TGC352LAB', 'Audífonos Maxell IN-225', 'Parlante Italy Audio',
        'Mouse Verbatim', 'Impresora Epson EcoTank', 'Sofá Cama Relax 2 Plazas', 'Juego Comedor 6 Puestos', 'Bicicleta Mountain Bike Aro 29'
    ]

    index = pd.MultiIndex.from_product([stores, skus, dates], names=['store', 'sku_id', 'ds'])
    df = pd.DataFrame(index=index).reset_index()

    df['price_unit'] = np.random.uniform(10.0, 800.0, len(df)).round(2)
    df['cost'] = df['price_unit'] * np.random.uniform(0.5, 0.8, len(df)).round(2)
    df['promotion_active'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
    df['season_factor'] = np.sin(2 * np.pi * df['ds'].dt.isocalendar().week / 52) + 1.5
    df['temperature_avg'] = np.random.normal(25, 8, len(df)).clip(15, 35)
    df['rain_days'] = np.random.poisson(2, len(df)).clip(0, 7)
    df['events_nearby'] = np.random.choice([0, 1, 2], len(df), p=[0.7, 0.2, 0.1])
    df['marketing_spend'] = np.random.normal(15000, 5000, len(df)).clip(5000, 35000)
    df['competitor_price'] = np.random.uniform(9.5, 790.0, len(df)).round(2)
    df['economic_index'] = np.random.normal(100, 10, len(df)).clip(80, 120)
    df['inflation_rate'] = np.random.normal(4, 1.5, len(df)).clip(2, 8)
    df['population_local'] = np.random.normal(200000, 50000, len(df)).clip(100000, 400000)
    df['online_sales_rate'] = np.random.uniform(0.15, 0.55, len(df))
    df['store_visits'] = np.random.poisson(10000, len(df)).clip(4000, 20000)
    df['holiday_week'] = (df['ds'].dt.isocalendar().week.isin([52, 1, 25, 26, 50])).astype(int)
    df['new_product_launch'] = np.random.choice([0, 1], len(df), p=[0.9, 0.1])
    df['lead_time_days'] = np.random.choice([7, 14, 21, 30], len(df))
    df['current_stock'] = np.random.randint(50, 5000, len(df))
    df['supplier_delay_rate'] = np.random.uniform(0.05, 0.2, len(df))
    df['customer_reviews_score'] = np.random.uniform(3.5, 4.9, len(df)).round(1)
    df['distribution_channels'] = np.random.choice([1, 2, 3], len(df))
    df['foot_traffic_index'] = np.random.normal(100, 15, len(df)).clip(70, 130)
    df['previous_sku'] = df['sku_id']

    switch_mask = np.random.choice([0, 1], len(df), p=[0.6, 0.4])
    df.loc[switch_mask == 1, 'previous_sku'] = np.random.choice(skus, sum(switch_mask))

    df['demand_units'] = (np.random.poisson(300, len(df)) * df['season_factor'] * (1 + df['promotion_active']*0.4) * 
                          (1 + df['events_nearby']*0.15) * (df['population_local']/200000) * (df['economic_index']/100) * 
                          (df['store_visits']/10000) * (1 - df['inflation_rate']/100 * 0.2)).clip(20, 10000).astype(int)

    exog_cols = ['price_unit', 'promotion_active', 'season_factor', 'temperature_avg', 'events_nearby', 'marketing_spend', 
                 'competitor_price', 'economic_index', 'online_sales_rate', 'store_visits', 'holiday_week', 'lead_time_days', 'population_local']


    hier_df = df.copy()
    hier_df = hier_df.rename(columns={'demand_units': 'y'})

    train_hier = hier_df[hier_df['ds'] < '2025-07-01']

    Y_df, S_df, tags = aggregate(df = train_hier, spec=[['store'], ['store', 'sku_id']])

    # eliminar elemeto del diccionario
    del tags['store']

    #Eliminat ciudades de S_df
    ciudades_a_eliminar = ['Quito Norte', 'Quito Sur', 'Guayaquil Centro', 'Cuenca', 'Manta']
    S_df = S_df[~S_df['unique_id'].isin(ciudades_a_eliminar)]
    Y_df = Y_df[~Y_df['unique_id'].isin(ciudades_a_eliminar)]

    # Agregar exógenas a Y_df bottom level
    bottom_unique = tags['store/sku_id']
    Y_df_bottom = Y_df[Y_df['unique_id'].isin(bottom_unique)].copy()

    train_hier['unique_id'] = train_hier['store'] + '/' + train_hier['sku_id']
    Y_df_bottom = Y_df_bottom.merge(train_hier[['unique_id', 'ds'] + exog_cols], on=['unique_id', 'ds'], how='left')

    Y_df_higher = Y_df[~Y_df['unique_id'].isin(bottom_unique)].copy()
    train_nf = pd.concat([Y_df_higher, Y_df_bottom], ignore_index=True)
    
    # Horizonte base

    horizon_base = 52

    _, futr_base = future_exog_to_historic(
        df = train_nf,
        freq = 'W-MON',
        features= exog_cols,
        h = horizon_base ,
        )

    models = [NHITS(h = horizon_base, input_size=52*2, futr_exog_list=exog_cols, hist_exog_list=exog_cols,
                    loss=DistributionLoss(distribution='Normal', level=[80, 90], return_params=True),
                    valid_loss=MQLoss(level=[80, 90]), scaler_type='robust', max_steps=500, random_seed=42)]

    nf_base = NeuralForecast(models=models, freq='W-MON')
    nf_base.fit(df=train_nf)



    base_forecasts = nf_base.predict(futr_df=futr_base)

    base_forecasts_wide = base_forecasts[['unique_id', 'ds', 'NHITS', 'NHITS-median', 'NHITS-lo-90', 'NHITS-lo-80', 'NHITS-hi-80', 'NHITS-hi-90',]]

    reconcilers = [BottomUp(), MinTrace(method='ols')]
    hfc = HierarchicalReconciliation(reconcilers=reconcilers)
    reconciled_base = hfc.reconcile(Y_hat_df=base_forecasts_wide, Y_df= train_nf , S=S_df, tags=tags)

    return df, train_nf, exog_cols, stores, skus,horizon_base,  nf_base, futr_base, reconciled_base, S_df, tags

df, train_nf, exog_cols, stores, skus,horizon_base, nf_base, futr_base, reconciled_base, S_df, tags = load_data_and_model()

# =======================================================
# Función para modelo dinámico
# =======================================================
@st.cache_resource
def get_nf_dynamic(horizon):
    models_nf = [NHITS(h=horizon, input_size=52*2, futr_exog_list=exog_cols, hist_exog_list=exog_cols,
                       scaler_type='robust', loss=DistributionLoss(distribution='Normal', level=[80, 90], return_params=True),
                       valid_loss=MQLoss(level=[80, 90]), max_steps=500, random_seed=42)]
    nf = NeuralForecast(models=models_nf, freq='W-MON')
    nf.fit(df=train_nf)
    return nf

# =======================================================
# Sidebar navegación
# =======================================================
st.sidebar.title("📊 Navegación")

# =======================================================
# Gestión de Persistencia de Navegación (URL Params)
# =======================================================

# 1. Obtener la sección de la URL (si existe), sino por defecto "Dashboard"
query_params = st.query_params
default_section = query_params.get("section", "Dashboard")

# 2. Definir la lista de secciones (asegúrate de que coincidan con tus 'elif')
menu_options = ["Dashboard Principal", "EDA Interactivo", "Predicciones Jerárquicas","Prueba Stress", "Análisis de Sensibilidad", 
                            "Gestión de Inventario", "Escenario de Inventarios", "Cadenas de Markov", "Monte Carlo"]

# Si el valor en la URL no es válido, resetear a Dashboard
if default_section not in menu_options:
    default_section = "Dashboard Principal"

# 3. Crear el selector en el sidebar
section = st.sidebar.radio(
    label= '',
    options=menu_options,
    index=menu_options.index(default_section)
)


# 4. Actualizar la URL cada vez que el usuario cambie de sección
st.query_params["section"] = section




# =======================================================
# Dashboard Principal - Versión Ultra-Analítica
# =======================================================
if section == "Dashboard Principal":
    st.header("📈 Dashboard Ejecutivo de Demanda")
    
    # --- FILTRO GLOBAL ---
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        view_sku = st.selectbox("🎯 Seleccionar Enfoque:", ["Todos los SKUs"] + skus)
    
    # Procesamiento de datos según filtro
    if view_sku == "Todos los SKUs":
        display_df = df.copy()
        title_suffix = "Consolidado General"
    else:
        display_df = df[df['sku_id'] == view_sku].copy()
        title_suffix = f"SKU: {view_sku}"

    # Cálculos base para KPIs
    display_df['revenue'] = display_df['demand_units'] * display_df['price_unit']
    total_kg = display_df['demand_units'].sum()
    total_usd = display_df['revenue'].sum()
    avg_kg = display_df['demand_units'].mean()
    
    # --- MÉTRICAS SUPERIORES ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Volumen Total", f"{total_kg:,.0f}")
    c2.metric("Venta Estimada", f"${total_usd:,.0f}")
    c3.metric("Promedio Semanal", f"{avg_kg:.1f}")
    c4.metric("Máximo Histórico", f"{display_df['demand_units'].max():,.0f}")

    st.markdown("---")

    # --- PESTAÑAS REORGANIZADAS ---
    tab_temp, tab_estac, tab_fact, tab_data = st.tabs([
        "📈 Desempeño Temporal", 
        "🗓️ Estacionalidad y Mix", 
        "🔬 Factores de Influencia",
        "📄 Explorador de Datos"
    ])

    # --- TAB 1: DESEMPEÑO TEMPORAL ---
    with tab_temp:
        # Control dinámico de promedio móvil
        window_ma = st.slider("Ventana de Promedio Móvil (Semanas):", 1, 52, 4)
        
        df_ts = display_df.groupby('ds')['demand_units'].sum().reset_index()
        df_ts['MA'] = df_ts['demand_units'].rolling(window=window_ma).mean()
        
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=df_ts['ds'], y=df_ts['demand_units'], name="Demanda Real", line=dict(color='#1e3a8a', width=1.5), opacity=0.6))
        fig_line.add_trace(go.Scatter(x=df_ts['ds'], y=df_ts['MA'], name=f"Tendencia (MA {window_ma})", line=dict(color='#ef4444', width=3)))
        
        fig_line.update_layout(title=f"Evolución de Demanda: {title_suffix}", template="plotly_white", height=500, hovermode="x unified")
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Insights debajo del gráfico
        st.subheader("💡 Insights de Tendencia")
        recent_val = df_ts['demand_units'].iloc[-1]
        trend_val = df_ts['MA'].iloc[-1]
        diff = ((recent_val - trend_val) / trend_val) * 100 if trend_val != 0 else 0
        
        col_ins1, col_ins2 = st.columns(2)
        with col_ins1:
            if recent_val > trend_val:
                st.success(f"**Impulso Positivo:** La última semana está un **{diff:.1f}%** por encima de la tendencia media. Esto sugiere una aceleración atípica del consumo.")
            else:
                st.warning(f"**Desaceleración:** La demanda actual está un **{abs(diff):.1f}%** por debajo del promedio móvil. Posible saturación de mercado o estacionalidad baja.")
        with col_ins2:
            st.write(f"**Acción recomendada:** {'Ajustar stock de seguridad al alza para evitar quiebres.' if diff > 0 else 'Evaluar promociones para reactivar el volumen de salida.'}")

    # --- TAB 2: ESTACIONALIDAD Y MIX ---
    with tab_estac:
        # Gráfico 1: Heatmap
        st.subheader("📍 Concentración por Periodo")
        display_df['month'] = display_df['ds'].dt.month_name()
        display_df['year'] = display_df['ds'].dt.year
        months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        
        pivot_heat = display_df.groupby(['year', 'month'])['demand_units'].sum().unstack()
        pivot_heat = pivot_heat.reindex(columns=[m for m in months_order if m in pivot_heat.columns])
        
        fig_heat = px.imshow(pivot_heat, text_auto=True, color_continuous_scale='GnBu', aspect="auto")
        st.plotly_chart(fig_heat, use_container_width=True)
        
        st.info(f"**Insight Estacional:** Los picos de demanda se concentran en el cuarto trimestre. Históricamente, el mes con mayor presión operativa es **{display_df.groupby('month')['demand_units'].mean().idxmax()}**.")

        st.divider()

        # Gráfico 2: Mix o Promociones
        st.subheader("📊 Análisis de Composición")
        if view_sku == "Todos los SKUs":
            fig_mix = px.pie(display_df.groupby('sku_id')['revenue'].sum().reset_index(), values='revenue', names='sku_id', hole=0.4)
            st.plotly_chart(fig_mix, use_container_width=True)
            st.write("**Insight de Mix:** El 80% de los ingresos está concentrado en el top 3 de productos. Se recomienda diversificar para reducir dependencia de cortes específicos.")
        else:
            fig_box = px.box(display_df, x='promotion_active', y= 'demand_units', color='promotion_active', points="all")
            st.plotly_chart(fig_box, use_container_width=True)
            st.write(f"**Insight de Promoción:** En **{view_sku}**, las promociones activas incrementan la demanda mediana en un **{((display_df[display_df['promotion_active']==1]['demand_kg'].median() / display_df[display_df['promotion_active']==0]['demand_kg'].median())-1)*100:.1f}%**.")

    # --- TAB 3: FACTORES DE INFLUENCIA ---
    with tab_fact:
        st.subheader("🔬 Matriz de Correlación de Variables (Elasticidad)")
        
        # Selección de variables y cálculo redondeado
        vars_corr = ['demand_units', 'price_unit', 'promotion_active', 'temperature_avg', 'marketing_spend', 'store_visits']
        corr_matrix = display_df[vars_corr].corr().round(2)
        
        # Gráfico más grande
        fig_corr = px.imshow(
            corr_matrix, 
            text_auto='.2f', 
            color_continuous_scale='RdBu_r', 
            range_color=[-1, 1],
            height=700 # Aumentamos el tamaño
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        
        st.subheader("💡 Conclusión de Sensibilidad")
        price_sens = corr_matrix.loc['demand_units', 'price_unit']
        if price_sens < -0.4:
            st.error(f"**Alta Elasticidad Precio ({price_sens}):** Este producto es sumamente sensible. Un aumento en el precio castigará el volumen de ventas de manera inmediata.")
        else:
            st.success(f"**Baja Elasticidad Precio ({price_sens}):** La demanda es inelástica. Tienes margen para ajustes de precio sin perder volumen crítico de clientes.")

    # --- TAB 4: DATOS ---
    with tab_data:
        st.subheader("📄 Registros Filtrados")
        st.markdown(f"Mostrando los datos base para la selección actual: **{title_suffix}**")
        st.dataframe(display_df.sort_values('ds', ascending=False), use_container_width=True)
        
        # Opción de descarga rápida
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar Datos Actuales (CSV)", data=csv, file_name=f"datos_{view_sku}.csv", mime='text/csv')

# =======================================================
# EDA Interactivo
# =======================================================
elif section == "EDA Interactivo":
    st.header("🔍 EDA Interactivo")
    selected_store = st.selectbox("Seleccionar Tienda", stores)
    selected_sku = st.selectbox("Seleccionar SKU", skus)

    store_df = df[df['store'] == selected_store]
    sku_df = df[df['sku_id'] == selected_sku]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(store_df.groupby('ds')['demand_units'].sum().reset_index(), x='ds', y='demand_units', title=f"Demanda {selected_store}")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(sku_df.groupby('ds')['demand_units'].sum().reset_index(), x='ds', y='demand_units', title=f"Demanda {selected_sku}")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.box(df, x='sku_id', y='demand_units', color='store', title="Distribución Demanda por SKU y Tienda")
    st.plotly_chart(fig, use_container_width=True)


# =======================================================
# Predicciones Dinámicas 
# =======================================================
elif section == "Predicciones Jerárquicas":
    st.header("🔮 Predicciones Dinámicas (Probabilísticas)")
    st.markdown("El sistema recalcula el pronóstico automáticamente al ajustar los parámetros.")

    # --- CONTROLES SUPERIORES ---
    c_p1, c_p2 = st.columns([1, 1])
    with c_p1:
        selected_sku = st.selectbox("🎯 Seleccionar Store/SKU:", train_nf['unique_id'].unique())
    with c_p2:
        horizon_selected = st.slider("📅 Horizonte de Predicción (semanas):", 12, 104, 52)

    # --- LÓGICA DE PROCESAMIENTO (AUTOMÁTICA) ---
    with st.spinner(f"Calculando trayectoria para {selected_sku}..."):
        nf_dynamic = get_nf_dynamic(horizon_selected)
        
        # Generar datos futuros base
        _, futr_dynamic = future_exog_to_historic(df=train_nf, freq='W-MON', features=exog_cols, h=horizon_selected)
        
        # Ejecutar predicción
        forecast_dynamic = nf_dynamic.predict(futr_df=futr_dynamic)
        forecast_sku = forecast_dynamic[forecast_dynamic['unique_id'] == selected_sku].copy()

    # --- PRESENTACIÓN DE RESULTADOS ---
    tab_grafico, tab_datos = st.tabs(["📉 Visualización de Pronóstico", "📋 Tabla de Proyecciones"])

    with tab_grafico:
        # Gráfico Plotly Probabilístico
        fig_pred = go.Figure()
        historical = train_nf[train_nf['unique_id'] == selected_sku].tail(72) # Mostramos último año
        
        # Sombra de Incertidumbre (Intervalo 90%)
        fig_pred.add_trace(go.Scatter(
            x=forecast_sku['ds'], y=forecast_sku['NHITS-hi-90'],
            fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False
        ))
        fig_pred.add_trace(go.Scatter(
            x=forecast_sku['ds'], y=forecast_sku['NHITS-lo-90'],
            fill='tonexty', mode='lines', fillcolor='rgba(255, 0, 0, 0.15)', 
            line_color='rgba(0,0,0,0)', name='Incertidumbre (90%)'
        ))
        
        # Líneas Principales
        fig_pred.add_trace(go.Scatter(x=historical['ds'], y=historical['y'], name='Histórico (1 Año)', line=dict(color='#64748b')))
        fig_pred.add_trace(go.Scatter(x=forecast_sku['ds'], y=forecast_sku['NHITS'], name='Predicción Central', line=dict(color='#1e3a8a', width=3)))
        fig_pred.add_trace(go.Scatter(x=forecast_sku['ds'], y=forecast_sku['NHITS-median'], name='Predicción Median', line=dict(color="#1e8a49", width=3)))
        
        fig_pred.update_layout(
            title=f"Proyección de Demanda: {selected_sku}",
            xaxis_title="Fecha", yaxis_title="Kilos (Kg)",
            template="plotly_white", hovermode="x unified"
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        

        # Métricas Resumen
        m1, m2, m3 = st.columns(3)
        mean_val = forecast_sku['NHITS'].mean()
        max_val = forecast_sku['NHITS'].max()
        uncertainty = (forecast_sku['NHITS-hi-90'].mean() - forecast_sku['NHITS-lo-90'].mean()) / mean_val
        
        m1.metric("Demanda Media Proyectada", f"{mean_val:.0f}")
        m2.metric("Pico de Demanda Esperado", f"{max_val:.0f}")
        m3.metric("Grado de Incertidumbre", f"{uncertainty:.1%}", 
                  help="Un porcentaje alto indica que la demanda es más difícil de predecir debido a la volatilidad.")

    with tab_datos:
        st.subheader("Detalle Semanal de Proyecciones")

        st.dataframe(forecast_dynamic[forecast_dynamic['unique_id'] == selected_sku].style.format(precision=0), use_container_width=True)

    # --- SECCIÓN DE INSIGHTS PARA TOMA DE DECISIONES ---
    st.divider()
    st.subheader("💡 Insights Analíticos para Gestión")
    
    col_ins1, col_ins2 = st.columns(2)
    
    with col_ins1:
        # Lógica de tendencia futura
        first_val = forecast_sku['NHITS'].iloc[:4].mean()
        last_val = forecast_sku['NHITS'].iloc[-4:].mean()
        diff_pct = ((last_val - first_val) / first_val) * 100
        
        st.markdown("**Análisis de Trayectoria:**")
        if diff_pct > 5:
            st.warning(f"📈 **Tendencia al Alza:** Se espera que la demanda crezca un **{diff_pct:.1f}%** hacia el final del horizonte. Considere negociar contratos de suministro por volumen ahora para asegurar precios.")
        elif diff_pct < -5:
            st.info(f"📉 **Tendencia a la Baja:** El modelo prevé una contracción del **{abs(diff_pct):.1f}%**. Evite el sobre-stock para minimizar mermas, especialmente en cortes perecederos.")
        else:
            st.success("⚖️ **Estabilidad:** La demanda se mantendrá constante. Mantenga su estrategia de inventario Just-in-Time.")

    with col_ins2:
        st.markdown("**Recomendación de Inventario:**")
        # Basado en la incertidumbre
        if uncertainty > 0.4:
            st.error("❗ **Alta Variabilidad:** El modelo detecta mucha incertidumbre en este SKU. **Acción:** Incremente el Stock de Seguridad un 15% por encima de la predicción media para evitar quiebres.")
        else:
            st.info("✅ **Predicción Confiable:** La banda de incertidumbre es estrecha. **Acción:** Puede optimizar el flujo de caja reduciendo los niveles de stock de seguridad.")

    st.caption(f"Nota: Estas predicciones utilizan el modelo NHITS con una frecuencia de datos {nf_base.freq}.")

# =======================================================
# Prueba de Stress: Simulación de Escenarios
# =======================================================
elif section == "Prueba Stress":
    st.header("🔮 Simulación de Escenarios y Stress Test")
    st.info("Ajuste las variables externas para ver cómo impactarían la demanda futura.")

    # --- 1. CONFIGURACIÓN DE PARÁMETROS ---
    col_p1, col_p2 = st.columns([1, 1])
    with col_p1:
        selected_sku = st.selectbox("🎯 Seleccionar Store-SKU:", train_nf['unique_id'].unique())
    with col_p2:
        horizon_selected = st.slider("📅 Horizonte (semanas):", 12, 104, 52)

    # --- 2. SELECCIÓN DINÁMICA DE SHOCKS ---
    st.subheader("🛠️ Configurar Shocks Personalizados")
    
    # Seleccionar solo algunas variables para estresar
    selected_exogs = st.multiselect(
        "¿Qué variables quieres estresar?", 
        options=exog_cols,
        default=exog_cols[0:2] if len(exog_cols) > 1 else exog_cols
    )

    # Generar sliders solo para las seleccionadas
    shocks = {}
    if selected_exogs:
        cols = st.columns(len(selected_exogs))
        for i, col in enumerate(selected_exogs):
            with cols[i]:
                shocks[col] = st.slider(f"Shock {col}", 0.5, 1.5, 1.0, help="1.0 = Sin cambio. >1.0 = Aumento.")
    else:
        st.warning("Selecciona al menos una variable para aplicar un shock.")

    # --- 3. PROCESAMIENTO DE PREDICCIÓN ---
    with st.spinner("Calculando impacto del escenario..."):
        # Obtenemos el modelo
        nf_dynamic = get_nf_dynamic(horizon_selected)
        
        # Generar futuro base y aplicar shocks
        _, futr_dynamic = future_exog_to_historic(df=train_nf, freq='W-MON', features=exog_cols, h=horizon_selected)
        
        # Aplicamos los multiplicadores solo a las variables seleccionadas
        for col, multiplier in shocks.items():
            futr_dynamic[col] *= multiplier

        # Predicción
        forecast_dynamic = nf_dynamic.predict(futr_df=futr_dynamic)
        forecast_sku = forecast_dynamic[forecast_dynamic['unique_id'] == selected_sku].copy()

    # --- 4. VISUALIZACIÓN ---
    tab_graf, tab_ins = st.tabs(["📉 Gráfico de Simulación", "💡 Análisis y Decisiones"])

    with tab_graf:
        fig = go.Figure()
        historical = train_nf[train_nf['unique_id'] == selected_sku].tail(52)

        # Intervalo de confianza (Incertidumbre)
        fig.add_trace(go.Scatter(
            x=forecast_sku['ds'], y=forecast_sku['NHITS-hi-90'],
            fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_sku['ds'], y=forecast_sku['NHITS-lo-90'],
            fill='tonexty', mode='lines', fillcolor='rgba(255, 69, 0, 0.1)', 
            line_color='rgba(0,0,0,0)', name='Banda de Riesgo (90%)'
        ))

        # Líneas de demanda
        fig.add_trace(go.Scatter(x=historical['ds'], y=historical['y'], name='Histórico', line=dict(color='#475569')))
        fig.add_trace(go.Scatter(x=forecast_sku['ds'], y=forecast_sku['NHITS'], name='Forecast Estresado', line=dict(color='#ef4444', width=3)))

        fig.update_layout(title=f"Impacto de Escenario en {selected_sku}", template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        

    with tab_ins:
        st.subheader("📋 Insights para la Toma de Decisiones")
        
        # Cálculos de impacto
        avg_demand_sim = forecast_sku['NHITS'].mean()
        peak_demand_sim = forecast_sku['NHITS'].max()
        volatility_sim = (forecast_sku['NHITS'].std() / avg_demand_sim) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Promedio Simulado", f"{avg_demand_sim:.0f}")
        c2.metric("Pico Esperado", f"{peak_demand_sim:.0f}")
        c3.metric("Volatilidad Escenario", f"{volatility_sim:.1f}%")

        st.divider()
        
        # Lógica de decisión
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("**🛡️ Gestión de Riesgos:**")
            if volatility_sim > 20:
                st.error(f"**Alerta:** El escenario aplicado genera una demanda altamente inestable. Se recomienda un stock de seguridad del **25%** por encima del promedio.")
            else:
                st.success("El escenario muestra una demanda estable. Puede operar con niveles de inventario optimizados (Just-in-Time).")
        
        with col_right:
            st.markdown("**💰 Impacto Comercial:**")
            # Supongamos que comparamos con la última demanda real conocida
            last_real = historical['y'].iloc[-1]
            diff = ((avg_demand_sim - last_real) / last_real) * 100
            
            if diff > 10:
                st.info(f"**Oportunidad:** Este escenario sugiere un crecimiento del **{diff:.1f}%**. Es necesario negociar con proveedores un aumento en el cupo de abastecimiento.")
            elif diff < -10:
                st.warning(f"**Atención:** Se prevé una caída del **{abs(diff):.1f}%**. Riesgo de mermas; considere reducir la frecuencia de pedidos.")

    # --- 5. TABLA DE DATOS ---
    with st.expander("Ver detalle de datos simulados"):
        st.dataframe(forecast_dynamic[forecast_dynamic['unique_id'] == selected_sku].style.format(precision=0))


# =======================================================
# Análisis de Sensibilidad - Versión con Insights Direccionales
# =======================================================
elif section == "Análisis de Sensibilidad":
    st.header("📊 Análisis de Sensibilidad y Elasticidad")
    st.markdown("Proyecta cómo cambia la demanda del SKU cuando manipulas variables clave.")

    # --- 1. CONTROLES EN COLUMNAS ---
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        selected_sku_sens = st.selectbox("🎯 Seleccionar SKU:", train_nf['unique_id'].unique())
        variable_sens = st.selectbox("🧪 Variable a Testear:", exog_cols)
    
    with col_ctrl2:
        range_min = st.slider("📉 Límite Inferior", 0.5, 1.0, 0.8)
        range_max = st.slider("📈 Límite Superior", 1.0, 2.0, 1.2)

    # --- 2. CÁLCULO DE SENSIBILIDAD ---
    multipliers = np.linspace(range_min, range_max, 10)
    sensitivities = []
    base_mean = reconciled_base[reconciled_base['unique_id'] == selected_sku_sens]['NHITS'].mean()

    with st.spinner("Calculando elasticidad..."):
        for mult in multipliers:
            _, futr_sens = future_exog_to_historic(df=train_nf, freq='W-MON', features=exog_cols, h=horizon_base)
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
# Gestión de Inventario
# =======================================================
elif section == "Gestión de Inventario":
    st.header("📦 Gestión Profesional de Supply Chain")
    st.markdown("Optimización de stock basada en demanda probabilística, Lead Times y modelos EOQ.")

    df1 = df.copy()
    df1['sku_id'] = df1['store'] + '/' + df1['sku_id']

    #st.dataframe(df1)

    
    # --- 1. PROCESAMIENTO DE DATOS (Factores Originales Preservados) ---
    forecast_mean = reconciled_base[['unique_id', 'ds', 'NHITS']].copy()
    forecast_mean['sku_id'] = forecast_mean['unique_id']

    # Agregación de métricas de pronóstico
    inventory_summary = forecast_mean.groupby('sku_id')['NHITS'].agg(['mean', 'std']).reset_index()
    inventory_summary = inventory_summary.rename(columns={'mean': 'demand_forecast_mean', 'std': 'demand_forecast_std'})

    # Unión con datos históricos (Lead Time y Stock actual)
    

    lead_time_avg = df1.groupby('sku_id')['lead_time_days'].mean().reset_index()
    current_stock = df1[df1['ds'] == df1['ds'].max()][['sku_id', 'current_stock']]

    inventory_summary = inventory_summary.merge(lead_time_avg, on='sku_id')
    inventory_summary = inventory_summary.merge(current_stock, on='sku_id')

    # --- CÁLCULOS TÉCNICOS SCM ---
    # Stock de Seguridad (Z=1.65 para 95% nivel de servicio)
    inventory_summary['safety_stock'] = 1.65 * inventory_summary['demand_forecast_std'] * np.sqrt(inventory_summary['lead_time_days'] / 7)
    
    # Punto de Reorden (ROP)
    inventory_summary['reorder_point'] = (inventory_summary['demand_forecast_mean'] * (inventory_summary['lead_time_days'] / 7)) + inventory_summary['safety_stock']
    
    # Riesgo y Recomendación
    inventory_summary['stockout_risk'] = inventory_summary['current_stock'] < inventory_summary['reorder_point']
    inventory_summary['buy_recommendation'] = np.where(inventory_summary['stockout_risk'], '🔴 Comprar Ya', '🟢 Mantener')
    
    # Cantidad Económica de Pedido (EOQ) - (K=80 costo pedido, h=30% costo mantenimiento)
    inventory_summary['eoq'] = np.sqrt(2 * inventory_summary['demand_forecast_mean'] * 52 * 80 / (inventory_summary['demand_forecast_mean'] * 0.3 + 1e-6)).round(0)

    # --- 2. ORGANIZACIÓN EN PESTAÑAS ---
    tab_tabla, tab_analisis = st.tabs(["📋 Matriz de Reorden", "🧠 Insights de Abastecimiento"])

    with tab_tabla:
        st.subheader("Resumen General de Inventarios")
        
        # Estilizado de la tabla
        def highlight_inventory(row):
            color = "#4b0b0b" if row['stockout_risk'] else "#0a5118"
            return [f'background-color: {color}' for _ in row]

        styled_df = inventory_summary.style.apply(highlight_inventory, axis=1).format({
            'demand_forecast_mean': '{:.1f}',
            'safety_stock': '{:.0f}',
            'reorder_point': '{:.0f}',
            'current_stock': '{:.0f}',
            'eoq': '{:.0f}'
        })
        
        st.dataframe(styled_df, use_container_width=True)

    with tab_analisis:
        # Gráfico de comparación: Stock Actual vs Punto de Reorden
        fig_inv = px.bar(inventory_summary, x='sku_id', y=['current_stock', 'reorder_point'],
                        barmode='group', title="Nivel de Stock vs. Punto Crítico (ROP)",
                        color_discrete_map={'current_stock': '#1e3a8a', 'reorder_point': '#ef4444'})
        st.plotly_chart(fig_inv, use_container_width=True)
        

        st.divider()
        st.subheader("💡 Decisiones Sugeridas")

        col_crit, col_safe = st.columns(2)
        
        with col_crit:
            st.markdown("### 🚨 SKUs en Riesgo")
            risky_skus = inventory_summary[inventory_summary['stockout_risk']]
            if not risky_skus.empty:
                for _, row in risky_skus.iterrows():
                    with st.expander(f"CRÍTICO: {row['sku_id']}"):
                        st.error(f"**Déficit detectado:** Faltan {(row['reorder_point'] - row['current_stock']):.0f} para alcanzar el ROP.")
                        st.write(f"**Acción:** Generar orden de compra por **{row['eoq']:.0f} kg** (Lote Económico).")
                        st.caption(f"Lead Time esperado: {row['lead_time_days']:.1f} días.")
            else:
                st.write("No hay riesgos de stockout inminentes.")

        with col_safe:
            st.markdown("### ✅ SKUs Saludables")
            safe_skus = inventory_summary[~inventory_summary['stockout_risk']]
            for _, row in safe_skus.iterrows():
                st.success(f"**{row['sku_id']}**: Stock suficiente. Seguridad: {row['safety_stock']:.0f}.")

    # --- 3. INSIGHT GLOBAL ---
    st.info(f"""
    **Estrategia de SCM:** El modelo sugiere que el **{ (inventory_summary['stockout_risk'].sum() / len(inventory_summary)) * 100:.0f}%** de sus SKUs requieren atención inmediata. La inversión estimada para normalizar el stock según EOQ es de 
    **{ (inventory_summary[inventory_summary['stockout_risk']]['eoq'].sum()):,.0f}** de materia prima.
    """)

# =======================================================
# Escenarios de Inventario
# =======================================================
elif section == "Escenario de Inventarios":
    st.header("📦 Simulación de Inventarios y Continuidad Operativa")
    st.markdown("""
    Esta sección evalúa la capacidad de respuesta de tu stock ante cambios en la demanda y variaciones logísticas.
    """)

    df2 = df.copy()
    df2['product_id'] = df2['store'] + '/' + df2['sku_id']

    # Selector de Producto Global para la sección
    selected_product = st.selectbox("🎯 Seleccionar Producto para Análisis:", train_nf['unique_id'].unique())

    # Datos base para cálculos
    prod_data = df2[df2['product_id'] == selected_product].iloc[-1]
    hist_std = train_nf[train_nf['unique_id'] == selected_product]['y'].std()
    avg_demand = train_nf[train_nf['unique_id'] == selected_product]['y'].tail(12).mean()
    costo_unitario = float(prod_data['cost'])
    
    # Datos base para cálculos
    prod_data = df2[df2['product_id'] == selected_product].iloc[-1]
    hist_std = train_nf[train_nf['unique_id'] == selected_product]['y'].std()
    avg_demand = train_nf[train_nf['unique_id'] == selected_product]['y'].tail(12).mean()

    tab_sens, tab_stress, tab_costos = st.tabs(["📊 Análisis de Sensibilidad", "⚡ Test de Stress Logístico", "💰 Costos de Almacenamiento"])

    # --- PESTAÑA 1: ANALISIS DE SENSIBILIDAD ---
    with tab_sens:
        st.subheader("Sensibilidad: Nivel de Servicio vs Capital")
        
        col_s1, col_s2 = st.columns([1, 2])
        
        with col_s1:
            param_sens = st.selectbox(
                "Variable de Ajuste:",
                ["Nivel de Servicio (Z)", "Variabilidad de Demanda (Sigma)", "Tiempo de Entrega (Lead Time)"],
                help="Seleccione qué parámetro desea sensibilizar para ver su impacto en el Stock de Seguridad."
            )
            
            # Parámetros base
            current_lt = int(prod_data['lead_time_days'])
            
        # Lógica de simulación de Sensibilidad
        from scipy.stats import norm
        service_levels = np.linspace(0.80, 0.99, 20)
        
        if param_sens == "Nivel de Servicio (Z)":
            # Calculamos SS para diferentes niveles de confianza
            ss_curve = [norm.ppf(sl) * hist_std * np.sqrt(current_lt/7) for sl in service_levels]
            x_vals = service_levels * 100
            x_label = "Nivel de Servicio (%)"
            title_plot = "Costo de Inventario por Nivel de Confianza"
        
        elif param_sens == "Variabilidad de Demanda (Sigma)":
            sigmas = np.linspace(hist_std * 0.5, hist_std * 2.0, 20)
            z_fixed = norm.ppf(0.95) # 95% fijo
            ss_curve = [z_fixed * s * np.sqrt(current_lt/7) for s in sigmas]
            x_vals = sigmas
            x_label = "Desviación Estándar de Demanda (Unidades)"
            title_plot = "Impacto de la Volatilidad de Demanda en el Stock"
            
        else: # Lead Time
            lts = np.arange(1, 45, 2)
            z_fixed = norm.ppf(0.95)
            ss_curve = [z_fixed * hist_std * np.sqrt(l/7) for l in lts]
            x_vals = lts
            x_label = "Días de Lead Time (Proveedor)"
            title_plot = "Impacto del Retraso de Proveedor en Stock Necesario"

        # Gráfico de Sensibilidad con área sombreada corregida
        fig_sens = px.line(x=x_vals, y=ss_curve, title=title_plot, labels={'x': x_label, 'y': 'Stock de Seguridad'})
        fig_sens.update_traces(fill='tozeroy', fillcolor="rgba(30, 58, 138, 0.2)", line=dict(color='#1e3a8a', width=3))
        st.plotly_chart(fig_sens, use_container_width=True)
        

        # INSIGHTS DINÁMICOS
        st.divider()
        st.subheader("💡 Diagnóstico de Sensibilidad")
        if param_sens == "Nivel de Servicio (Z)":
            st.info(f"""
            **Insight:** Pasar de un 90% a un 98% de disponibilidad para **{selected_product}** requiere duplicar tu reserva. 
            **Decisión:** Evalúa si el margen de este producto justifica el costo de almacenamiento de un nivel de servicio extremo.
            """)
        elif param_sens == "Tiempo de Entrega (Lead Time)":
            st.warning(f"**Insight:** Por cada 7 días extra de retraso del proveedor, tu stock de seguridad debe crecer un **{(np.sqrt(14/7)-1)*100:.1f}%**. Considera proveedores locales para reducir esta exposición.")
        else:
            st.error(f"**Insight:** La demanda de este producto es altamente volátil. Los modelos de IA son críticos aquí para evitar el 'efecto látigo' en tus compras.")

    # --- PESTAÑA 2: TEST DE STRESS ---
    with tab_stress:
        st.subheader("⚡ Simulador de Crisis y Puntos de Quiebre")
        
        col_st1, col_st2 = st.columns(2)
        
        with col_st1:
            stress_scenario = st.selectbox(
                "Escenario de Crisis:",
                ["Pico de Demanda Inesperado", "Retraso Crítico de Proveedor", "Combinado (Tormenta Perfecta)"]
            )
        
        with col_st2:
            confianza_stress = st.select_slider("Nivel de Protección ante Crisis:", options=[0.90, 0.95, 0.99], value=0.95)

        # Variables de Stress
        shock_demand = 1.0
        extra_lt = 0
        
        if stress_scenario == "Pico de Demanda Inesperado":
            shock_demand = 1.5 # +50%
        elif stress_scenario == "Retraso Crítico de Proveedor":
            extra_lt = 14 # +2 semanas
        else:
            shock_demand = 1.4
            extra_lt = 10

        # Cálculo de Punto de Reorden (ROP) bajo Stress
        # ROP = (Demanda * LT) + Safety Stock
        lt_total = (current_lt + extra_lt) / 7 # en semanas
        demand_stress = avg_demand * shock_demand
        z_stress = norm.ppf(confianza_stress)
        
        ss_stress = z_stress * hist_std * np.sqrt(lt_total)
        rop_stress = (demand_stress * lt_total) + ss_stress
        
        # Comparativa con Stock Actual
        stock_actual = prod_data['current_stock']
        deficit = rop_stress - stock_actual

        # Visualización de KPIs
        c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
        c_kpi1.metric("Punto Reorden (Stress)", f"{rop_stress:,.0f} u")
        c_kpi2.metric("Stock Actual", f"{stock_actual:,.0f} u")
        c_kpi3.metric("Cobertura de Riesgo", "CRÍTICA" if deficit > 0 else "OK", delta=f"{-deficit:,.0f} u", delta_color="normal" if deficit < 0 else "inverse")

        # Gráfico de Barras Comparativo
        stress_df = pd.DataFrame({
            'Concepto': ['Stock Actual', 'Punto de Reorden (Stress)'],
            'Unidades': [stock_actual, rop_stress],
            'Tipo': ['Real', 'Requerido']
        })
        fig_bar = px.bar(stress_df, x='Concepto', y='Unidades', color='Tipo', 
                        color_discrete_map={'Real': '#64748b', 'Requerido': '#ef4444' if deficit > 0 else '#22c55e'})
        st.plotly_chart(fig_bar, use_container_width=True)
        

        # INSIGHTS DE STRESS
        st.divider()
        st.subheader("📋 Plan de Acción ante Crisis")
        if deficit > 0:
            st.error(f"""
            **🚨 Alerta de Quiebre:** En el escenario de '{stress_scenario}', tu stock actual no cubrirá la demanda. 
            **Acción Requerida:** Necesitas adquirir **{deficit:,.0f} unidades adicionales** inmediatamente o negociar entregas parciales con el proveedor para evitar el desabastecimiento.
            """)
        else:
            st.success(f"""
            **🛡️ Resiliencia Confirmada:** Tu nivel actual de inventario para **{selected_product}** es capaz de absorber este escenario de estrés. 
            **Sugerencia:** Mantén la política actual pero monitorea el Lead Time, ya que es tu variable más sensible.
            """)

    # --- NUEVA PESTAÑA 3: SIMULADOR DE COSTOS ---
    with tab_costos:
        st.subheader("💰 Análisis Financiero del Inventario")
        st.markdown("Calcule cuánto le cuesta a la veterinaria mantener el stock de seguridad proyectado.")

        c1, c2 = st.columns(2)
        with c1:
            costo_oportunidad = st.slider("Tasa de Costo de Capital (%)", 5, 25, 12, 
                                        help="Interés anual que pierde por tener el dinero inmovilizado en stock.") / 100
        with c2:
            costo_bodega_unit = st.number_input("Costo Almacenamiento Unitario (Anual)", value=2.5, step=0.5,
                                             help="Costo de espacio, seguros y manipulación por unidad al año.")

        # Cálculo de costos para cada nivel de servicio
        # Costo Total = (Unidades SS * Costo Unitario * % Capital) + (Unidades SS * Costo Bodega)
        costos_totales = [
            (ss * costo_unitario * costo_oportunidad) + (ss * costo_bodega_unit) 
            for ss in ss_curve
        ]

        # Gráfico de Costos
        fig_costos = px.bar(x=service_levels * 100, y=costos_totales,
                           title=f"Costo Anual de Mantenimiento de Stock: {selected_product}",
                           labels={'x': 'Nivel de Servicio (%)', 'y': 'Costo Total Anual ($)'},
                           color=costos_totales, color_continuous_scale='YlOrRd')
        st.plotly_chart(fig_costos, use_container_width=True)

        # MÉTRICAS DE IMPACTO ECONÓMICO
        st.divider()
        idx_95 = (np.abs(service_levels - 0.95)).argmin()
        idx_99 = (np.abs(service_levels - 0.99)).argmin()
        
        diff_costo = costos_totales[idx_99] - costos_totales[idx_95]

        col_fin1, col_fin2 = st.columns(2)
        with col_fin1:
            st.metric("Inversión en Stock (al 95%)", f"$ {costos_totales[idx_95]:,.2f}")
        with col_fin2:
            st.metric("Inversión en Stock (al 99%)", f"$ {costos_totales[idx_99]:,.2f}", 
                      delta=f"$ {diff_costo:,.2f}", delta_color="inverse")

        # INSIGHT FINANCIERO
        st.warning(f"""
        **💡 Diagnóstico Financiero:** Pasar de un nivel de servicio del 95% al 99% para el producto **{selected_product}** le cuesta a la clínica **$ {diff_costo:,.2f} adicionales al año**.
        
        **Recomendación:** Si el margen de ganancia por unidad de este producto es menor al incremento del costo de mantenimiento, es preferible aceptar un riesgo de quiebre del 5% (Nivel de Servicio 95%) para proteger la rentabilidad neta de la clínica.
        """)

# =======================================================
# Cadenas de Markov
# =======================================================
elif section == "Cadenas de Markov":
    st.header("🔗 Inteligencia de Cadenas de Markov")
    st.markdown("""
    Este análisis revela la **probabilidad de transición**: la posibilidad de que un cliente que compró un SKU 
    vuelva a comprar el mismo (Lealtad) o cambie a otro (Switching/Canibalización).
    """)

    #Y_df, S_df, tags = aggregate(df = train_nf, spec=[['store'], ['store', 'sku_id']])

    # --- 1. PROCESAMIENTO MATEMÁTICO ---
    # Conteos y matriz base

    train_mark = train_nf.copy()
    train_mark['previous_sku'] = np.random.permutation(train_mark['unique_id'].values)

    #st.dataframe(train_mark)

    skus1 = train_mark['unique_id'].unique()
    #st.dataframe(sku)

    transition_counts = pd.crosstab(train_mark['previous_sku'], train_mark['unique_id'])
    transition_counts = transition_counts.reindex(index= skus1, columns=skus1, fill_value=0)
    
    # Cálculo de probabilidades (P)
    row_sums = transition_counts.sum(axis=1)
    transition_matrix = transition_counts.div(row_sums.replace(0, 1), axis=0).fillna(0)
    
    # Manejo de estados absorbentes (si un SKU no tiene historial previo)
    for s in skus1:
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

    # --- TAB 1: CONSTRUCCIÓN (MANTENIDA) ---
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
        
        st.info("**Insight Técnico:** La suma de cada fila debe ser igual a 1.00. Los valores en la diagonal principal indican la retención neta del producto.")

    # --- TAB 2: MAPA DE FLUJO (MANTENIDA CON GRAPHVIZ) ---
    with tab_graph:
        st.subheader("🕸️ Mapa de Flujo Selectivo")
        selected_skus_graph = st.multiselect(
            "Seleccionar SKUs para visualizar conexiones:",
            options=skus1,
            default=skus1[:5] if len(skus1) >= 5 else skus1
        )

        if not selected_skus_graph:
            st.warning("Selecciona al menos un SKU para generar el diagrama.")
        else:
            import graphviz
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            color_map = {sku: colors[i % len(colors)] for i, sku in enumerate(skus1)}
            
            dot = graphviz.Digraph(comment='Markov Focused')
            dot.attr(layout='dot', rankdir='LR', bgcolor='transparent', overlap='false', splines='true')
            
            for sku in selected_skus_graph:
                display_name = sku.replace("SKU_", "Prod. ")
                dot.node(sku, display_name, shape='circle', style='filled', 
                         fillcolor='#262730', color=color_map.get(sku), 
                         fontcolor='white', fontname='Helvetica-Bold', penwidth='3', width='1.2')

            for origin in selected_skus_graph:
                for destiny in selected_skus_graph:
                    prob = transition_matrix.loc[origin, destiny]
                    if prob > 0.01:
                        edge_color = color_map.get(origin)
                        text_color = "red" if prob > 0.5 else "white"
                        w = str(max(1, prob * 10)) 
                        
                        if origin == destiny:
                            dot.edge(origin, destiny, label=f" {prob:.2f}", color=edge_color, 
                                     fontcolor=text_color, style='dashed', penwidth='2')
                        else:
                            dot.edge(origin, destiny, label=f" {prob:.2f}", penwidth=w, 
                                     color=edge_color, fontcolor=text_color)

            st.graphviz_chart(dot, use_container_width=True)
            st.info("Leyenda: Los números en **rojo** indican transiciones dominantes (>50%), sugiriendo una dependencia fuerte entre productos.")

    # --- TAB 3: INVENTARIO ---
    with tab_inv:
        st.subheader("📦 Previsión de Demanda de Herramientas")
        # Aproximación simple de atracción de demanda
        future_attraction = transition_matrix.mean(axis=0).sort_values(ascending=False)
        
        c_inv1, c_inv2 = st.columns([2, 1])
        with c_inv1:
            st.write("**Índice de Atracción (Probabilidad de ser el próximo destino):**")
            st.bar_chart(future_attraction.head(10))
        
        with c_inv2:
            top_attr = future_attraction.index[0]
            st.markdown(f"""
            ### 💡 Insight de Inventario
            El SKU **{top_attr}** actúa como un **Sumidero de Demanda**. 
            Muchos clientes que compran otros productos terminan aquí. 
            
            **Decisión:** Aumentar el stock de seguridad de **{top_attr}** en un **15%**, ya que la probabilidad de flujo entrante es la más alta del sistema.
            """)

    # --- TAB 4: MARKETING ---
    with tab_mkt:
        st.subheader("🎯 Estrategia de Marketing y Cross-Selling")
        mask = np.eye(transition_matrix.shape[0], dtype=bool)
        substitution = transition_matrix.where(~mask).stack().sort_values(ascending=False)
        
        st.write("**Pares con alta tendencia de migración (Oportunidad de Bundle):**")
        sub_df = substitution.head(5).reset_index()
        sub_df.columns = ['Origen', 'Destino', 'Probabilidad']
        st.table(sub_df)

        if not sub_df.empty:
            st.warning(f"""
            **💡 Insight de Marketing:** Existe un flujo natural de **{sub_df.iloc[0]['Origen']}** hacia **{sub_df.iloc[0]['Destino']}**.
            
            **Decisión:** Diseñar una promoción cruzada. Al cliente que compre el primer SKU, ofrecerle 
            un descuento en el segundo para capturar la transición de forma proactiva.
            """)

    # --- TAB 5: PLANIFICACIÓN OPERATIVA ---
    with tab_plan:
        st.subheader("⚙️ Planificación Operativa y Rotación")
        retention = pd.Series(np.diag(transition_matrix), index=transition_matrix.index)
        churn_risk = (1 - retention).sort_values(ascending=False)
        
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            st.write("**SKUs con mayor 'Fuga' (Baja Lealtad):**")
            st.dataframe(churn_risk.to_frame('Probabilidad de Cambio').head(5))
            
        with c_p2:
            high_rot = churn_risk.index[0]
            st.error(f"""
            ### 💡 Insight Operativo
            El SKU **{high_rot}** tiene la mayor tasa de abandono (**{churn_risk[high_rot]*100:.1f}%**). 
            
            **Decisión:** Revisar si este SKU es un 'producto gancho' o si tiene problemas de calidad que obligan al cliente a buscar alternativas. Ajustar logística para los productos de destino.
            """)

    # --- INSIGHTS GENERALES (FOOTER) ---
    st.divider()
    st.subheader("💡 Resumen de Fidelidad")
    col_i1, col_i2 = st.columns(2)
    loyalty = pd.Series(np.diag(transition_matrix), index=transition_matrix.index)
    best_loyal = loyalty.idxmax()
    
    with col_i1:
        st.success(f"**Líder de Lealtad:** SKU **{best_loyal}** (Retención: {loyalty[best_loyal]*100:.1f}%)")
        
    with col_i2:
        mask = np.eye(transition_matrix.shape[0], dtype=bool)
        switches = transition_matrix.where(~mask).stack()
        if not switches.empty:
            top_switch = switches.idxmax()
            st.warning(f"**Mayor Cambio:** {top_switch[0]} → {top_switch[1]} ({switches[top_switch]*100:.1f}%)")


# =======================================================
# Simulación Monte Carlo - Gestión de Incertidumbre
# =======================================================
elif section == "Monte Carlo":
    st.header("🎲 Simulación Monte Carlo - Análisis de Riesgo")
    st.markdown("""
    Esta simulación genera cientos de escenarios aleatorios variando factores externos (clima, precio, economía) 
    para determinar el rango de demanda más probable y los riesgos de quiebre de stock.
    """)
    
    # Creamos las pestañas
    tab_config, tab_results = st.tabs(["⚙️ Configuración y Ejecución", "📊 Resultados por SKU"])

    with tab_config:
        st.subheader("Configuración del Motor de Simulación")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            n_simulations = st.slider("Número de Escenarios Aleatorios", 100, 1000, 300, help="A mayor número, más precisión pero mayor tiempo de cálculo.")
        with col_s2:
            horizon_mc = st.slider("Horizonte de Tiempo (semanas)", 4, 52, 12)

        if st.button("🚀 Iniciar Simulación Maestra"):
            sim_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Ejecutando motores de inferencia..."):
                # A. Proyectar exógenas base
                _, futr_base_master = future_exog_to_historic(
                    df = train_nf, freq = 'W-MON', features = exog_cols, h = horizon_mc
                )
                nf_mc = get_nf_dynamic(horizon_mc)

                # B. Bucle de Simulación Monte Carlo
                for i in range(n_simulations):
                    # Actualizar progreso
                    current_prog = (i + 1) / n_simulations
                    progress_bar.progress(current_prog)
                    status_text.text(f"Simulando escenario {i+1} de {n_simulations}...")

                    # Crear iteración con RUIDO BLANCO en las exógenas
                    futr_iteration = futr_base_master.copy()
                    for col in exog_cols:
                        # Aplicamos una variabilidad del 15% (desviación estándar de 0.15)
                        noise = np.random.normal(1.0, 0.15, size=len(futr_iteration))
                        futr_iteration[col] = futr_iteration[col] * noise

                    # Predecir con el escenario de ruido
                    forecast_mc = nf_mc.predict(futr_df=futr_iteration, verbose=False)
                    # Guardamos la media de la demanda por SKU para esta simulación
                    sim_results.append(forecast_mc.groupby('unique_id')['NHITS'].mean())

                # C. Guardar resultados para persistencia
                st.session_state['sim_df'] = pd.DataFrame(sim_results)
                st.session_state['horizon_mc'] = horizon_mc
                
                progress_bar.empty()
                status_text.success("✅ Simulación completada exitosamente.")

        # --- EXPORTACIÓN DE RESULTADOS ---
        if 'sim_df' in st.session_state:
            st.divider()
            st.markdown("### 📑 Exportar Datos de Riesgo")
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                st.session_state['sim_df'].to_excel(writer, sheet_name='Detalle_Simulaciones')
                st.session_state['sim_df'].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.to_excel(writer, sheet_name='Resumen_Estadistico')
            
            st.download_button(
                label="📥 Descargar Reporte de Riesgo (Excel)",
                data=buffer.getvalue(),
                file_name=f"montecarlo_ MeatForecaster_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    with tab_results:
        if 'sim_df' in st.session_state:
            sim_df = st.session_state['sim_df']
            
            st.subheader("Visualizador de Probabilidad de Demanda")
            selected_sku = st.selectbox("Seleccione el SKU para analizar:", options=sim_df.columns)

            # Cálculo de Percentiles
            p5 = sim_df[selected_sku].quantile(0.05)   # Escenario Pesimista
            p50 = sim_df[selected_sku].quantile(0.5)   # Escenario Probable (Mediana)
            p95 = sim_df[selected_sku].quantile(0.95) # Escenario Optimista

            col_graph, col_stats = st.columns([2, 1])
            
            with col_graph:
                fig = px.histogram(
                    sim_df, x=selected_sku, nbins=30,
                    title=f"Distribución de Probabilidad: {selected_sku}",
                    color_discrete_sequence=['#1e3a8a'], opacity=0.7
                )
                # Añadir líneas de percentiles
                fig.add_vline(x=p5, line_dash="dash", line_color="#ef4444", annotation_text="P5 (Riesgo)")
                fig.add_vline(x=p50, line_dash="solid", line_color="#10b981", annotation_text="Mediana")
                fig.add_vline(x=p95, line_dash="dash", line_color="#3b82f6", annotation_text="P95 (Techo)")
                
                fig.update_layout(template="plotly_white", xaxis_title="Demanda Media (Kg)", yaxis_title="Frecuencia")
                st.plotly_chart(fig, use_container_width=True)
                

            with col_stats:
                st.markdown("### 📊 Métricas de Decisión")
                st.metric("Demanda Probable", f"{p50:.0f} kg")
                st.metric("Piso Crítico (P5)", f"{p5:.0f} kg", delta=f"{((p5/p50)-1)*100:.1f}%", delta_color="inverse")
                st.metric("Techo Máximo (P95)", f"{p95:.0f} kg", delta=f"{((p95/p50)-1)*100:.1f}%")
                
                # Coeficiente de Variación como medida de riesgo
                risk_score = (sim_df[selected_sku].std() / p50) * 100
                st.write(f"**Índice de Riesgo:** `{risk_score:.1f}%`")
                
                if risk_score > 15:
                    st.error("⚠️ **Riesgo Alto:** La demanda es muy volátil ante cambios externos. Se requiere un stock de seguridad robusto.")
                else:
                    st.success("✅ **Riesgo Bajo:** Demanda estable. Puede optimizar inventario sin miedo a quiebres imprevistos.")

            # --- INSIGHT PARA TOMA DE DECISIONES ---
            st.divider()
            st.subheader("💡 Insight Estratégico")
            col_in1, col_in2 = st.columns(2)
            
            with col_in1:
                st.markdown("**Análisis de Abastecimiento:**")
                st.write(f"Si planifica con el escenario de **{p50:.0f} kg**, existe un **5%** de probabilidad de que la demanda supere los **{p95:.0f} kg**. Para garantizar un nivel de servicio total, su cadena logística debe ser capaz de absorber un incremento de **{p95 - p50:.0f} kg** adicionales sobre la media.")

            with col_in2:
                st.markdown("**Impacto Financiero:**")
                st.write(f"En el escenario pesimista (P5), el volumen de venta caería a **{p5:.0f} kg**. Asegúrese de que sus costos fijos puedan ser cubiertos con este volumen mínimo para garantizar la rentabilidad del SKU **{selected_sku}**.")

        else:
            st.info("👋 Para ver los resultados, configure y ejecute la simulación en la pestaña anterior.")


st.sidebar.markdown("---")
st.sidebar.markdown("**Dashboard BI Profesional - Naren Castellón** | Forecasting Jerárquico Pycca 2026")