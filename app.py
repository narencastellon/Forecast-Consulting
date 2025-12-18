import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utilsforecast.plotting import plot_series
from statsmodels.tsa.seasonal import seasonal_decompose
from mlforecast import MLForecast
from mlforecast.target_transforms import LocalStandardScaler
from mlforecast.utils import PredictionIntervals
from utilsforecast.feature_engineering import future_exog_to_historic
from utilsforecast.losses import mae, mape, mase, rmse, smape
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings("ignore")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.image("https://cdn.educba.com/academy/wp-content/uploads/2016/05/Business-Forecasting-2.jpg", width=200)
    st.header("Configuración General")
    h = st.selectbox("Horizonte de pronóstico (meses)", [6, 12, 18, 24, 36, 48], index=1)
    
    st.markdown("---")
    st.subheader("Navegación")
    st.info("Selecciona una pestaña en la parte superior para navegar.")

# Función para cargar y preparar los datos
@st.cache_data
def load_and_prepare_data():
    data = pd.read_csv("https://raw.githubusercontent.com/narencastellon/Serie-de-tiempo-con-Machine-Learning/refs/heads/main/Data/hospital_vivian.csv")
    data["ds"] = pd.to_datetime(data["ds"])
    
    results = []
    for uid in data["unique_id"].unique():
        series = data[data["unique_id"] == uid].set_index("ds")["y"]
        series = series.asfreq("MS")
        decomposition = seasonal_decompose(series, model="additive", period=12, extrapolate_trend=1)
        
        temp = pd.DataFrame({
            "unique_id": uid,
            "ds": series.index,
            "y": series.values,
            "trend": decomposition.trend.values,
            "seasonal": decomposition.seasonal.values,
            "resid": decomposition.resid.values
        })
        results.append(temp)
    
    df_decomp = pd.concat(results, ignore_index=True)
    df = df_decomp.copy()
    
    # Crear rezagos
    num_lags = 4
    for lag in range(1, num_lags + 1):
        df[f'lag{lag}'] = df.groupby('unique_id')['y'].shift(lag)
    
    df.dropna(inplace=True)
    
    # Eliminar las últimas 5 filas de cada grupo
    def remove_last_n_rows(group, n=5):
        if len(group) > n:
            return group.iloc[:-n]
        else:
            return pd.DataFrame(columns=group.columns)
    
    datos = df.groupby('unique_id', group_keys=False).apply(remove_last_n_rows, n=5)
    
    return data, df_decomp, datos

# Cargar datos (una sola vez)
original_data, df_decomp, datos = load_and_prepare_data()

# Entrenamiento de modelos
@st.cache_resource
def get_trained_model():
    models = {
        'LinearRegression': LinearRegression(),
        'KNeighbors': KNeighborsRegressor(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
        'CatBoost': CatBoostRegressor(verbose=0, random_state=42)
    }
    
    mlf = MLForecast(
        models=models,
        freq='MS',
        #lags=[1, 2, 3, 4],
        target_transforms=[LocalStandardScaler()],
    )
    
    with st.spinner("Entrenando los modelos (solo la primera vez)..."):
        mlf.fit(datos, static_features = [], fitted = True,
        prediction_intervals = PredictionIntervals(n_windows = 2, h = h, method = "conformal_distribution"))
    
    return mlf

mlf = get_trained_model()

# Función para obtener futuro_all
@st.cache_data
def get_futuro_all(h):
    _, futuro_all = future_exog_to_historic(
        df=datos,
        freq='MS',
        features=['trend', 'seasonal', 'resid', 'lag1', 'lag2', 'lag3', 'lag4'],
        h= h,
    )
    return futuro_all

# Session state para forecast
if 'forecast_df' not in st.session_state:
    st.session_state['forecast_df'] = None
if 'computed_h' not in st.session_state:
    st.session_state['computed_h'] = None
if 'computed_levels' not in st.session_state:
    st.session_state['computed_levels'] = None

# Session state para Cross-Validation
if 'cv_results_raw' not in st.session_state:
    st.session_state['cv_results_raw'] = None
if 'cv_h' not in st.session_state:
    st.session_state['cv_h'] = None

# ====================== INTERFAZ PRINCIPAL ======================
st.title("App de Análisis y Forecasting de Series Temporales")

unique_ids = sorted(original_data['unique_id'].unique())
selected_uid = st.selectbox("Selecciona un unique_id", unique_ids)

tab1, tab2, tab3, tab4 = st.tabs(["Datos Históricos", "EDA", "Forecasting", "Validación"])

with tab1:
    st.header("Datos Históricos")
    historical_df = original_data[original_data['unique_id'] == selected_uid]
    st.dataframe(historical_df)
    
    st.subheader("Gráfico de la Serie Histórica")
    fig_hist = plot_series(historical_df, max_ids=1, plot_random=False,)
    st.pyplot(fig_hist)

with tab2:
    st.header("Análisis Exploratorio de Datos (EDA)")
    
    decomp_df = df_decomp[df_decomp['unique_id'] == selected_uid]
    
    st.subheader("Estadísticas Descriptivas")
    st.write(decomp_df[['y', 'trend', 'seasonal', 'resid']].describe())

    
    st.subheader("Descomposición de la Serie")
    fig_decomp, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(decomp_df['ds'], decomp_df['y'], label='Original')
    axs[0].set_title('Original')
    axs[1].plot(decomp_df['ds'], decomp_df['trend'], color='orange')
    axs[1].set_title('Tendencia')
    axs[2].plot(decomp_df['ds'], decomp_df['seasonal'], color='green')
    axs[2].set_title('Estacionalidad')
    axs[3].plot(decomp_df['ds'], decomp_df['resid'], color='red')
    axs[3].set_title('Residuales')
    plt.tight_layout()
    st.pyplot(fig_decomp)
    
    st.subheader("Autocorrelación")
    from pandas.plotting import autocorrelation_plot
    fig_acf = plt.figure(figsize=(10, 4))
    autocorrelation_plot(decomp_df['y'])
    st.pyplot(fig_acf)

with tab3:
    st.header("Forecasting")
    
    levels = st.multiselect("Niveles de intervalo de predicción", [60, 70, 80, 90, 95], default=[80, 95], key="levels_forecast")
    
    need_compute = (
        st.session_state['forecast_df'] is None or
        st.session_state['computed_h'] != h or
        sorted(st.session_state['computed_levels'] or []) != sorted(levels)
    )
    
    if st.button("Generar/Actualizar Forecast para Todas las Series", key="btn_compute_all"):
        need_compute = True
    
    if need_compute:
        with st.spinner("Generando forecast para todas las series..."):
            futuro_all = get_futuro_all(h)
            forecast_df = mlf.predict(h=h, level=levels, X_df=futuro_all)
            st.session_state['forecast_df'] = forecast_df
            st.session_state['computed_h'] = h
            st.session_state['computed_levels'] = levels
        st.success("Forecast generado para todas las series.")
    
    if st.session_state['forecast_df'] is not None:
        filtered_forecast = st.session_state['forecast_df'][st.session_state['forecast_df']['unique_id'] == selected_uid]
        
        if filtered_forecast.empty:
            st.warning("No hay forecast disponible para esta serie.")
        else:
            historical_uid = original_data[original_data['unique_id'] == selected_uid][['unique_id', 'ds', 'y']]
            
            forecast_mode = st.radio(
                "Selecciona el modo de pronóstico",
                ("Ver todos los modelos", "Ver solo un modelo seleccionado"),
                key="forecast_mode"
            )
            
            if forecast_mode == "Ver todos los modelos":
                st.subheader("Tabla completa - Todos los modelos")
                st.dataframe(filtered_forecast)
                
                st.subheader("Gráfico - Todos los modelos")
                full_df = pd.concat([historical_uid, filtered_forecast])
                fig_forecast = plot_series(full_df, forecasts_df=filtered_forecast, ids=[selected_uid], level=levels,)
                st.pyplot(fig_forecast)
            
            else:
                available_models = [col for col in filtered_forecast.columns if col not in ['unique_id', 'ds'] and not col.startswith(('_lo_', '_hi_'))]
                selected_model = st.selectbox("Selecciona el modelo", available_models, key="single_model")
                
                model_cols = ['unique_id', 'ds', selected_model]
                for lv in levels:
                    model_cols += [f'{selected_model}-lo-{lv}', f'{selected_model}-hi-{lv}']
                selected_forecast = filtered_forecast[model_cols]
                
                st.subheader(f"Tabla de predicciones - {selected_model}")
                st.dataframe(selected_forecast)
                
                st.subheader(f"Gráfico - {selected_model}")
                full_df = pd.concat([historical_uid, selected_forecast])
                fig_forecast = plot_series(full_df, forecasts_df=selected_forecast, ids=[selected_uid])
                st.pyplot(fig_forecast)
    else:
        st.info("Presiona el botón para generar el forecast inicial.")

with tab4:
    st.header("Evaluación con Cross-Validation")
    
    if st.button("Ejecutar Cross-Validation (para el horizonte actual)", key="cv_btn"):
        with st.spinner("Realizando Cross-Validation (puede tomar tiempo)..."):
            cv_results_raw = mlf.cross_validation(
                df=datos,
                n_windows = 2,
                h=h,
                dropna = True,
                #level=[],  # Sin intervals para métricas
                static_features = [],
            )
            st.session_state['cv_results_raw'] = cv_results_raw
            st.session_state['cv_h'] = h
        st.success("Cross-Validation completada y guardada.")
    
    if st.session_state['cv_results_raw'] is not None and st.session_state['cv_h'] == h:
        cv_df = st.session_state['cv_results_raw']
        
        # Columnas de modelos
        model_columns = [col for col in cv_df.columns if col not in ['unique_id', 'ds', 'cutoff', 'y']]
        
        # Selector de métrica
        metric_option = st.selectbox(
            "Selecciona la métrica",
            options=["mae", "mape", "mase", "rmse", "smape"],
            format_func=lambda x: x.upper(),
            key="metric_select_cv"
        )
        
        metric_funcs = {
            "mae": mae,
            "mape": mape,
            "mase": mase,
            "rmse": rmse,
            "smape": smape
        }
        selected_metric = metric_funcs[metric_option]
        
        # Calcular métricas
        eval_df = selected_metric(cv_df, models=model_columns)
        eval_df['best_model'] = eval_df[model_columns].idxmin(axis=1)
        
        # Filtros
        st.subheader("Filtros (opcional)")
        col1, col2 = st.columns(2)
        with col1:
            filter_models = st.multiselect("Filtrar por modelo(s)", options=model_columns, key="filter_models_cv")
        with col2:
            filter_uids = st.multiselect("Filtrar por unique_id(s)", options=eval_df['unique_id'].unique(), key="filter_uids_cv")
        
        # Aplicar filtros
        display_df = eval_df.copy()
        if filter_models:
            cols_to_keep = ['unique_id', 'best_model'] + filter_models
            display_df = display_df[cols_to_keep]
        if filter_uids:
            display_df = display_df[display_df['unique_id'].isin(filter_uids)]
        
        # Mostrar tabla
        st.subheader(f"Resultados de {metric_option.upper()} (por serie y cutoff)")
        st.dataframe(display_df.sort_values(['unique_id', ]))
        
        # Resumen promedio por modelo
        st.subheader(f"Promedio {metric_option.upper()} por modelo (todo el dataset)")
        avg_metrics = eval_df[model_columns].mean().sort_values()
        st.dataframe(avg_metrics.to_frame(name="Promedio"))
        
        # Mejor modelo general
        best_model_overall = avg_metrics.idxmin()
        st.success(f"Mejor modelo general ({metric_option.upper()} promedio): **{best_model_overall}**")
        
    else:
        st.info("Presiona el botón para ejecutar la Cross-Validation con el horizonte seleccionado.")