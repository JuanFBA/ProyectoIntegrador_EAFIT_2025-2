import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path   # ← Nuevo

# -----------------------------
# 1. Configuración básica
# -----------------------------
st.set_page_config(page_title="Pronóstico de ventas", layout="centered")

st.title("📈 Pronóstico de ventas - Modelo Random Forest Global")

st.markdown(
    """
    Esta aplicación usa un modelo **Random Forest global** entrenado sobre el histórico
    de ventas (2013-2017) para predecir ventas diarias por **tienda** y **producto**.
    
    > El modelo fue entrenado previamente en este mismo entorno local para garantizar compatibilidad.
    """
)

# -----------------------------
# 2. Cargar modelo y datos     
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
MODEL_PATH = BASE_DIR / "rf_final_model_local.pkl"

@st.cache_data
def load_data():
    df_train = pd.read_csv(DATA_PATH)
    df_train["date"] = pd.to_datetime(df_train["date"])
    return df_train

@st.cache_resource
def load_model():
    rf_model = joblib.load(MODEL_PATH)
    return rf_model

df_train = load_data()
rf_model = load_model()

st.success("Modelo y datos históricos cargados correctamente.")

# -----------------------------
# 3. Construir pipeline de features
# -----------------------------
@st.cache_data
def build_feature_dataframe(df_train, max_future_date="2018-03-31"):

    df_train = df_train.copy()
    df_train = df_train.sort_values(["store", "item", "date"])

    last_date = df_train["date"].max()
    max_future_date = pd.to_datetime(max_future_date)

    stores = df_train["store"].unique()
    items = df_train["item"].unique()
    all_pairs = pd.MultiIndex.from_product(
        [stores, items], names=["store", "item"]
    ).to_frame(index=False)

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        end=max_future_date,
        freq="D"
    )

    if len(future_dates) == 0:
        df_all = df_train.copy()
    else:
        future_grid = (
            all_pairs.assign(key=1)
                .merge(pd.DataFrame({"date": future_dates, "key": 1}), on="key")
                .drop("key", axis=1)
        )
        future_grid["sales"] = np.nan

        df_all = pd.concat(
            [df_train[["date", "store", "item", "sales"]], future_grid],
            ignore_index=True
        ).sort_values(["store", "item", "date"]).reset_index(drop=True)

    # Variables de calendario
    df_all["day"]        = df_all["date"].dt.day
    df_all["dayofweek"]  = df_all["date"].dt.dayofweek
    df_all["weekofyear"] = df_all["date"].dt.isocalendar().week.astype(int)
    df_all["month"]      = df_all["date"].dt.month
    df_all["year"]       = df_all["date"].dt.year
    df_all["is_weekend"] = df_all["dayofweek"].isin([5, 6]).astype(int)

    # Lags
    for lag in [1, 7, 14, 28]:
        df_all[f"lag_{lag}"] = (
            df_all.groupby(["store", "item"])["sales"].shift(lag)
        )

    # Rolling windows
    for win in [7, 30]:
        df_all[f"rolling_mean_{win}"] = (
            df_all.groupby(["store", "item"])["sales"]
                  .shift(1).rolling(win).mean()
        )
        df_all[f"rolling_std_{win}"] = (
            df_all.groupby(["store", "item"])["sales"]
                  .shift(1).rolling(win).std()
        )

    return df_all


st.info("Construyendo dataframe con features (esto se hace una sola vez y se cachea)...")
df_all = build_feature_dataframe(df_train, max_future_date="2018-03-31")
st.success("Features construidas correctamente.")

# -----------------------------
# 4. Columnas de features
# -----------------------------
feature_cols = [
    "store", "item", "day", "dayofweek", "weekofyear", "month", "year", "is_weekend",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_30",
    "rolling_std_7", "rolling_std_30"
]

# -----------------------------
# 4.5. Predicción recursiva (multi-día)
# -----------------------------
def predict_recursive(df_all, store, item, start_date, end_date, feature_cols, model):
    """
    Hace pronósticos día a día entre start_date y end_date para un par (store, item),
    usando predicciones anteriores para actualizar lags y rolling.
    Devuelve un DataFrame con columnas: date, prediction.
    """
    # Trabajamos sobre una copia SOLO de esa serie (esa tienda-item)
    df_series = (
        df_all[(df_all["store"] == store) & (df_all["item"] == item)]
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )

    resultados = []

    # Rango de fechas a predecir (por seguridad, acotado al rango disponible)
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    for fecha_actual in pd.date_range(start=start_date, end=end_date, freq="D"):

        # Recalcular lags y rolling con el estado actual de 'sales'
        for lag in [1, 7, 14, 28]:
            df_series[f"lag_{lag}"] = df_series["sales"].shift(lag)

        for win in [7, 30]:
            df_series[f"rolling_mean_{win}"] = (
                df_series["sales"].shift(1).rolling(win).mean()
            )
            df_series[f"rolling_std_{win}"] = (
                df_series["sales"].shift(1).rolling(win).std()
            )

        # Tomar la fila de la fecha actual
        fila = df_series[df_series["date"] == fecha_actual]

        if fila.empty:
            # No hay registro para esta fecha (no debería pasar si df_all cubre hasta max_future_date)
            resultados.append({"date": fecha_actual, "prediction": np.nan})
            continue

        # Verificar si ya tenemos todos los features sin NaN
        if fila[feature_cols].isna().any().any():
            # Aun sin suficiente historial para rolling/lags
            resultados.append({"date": fecha_actual, "prediction": np.nan})
            continue

        # Predecir
        X = fila[feature_cols]   # 👈 DataFrame, conserva nombres de columnas
        y_hat = model.predict(X)[0]
        y_hat = max(y_hat, 0)    # evitamos negativos

        resultados.append({"date": fecha_actual, "prediction": y_hat})

        # Actualizar 'sales' con la predicción para usarla como lag en fechas futuras
        df_series.loc[df_series["date"] == fecha_actual, "sales"] = y_hat

    return pd.DataFrame(resultados)

# -----------------------------
# 5. Interfaz de usuario
# -----------------------------
st.subheader("🔢 Ingrese los datos para la predicción")

col1, col2 = st.columns(2)
with col1:
    store = st.number_input("Store (1-10)", min_value=1, max_value=10, value=1)
with col2:
    item = st.number_input("Item (1-50)", min_value=1, max_value=50, value=1)

min_date = df_train["date"].max() + pd.Timedelta(days=1)
max_date = datetime(2018, 3, 31)

date_input = st.date_input(
    "Fecha a predecir",
    value=min_date,
    min_value=min_date,
    max_value=max_date
)

if st.button("🔮 Predecir ventas"):

    # Fecha inicial de predicción = primer día futuro después del histórico
    start_pred_date = df_train["date"].max() + pd.Timedelta(days=1)
    end_pred_date = pd.to_datetime(date_input)

    if end_pred_date < start_pred_date:
        st.error("La fecha seleccionada debe ser posterior al último día del histórico.")
    else:
        with st.spinner("Calculando pronóstico recursivo..."):
            df_preds = predict_recursive(
                df_all=df_all,
                store=store,
                item=item,
                start_date=start_pred_date,
                end_date=end_pred_date,
                feature_cols=feature_cols,
                model=rf_model,
            )

        # Tomar la predicción de la fecha seleccionada
        fila_objetivo = df_preds[df_preds["date"] == end_pred_date]

        if fila_objetivo.empty or np.isnan(fila_objetivo["prediction"].iloc[0]):
            st.warning(
                "No se pudieron calcular todos los lags/rolling para llegar hasta esa fecha "
                "(revisa que haya suficiente historial)."
            )
        else:
            y_hat = fila_objetivo["prediction"].iloc[0]
            st.success(
                f"📌 Predicción de ventas para {end_pred_date.date()} "
                f"(store={store}, item={item}):"
            )
            st.markdown(f"### 🔵 {y_hat:.0f} unidades")
            st.caption(
                "La predicción utiliza un esquema recursivo: "
                "las ventas pronosticadas para días anteriores se incorporan como rezagos "
                "y en los promedios móviles para fechas posteriores."
            )

            # (Opcional) Mostrar tabla con todas las fechas desde 2018-01-01 hasta la seleccionada
            st.write("Pronóstico diario desde el inicio del horizonte futuro:")
            df_preds["prediction"] = df_preds["prediction"].round().astype("Int64")
            st.dataframe(df_preds)

# correr streamlit run app.py