# train_model_local.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import joblib

# --------- Rutas ---------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
MODEL_PATH = BASE_DIR / "rf_final_model_local.pkl"

print(f"Usando train.csv en: {DATA_PATH}")

# --------- 1. Cargar datos ---------
df_train = pd.read_csv(DATA_PATH)
df_train["date"] = pd.to_datetime(df_train["date"])
df_train = df_train.sort_values(["store", "item", "date"])

last_date_train = df_train["date"].max()
print(f"Última fecha en train: {last_date_train}")

# --------- 2. Construir features (igual lógica que en la app) ---------
def build_feature_dataframe(df_train, max_future_date=None):
    df_train = df_train.copy()
    df_train = df_train.sort_values(["store", "item", "date"])

    # En el entrenamiento solo necesitamos hasta la última fecha real
    df_all = df_train[["date", "store", "item", "sales"]].copy()

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
            df_all.groupby(["store", "item"])["sales"]
                  .shift(lag)
        )

    # Rolling windows
    for win in [7, 30]:
        df_all[f"rolling_mean_{win}"] = (
            df_all.groupby(["store", "item"])["sales"]
                  .shift(1)
                  .rolling(win)
                  .mean()
        )
        df_all[f"rolling_std_{win}"] = (
            df_all.groupby(["store", "item"])["sales"]
                  .shift(1)
                  .rolling(win)
                  .std()
        )

    return df_all

print("Construyendo features...")
df_all = build_feature_dataframe(df_train)

feature_cols = [
    "store", "item", "day", "dayofweek", "weekofyear", "month", "year", "is_weekend",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_30",
    "rolling_std_7", "rolling_std_30"
]

# Eliminamos filas con NaN en features o en sales (inicio de las series)
df_train_feats = df_all.dropna(subset=feature_cols + ["sales"])

X_train = df_train_feats[feature_cols]
y_train = df_train_feats["sales"]

print(f"Tamaño X_train: {X_train.shape}, y_train: {y_train.shape}")

# --------- 3. Entrenar Random Forest ---------
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)

print("Entrenando Random Forest...")
rf_model.fit(X_train, y_train)
print("Entrenamiento completado.")

# --------- 4. Guardar modelo ---------
joblib.dump(rf_model, MODEL_PATH)
print(f"Modelo guardado en: {MODEL_PATH}")
