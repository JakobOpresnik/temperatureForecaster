import yaml
import mlflow
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


def load_models(models_dict: dict):
    params = yaml.safe_load(open("../params.yaml"))
    stations = params["stations"]
    params_train = params["train"]

    mlflow.set_tracking_uri(uri=params_train["mlflow_uri"])
    mlflow_registered_model_name_template = params_train["mlflow_registered_model_name"]

    client = mlflow.tracking.MlflowClient()

    model_names = [mlflow_registered_model_name_template.format(station=station) for station in stations]

    for model_name in model_names:
        model = client.get_registered_model(name=model_name)
        for latest_version in model.latest_versions:
            model_uri = latest_version.source
            models_dict[model_name] = mlflow.pyfunc.load_model(model_uri)

    print(f"Loaded models for stations: {list(models_dict.keys())}")


def load_model_by_name(model_name: str, models_dict: dict):
    params_train = yaml.safe_load(open("../params.yaml"))["train"]
    mlflow.set_tracking_uri(uri=params_train["mlflow_uri"])

    client = mlflow.tracking.MlflowClient()

    model = client.get_registered_model(name=model_name)
    for latest_version in model.latest_versions:
        model_uri = latest_version.source
        models_dict[model_name] = mlflow.pyfunc.load_model(model_uri)

    print(f"Loaded model: {model_name}")


def evaluate_model(station: str, data: DataFrame, models_dict: dict, forecast_horizon: int = 240, columns_to_drop: list = []):
    df = pd.DataFrame(data=data)
    df = df.drop(columns=columns_to_drop)

    df["Date"] = pd.to_datetime(df["Date"])

    df["hour"] = df["Date"].dt.hour
    df["minute"] = df["Date"].dt.minute
    df["dayofweek"] = df["Date"].dt.dayofweek + 1   # make Monday 1st day of the week
    df["month"] = df["Date"].dt.month

    df["total_minutes"] = df["hour"] * 60 + df["minute"]
    df["time_sin"] = np.sin(2 * np.pi * df["total_minutes"] / 1440)
    df["time_cos"] = np.cos(2 * np.pi * df["total_minutes"] / 1440)

    # drop all columns that are entirely NaN
    all_nan_cols = df.columns[df.isna().all()]
    print(f"\nDropping columns with all NaNs: {list(all_nan_cols)}")
    df.drop(columns=all_nan_cols, inplace=True)

    # fill remaining missing values (forward and backward fill)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    print("DF: ", df)

    temp_scaler = MinMaxScaler()
    other_scaler = MinMaxScaler()

    actuals = df["Temperature"].tail(18).tolist()
    timestamps = df["Date"].tail(18).tolist()
    timestamps_parsed = [datetime.fromisoformat(str(timestamp)).strftime("%H:%M") for timestamp in timestamps]

    # fit temperature scaler
    df["Temperature_scaled"] = temp_scaler.fit_transform(df[["Temperature"]])

    other_columns = df.columns.drop(["Date", "Temperature"])  # keep everything except Temperature (target) and Date
    other_features = df[other_columns]
    other_scaled = pd.DataFrame(
        other_scaler.fit_transform(other_features),
        columns=other_columns
    )

    # combine everything
    scaled_df = pd.concat([other_scaled, df[["Temperature_scaled"]].rename(columns={"Temperature_scaled": "Temperature"})], axis=1)

    X = scaled_df.to_numpy(dtype=np.float32)  # shape (240, 14)
    X = X.reshape(1, X.shape[0], X.shape[1])  # reshape to (1, 240, 14)

    print("X: ", X)

    model_key = f"TemperatureForecaster-{station}"
    
    if model_key not in models_dict:
        raise ValueError(f"Model '{model_key}' cannot be loaded.")

    print("MODELS: ", models_dict)
    model = models_dict[model_key]

    predictions = model.predict(X)
    print("predictions: ", predictions)
    
    predictions_2d = predictions.reshape(-1, 1)
    n_samples, forecast_horizon = predictions.shape

    # apply inverse transform
    predictions_rescaled = temp_scaler.inverse_transform(predictions_2d).reshape(n_samples, forecast_horizon)
    print("predictions rescaled: ", predictions_rescaled)

    predictions_list = [round(val, 1) for val in predictions_rescaled.tolist()[0]]

    return predictions_list, actuals, timestamps_parsed