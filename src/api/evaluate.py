import yaml
import mlflow
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from supabase_client import get_supabase_client

supabase = get_supabase_client()

def load_models(models_dict: dict):
    params = yaml.safe_load(open("../../params.yaml"))
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


def load_model_metrics(metrics_dict: dict):
    params = yaml.safe_load(open("../../params.yaml"))
    stations = params["stations"]
    params_train = params["train"]

    mlflow.set_tracking_uri(uri=params_train["mlflow_uri"])
    mlflow_registered_model_name_template = params_train["mlflow_registered_model_name"]

    client = mlflow.tracking.MlflowClient()

    model_names = [mlflow_registered_model_name_template.format(station=station) for station in stations]

    for model_name in model_names:
        model = client.get_registered_model(name=model_name)
        for latest_version in model.latest_versions:
            # get latest version run
            run_id = latest_version.run_id

            # fetch metrics for this model version
            run_info = client.get_run(run_id=run_id)
            metrics = run_info.data.metrics

            # extract relevant evaluation metrics
            metrics_dict[model_name] = {
                "mae": metrics.get("test_mae"),
                "mse": metrics.get("test_mse"),
                "rmse": metrics.get("test_rmse"),
                "run_id": run_id
            }

    print(f"Loaded evaluation metrics: {list(metrics_dict.values())}")


def load_model_params(params_dict: dict):
    params = yaml.safe_load(open("../../params.yaml"))
    stations = params["stations"]
    params_train = params["train"]

    mlflow.set_tracking_uri(uri=params_train["mlflow_uri"])
    mlflow_registered_model_name_template = params_train["mlflow_registered_model_name"]

    client = mlflow.tracking.MlflowClient()

    model_names = [mlflow_registered_model_name_template.format(station=station) for station in stations]

    for model_name in model_names:
        model = client.get_registered_model(name=model_name)
        for latest_version in model.latest_versions:
            # get latest version run
            run_id = latest_version.run_id

            # fetch metrics for this model version
            run_info = client.get_run(run_id=run_id)
            params = run_info.data.params

            # extract relevant evaluation metrics
            params_dict[model_name] = {
                "test_size": params.get("test_size"),
                "val_size": params.get("val_size"),
                "lookback": params.get("lookback"),
                "forecast_horizon": params.get("forecast_horizon"),
                "batch_size": params.get("batch_size"),
                "dropout": params.get("dropout"),
                "learning_rate": params.get("learning_rate"),
                "patience": params.get("patience"),
                "epochs": params.get("epochs"),
                "run_id": run_id
            }

    print(f"Loaded model hyperparameters: {list(params_dict.values())}")


def load_model_by_name(model_name: str, models_dict: dict):
    params_train = yaml.safe_load(open("../../params.yaml"))["train"]
    mlflow.set_tracking_uri(uri=params_train["mlflow_uri"])

    client = mlflow.tracking.MlflowClient()

    model = client.get_registered_model(name=model_name)
    for latest_version in model.latest_versions:
        model_uri = latest_version.source
        models_dict[model_name] = mlflow.pyfunc.load_model(model_uri)

    print(f"Loaded model: {model_name}")


def generate_forecast_timestamps(timestamps: list[str]) -> list[str]:
    today = datetime.today().date()

    datetime_format = "%Y-%m-%d %H:%M"

    last_timestamp: str = timestamps[-1]
    last_datetime: datetime = datetime.strptime(f"{today} {last_timestamp}", datetime_format)

    forecast_timestamps: list[str] = [
        (last_datetime + timedelta(minutes=30 * i)).strftime(datetime_format)
        for i in range(1, 7)
    ]
    return forecast_timestamps


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

    print("df: ", df)

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

    predictions_list: list[float] = [float(round(val, 1)) for val in predictions_rescaled.tolist()[0]]
    forecast_timestamps: list[str] = generate_forecast_timestamps(timestamps=timestamps_parsed)
    forecast_timestamps_parsed = [timestamp.split(' ')[1] for timestamp in forecast_timestamps]
    # all_timestamps = timestamps_parsed + forecast_timestamps_parsed

    on_conflict_cols = "station"

    # insert predictions and their corresponding timestamps into db
    response = supabase.table("forecast").upsert(
        {
            "predictions": predictions_list,
            "timestamps": forecast_timestamps,
            "station": station
        },
        on_conflict=on_conflict_cols,
        ignore_duplicates=True
    ).execute()

    print(f"Supabase insert into table 'forecast' response: {response}")

    return predictions_list, actuals, timestamps_parsed


def get_latest_forecasts():
    response = supabase.from_('forecast').select('*').execute()
    return response.data


def get_latest_forecast_by_station(station_name: str):
    response = supabase.from_('forecast').select('*').eq('name', station_name).single().execute()
    return response.data