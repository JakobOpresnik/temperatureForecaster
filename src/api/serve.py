from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from station import fetch_station_by_name, fetch_stations
from weather import fetch_weather_data_for_station

app = FastAPI()

# allow FE to make requests (production & local)
origins = [
    "https://temperatureforecaster-frontend-production.up.railway.app",
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = {}

@app.on_event("startup")
def load_models():
    stations = yaml.safe_load(open("../../params.yaml"))["stations"]
    params_train = yaml.safe_load(open("../../params.yaml"))["train"]

    mlflow.set_tracking_uri(uri=params_train["mlflow_uri"])
    mlflow_registered_model_name_template = params_train["mlflow_registered_model_name"]

    client = mlflow.tracking.MlflowClient()

    model_names = [mlflow_registered_model_name_template.format(station=station) for station in stations]

    for model_name in model_names:
        model = client.get_registered_model(name=model_name)
        # print(f"  Model name: {model.name}")
        for latest_version in model.latest_versions:
            # print(f"  Latest version: {latest_version.version}")
            # print(f"  Model UR: {latest_version.source}")
            model_uri = latest_version.source
            MODELS[model_name] = mlflow.pyfunc.load_model(model_uri)

    print(f"Loaded models for stations: {list(MODELS.keys())}")


@app.get("/")
def root():
    return list(MODELS.keys())


@app.get("/stations")
def get_stations():
    try:
        stations = fetch_stations()
        return stations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("stations/{name}")
def get_station_by_name(station_name: str):
    try:
        station = fetch_station_by_name(station_name=station_name)
        if not station:
            raise HTTPException(status_code=404, detail="Station not found")
        return station
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/predict/{station}")
def predict(station: str):
    params_train = yaml.safe_load(open("../params.yaml"))["train"]
    lookback = params_train["lookback"]
    forecast_horizon = params_train["forecast_horizon"]
    columns_to_drop = params_train["columns_to_drop"]

    station_capitalized = station.capitalize()

    print("station: ", station)
    print("lookback: ", lookback)

    df = fetch_weather_data_for_station(station=station_capitalized, limit=lookback)

    df = pd.DataFrame(df)

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

    print("MODELS: ", MODELS)
    model = MODELS[model_key]

    predictions = model.predict(X)
    print("predictions: ", predictions)
    
    predictions_2d = predictions.reshape(-1, 1)
    n_samples, forecast_horizon = predictions.shape

    # apply inverse transform
    predictions_rescaled = temp_scaler.inverse_transform(predictions_2d).reshape(n_samples, forecast_horizon)
    print("predictions rescaled: ", predictions_rescaled)

    predictions_list = [round(val, 1) for val in predictions_rescaled.tolist()[0]]

    return { "predictions": predictions_list, "actuals": actuals, "timestamps": timestamps_parsed }


@app.get("/fetch_data/{station}")
def predict_from_db(station: str):
    station = station.upper()
    if station not in MODELS:
        raise HTTPException(status_code=404, detail="Station model not found")
    
    try:
        input_data = fetch_weather_data_for_station(station)
        model = MODELS[station]
        prediction = model.predict(input_data)
        return {"station": station, "prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))