from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import yaml
from fastapi.middleware.cors import CORSMiddleware
from evaluate import load_models, evaluate_model
from station import fetch_station_by_name, fetch_stations
from weather import fetch_weather_data_for_station


MODELS = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models on startup...")
    load_models(models_dict=MODELS)
    yield

app = FastAPI(lifespan=lifespan)

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


# @app.on_event("startup")
# def load_models():
#     load_models(MODELS)


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


@app.get("/stations/{name}")
def get_station_by_name(station_name: str):
    try:
        station = fetch_station_by_name(station_name=station_name)
        if not station:
            raise HTTPException(status_code=404, detail="Station not found")
        return station
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/load_models/{name}")
def load_model_by_name(model_name: str):
    load_model_by_name(model_name=model_name)


@app.get("/predict/{station}")
def predict(station: str):
    params_train = yaml.safe_load(open("../params.yaml"))["train"]
    lookback = params_train["lookback"]
    forecast_horizon = params_train["forecast_horizon"]
    columns_to_drop = params_train["columns_to_drop"]

    station_capitalized = station.capitalize()

    df = fetch_weather_data_for_station(station=station_capitalized, limit=lookback)
    predictions, actuals, timestamps = evaluate_model(station=station, data=df, models_dict=MODELS, forecast_horizon=forecast_horizon, columns_to_drop=columns_to_drop)

    return { "predictions": predictions, "actuals": actuals, "timestamps": timestamps }