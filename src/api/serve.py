import yaml
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from evaluate import load_models, load_model_metrics, evaluate_model, get_latest_forecasts, get_latest_forecast_by_station
from station import fetch_station_by_name, fetch_stations
from weather import fetch_weather_data_for_station


MODELS = {}
EVAL_METRICS = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models and evaluation metrics on startup...")
    load_models(models_dict=MODELS)
    load_model_metrics(metrics_dict=EVAL_METRICS)
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


@app.get("/")
def root():
    return {
       "models": list(MODELS.keys()),
       "metrics": list(EVAL_METRICS.values())
    }


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
    params_train = yaml.safe_load(open("../../params.yaml"))["train"]
    lookback = params_train["lookback"]
    forecast_horizon = params_train["forecast_horizon"]
    columns_to_drop = params_train["columns_to_drop"]

    station_capitalized = station.capitalize()

    df = fetch_weather_data_for_station(station=station_capitalized, limit=lookback)
    predictions, actuals, timestamps = evaluate_model(station=station, data=df, models_dict=MODELS, forecast_horizon=forecast_horizon, columns_to_drop=columns_to_drop)

    return { "predictions": predictions, "actuals": actuals, "timestamps": timestamps }


@app.get("/evaluate")
def get_latest_forecasts():
    try:
        forecasts = get_latest_forecasts()
        return forecasts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluate/{station}")
def get_latest_forecast(station_name: str):
    try:
        forecast = get_latest_forecast_by_station(station_name=station_name)
        if not forecast:
            raise HTTPException(status_code=404, detail="Forecast not found")
        return forecast
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
