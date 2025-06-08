from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
import yaml
import logging

app = FastAPI()

stations = yaml.safe_load(open("../../params.yaml"))["stations"]
params_train = yaml.safe_load(open("../../params.yaml"))["train"]

mlflow_registered_model_name_template = params_train["mlflow_registered_model_name"]
MODELS = {}

@app.on_event("startup")
def load_models():
    logging.basicConfig(level=logging.INFO)
    logging.info("🚀 FastAPI app has started successfully.")
    for station in stations:
        mlflow_registered_model_name = mlflow_registered_model_name_template.format(station=station)
        model_uri = f"models:/{mlflow_registered_model_name}/Production"
        try:
            MODELS[station] = mlflow.pyfunc.load_model(model_uri)
            print(f"Loaded model for {station}")
        except Exception as e:
            print(f"Failed to load model for {station}: {e}")

@app.get("/")
def root():
    return {"message": "Weather station temperature forecaster"}

@app.post("/predict/{station}")
def predict(station: str, input_data: dict):
    station = station.upper()
    if station not in MODELS:
        raise HTTPException(status_code=404, detail="Station model not found")
    try:
        model = MODELS[station]
        prediction = model.predict(input_data)
        return {"station": station, "prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))