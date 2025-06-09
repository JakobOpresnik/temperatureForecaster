from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import yaml
import logging

app = FastAPI()

stations = yaml.safe_load(open("../params.yaml"))["stations"]
params_train = yaml.safe_load(open("../params.yaml"))["train"]

mlflow_client = MlflowClient()
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
    
@app.get("/models")
def list_models():
    try:
        model_names = [
            mlflow_registered_model_name_template.format(station=station)
            for station in stations
        ]
        print(model_names)
        model_infos = []
        for name in model_names:
            try:
                latest_versions = mlflow_client.get_latest_versions(name)
                print(latest_versions)
                model_infos.append({
                    "name": name,
                    "latest_versions": [
                        {
                            "version": v.version,
                            "stage": v.current_stage,
                            "status": v.status,
                            "run_id": v.run_id
                        }
                        for v in latest_versions
                    ]
                })
            except Exception as e:
                model_infos.append({
                    "name": name,
                    "error": str(e)
                })
        return {"models": model_infos}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_name}")
def get_model(model_name: str):
    try:
        latest_versions = mlflow_client.get_latest_versions(model_name)
        return {
            "name": model_name,
            "latest_versions": [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "status": v.status,
                    "run_id": v.run_id
                }
                for v in latest_versions
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found or error: {str(e)}")