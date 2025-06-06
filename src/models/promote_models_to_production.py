import yaml
import mlflow

def promote_latest_models_to_production():
    stations = yaml.safe_load(open("params.yaml"))["stations"]
    params_train = yaml.safe_load(open("params.yaml"))["train"]

    mlflow_registered_model_name_template = params_train["mlflow_registered_model_name"]
    mlflow_uri = params_train["mlflow_uri"]

    model_names = [mlflow_registered_model_name_template.format(station=station) for station in stations]

    mlflow.set_tracking_uri(uri=mlflow_uri)
    client = mlflow.tracking.MlflowClient()

    for model_name in model_names:
        # get all existing versions of the model from MLflow
        versions = client.get_latest_versions(name=model_name)
        
        # find the latest version by version number
        latest_version = max(int(v.version) for v in versions)
        
        print(f"Promoting model {model_name} version {latest_version} to Production!")
        
        # add "Production" alias to latest model version
        client.set_registered_model_alias(
            name=model_name,
            version=str(latest_version),
            alias="Production"
        )

if __name__ == "__main__":
    promote_latest_models_to_production()
