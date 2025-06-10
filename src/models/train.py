import os
import yaml
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from model import EarlyStopping, TemperatureForecaster
from preprocess import load_data, preprocess_data
matplotlib.use("Agg")  # use non-GUI backend suitable for saving figures

import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error


def save_model(model, model_full_path):
    # ensure the directory exists
    os.makedirs(os.path.dirname(model_full_path), exist_ok=True)

    # save trained model
    torch.save(model.state_dict(), model_full_path)
    print(f"Trained model saved to {model_full_path}")


def register_model(model, model_full_path, mlflow_registered_model_name_template, station, sequence_length, input_size, outputs, device):
    # log model path
    mlflow.log_artifact(model_full_path)

    input_example = torch.randn(1, sequence_length, input_size).to(device).cpu().numpy()
    output_example = outputs.cpu().numpy()
    registered_model_name = mlflow_registered_model_name_template.format(station=station)
    model_signature = infer_signature(model_input=input_example, model_output=output_example)

    # log and register model to MLflow
    mlflow.pytorch.log_model(
        model,
        artifact_path="model",
        registered_model_name=registered_model_name,
        input_example=input_example,
        signature=model_signature
    )


def calculate_eval_metrics(actuals, predictions):
    # calculate evaluation metrics
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)

    print(f"\nEvaluation Results:")
    print(f"\tMSE:  {mse:.4f}")
    print(f"\tMAE:  {mae:.4f}")
    print(f"\tRMSE: {rmse:.4f}\n")

    # log evaluation metrics using MLflow
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_mse", mse)
    mlflow.log_metric("test_rmse", rmse)


def train_model(station, X, train_loader, val_loader, hidden_size, num_layers, dropout, lr, patience, min_delta, epochs, model_full_path, mlflow_registered_model_name_template):
    sequence_length = X.shape[1]
    input_size = X.shape[2]  # number of features
    model = TemperatureForecaster(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

    # enable MLflow autologging to PyTorch
    mlflow.pytorch.autolog()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta) 

    for epoch in range(epochs):
        # training step
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataset)

        # validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # check early stopping condition
        if early_stopping.step(val_loss, model):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # restore best model weights
    early_stopping.restore_best_weights(model)

    # save and register trained model
    save_model(model, model_full_path)
    register_model(model, model_full_path, mlflow_registered_model_name_template, station, sequence_length, input_size, outputs, device)

    return model


def evaluate_full_dataset(model, X, y, batch_size):
    full_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_X, batch_y in full_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.append(outputs.cpu().numpy())
            actuals.append(batch_y.numpy())

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    # calculate full dataset metrics
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)

    print(f"\nFull Dataset Results:")
    print(f"\tMSE:  {mse:.4f}")
    print(f"\tMAE:  {mae:.4f}")
    print(f"\tRMSE: {rmse:.4f}\n")

    # log full dataset metrics using MLflow
    mlflow.log_metric("full_mse", mse)
    mlflow.log_metric("full_mae", mae)
    mlflow.log_metric("full_rmse", rmse)

    return predictions, actuals


def evaluate_model(X, test_loader, model_full_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading trained model from {model_full_path}...")

    model = TemperatureForecaster(input_size=X.shape[2])
    model.load_state_dict(torch.load(model_full_path))
    model.to(device)
    model.eval()

    # evaluation
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.append(outputs.cpu().numpy())
            actuals.append(batch_y.numpy())

    # concatenate all batches
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    calculate_eval_metrics(actuals, predictions)

    return predictions, actuals


def print_results(predictions, actuals, temp_scaler, df, lookback, X_test, station, plot_full_path):
    predictions_2d = predictions.reshape(-1, 1)
    actuals_2d = actuals.reshape(-1, 1)

    n_samples, forecast_horizon = predictions.shape

    # apply inverse transform
    predictions_rescaled = temp_scaler.inverse_transform(predictions_2d).reshape(n_samples, forecast_horizon)
    actuals_rescaled = temp_scaler.inverse_transform(actuals_2d).reshape(n_samples, forecast_horizon)

    forecast_horizon = 6
    time_step_minutes = 30  # since 6 steps correspond to 3 hours (6 * 30 mins)

    # create a continuous time axis for all predictions by expanding each start date + horizon steps
    all_times = []
    all_preds = []
    all_actuals = []

    # total samples in your dataset (X.shape[0])
    total_samples = len(df) - lookback - forecast_horizon + 1
    # all start dates for samples
    all_start_dates = df['Date'].iloc[:total_samples].reset_index(drop=True)
    # get test start dates
    test_dates = all_start_dates.iloc[-len(X_test):].reset_index(drop=True)

    for i, start_time in enumerate(test_dates):
        for step in range(forecast_horizon):
            all_times.append(start_time + pd.Timedelta(minutes=time_step_minutes * (step + 1)))
            all_preds.append(predictions_rescaled[i, step])
            all_actuals.append(actuals_rescaled[i, step])

    all_times = pd.to_datetime(all_times)
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    plt.figure(figsize=(15, 6))
    plt.plot(all_times, all_actuals, label='Actual')
    plt.plot(all_times, all_preds, label='Predicted', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title(f'Temperature Predictions and Actuals (unfolded in time) - {station}\n{timestamp}')
    plt.legend()
    plt.grid(True)
    # plt.show()

    # ensure the directory exists
    os.makedirs(os.path.dirname(plot_full_path), exist_ok=True)

    # save the figure instead of displaying it
    plt.savefig(plot_full_path, bbox_inches='tight')
    plt.close()


def train_model_on_temperature_data():
    params_train = yaml.safe_load(open("params.yaml"))["train"]
    params_preprocess = yaml.safe_load(open("params.yaml"))["preprocess"]
    stations = yaml.safe_load(open("params.yaml"))["stations"]

    output_file_path_template = params_preprocess["output_file_path_template"]

    target_column = params_train["target_column"]
    columns_to_drop = params_train["columns_to_drop"]
    temp_scaler_name = params_train["temp_scaler_name"]
    other_scaler_name = params_train["other_scaler_name"]
    model_path_template = params_train["model_path_template"]
    model_name_template = params_train["model_name_template"]
    plot_name_template = params_train["plot_name_template"]

    lookback = params_train["lookback"]
    forecast_horizon = params_train["forecast_horizon"]
    test_size = params_train["test_size"]
    val_size = params_train["val_size"]
    
    batch_size = params_train["batch_size"]
    hidden_size = params_train["hidden_size"]
    num_layers = params_train["num_layers"]
    dropout = params_train["dropout"]
    lr = params_train["learning_rate"]
    patience = params_train["patience"]
    min_delta = params_train["min_delta"]
    epochs = params_train["epochs"]

    mlflow_uri = params_train["mlflow_uri"]
    mlflow_experiment_name = params_train["mlflow_experiment_name"]
    mlflow_experiment_run_name_template = params_train["mlflow_experiment_run_name"]
    mlflow_registered_model_name_template = params_train["mlflow_registered_model_name"]

    mlflow.set_tracking_uri(uri=mlflow_uri)
    print("uri: ", mlflow_uri)
    mlflow.set_experiment(experiment_name=mlflow_experiment_name)

    for station in stations:
        # start MLflow run for each weather station
        mlflow_experiment_run_name = mlflow_experiment_run_name_template.format(station=station)
        print("mlflow experiment run name: ", mlflow_experiment_run_name)
        with mlflow.start_run(run_name=mlflow_experiment_run_name):
            mlflow.log_param("station", station)
            mlflow.log_param("target_column", target_column)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("val_size", val_size)
            mlflow.log_param("lookback", lookback)
            mlflow.log_param("forecast_horizon", forecast_horizon)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("dropout", dropout)
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("patience", patience)
            mlflow.log_param("epochs", epochs)

            df = load_data(station, output_file_path_template)
            train_loader, val_loader, test_loader, X, y, X_test, temp_scaler = preprocess_data(df, columns_to_drop, temp_scaler_name, other_scaler_name, lookback, forecast_horizon, batch_size, target_column, test_size, val_size)
            
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            model_name = model_name_template.format(station=station, timestamp=timestamp)
            model_path = model_path_template.format(station=station)
            model_full_path = os.path.join(model_path, model_name)

            plot_name = plot_name_template.format(station=station, timestamp=timestamp)
            plot_full_path = os.path.join(model_path, plot_name)

            print(f"\nTraining model for station: {station}\n")
            model = train_model(station, X, train_loader, val_loader, hidden_size, num_layers, dropout, lr, patience, min_delta, epochs, model_full_path, mlflow_registered_model_name_template)

            print(f"\nEvaluating model for station: {station}\n")
            predictions, actuals = evaluate_model(X, test_loader, model_full_path)
            print_results(predictions, actuals, temp_scaler, df, lookback, X_test, station, plot_full_path)

            print(f"\nEvaluating model for full dataset for station: {station}\n")
            # predictions_full, actuals_full = evaluate_full_dataset(model, X, y, batch_size)



if __name__ == "__main__":
    train_model_on_temperature_data()