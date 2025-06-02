import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from model import TimeSeriesDataset


def load_data(station, output_file_path_template):
    data_path = output_file_path_template.format(station=station)
    df = pd.read_csv(data_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def preprocess_data(df, columns_to_drop, temp_scaler_name, other_scaler_name, lookback, forecast_horizon, batch_size, target_column, test_size, val_size):
    
    print("\nPreprocessing data...\n")
    
    df = df.drop(columns=columns_to_drop)
    print(df.head(30))

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

    temp_scaler = MinMaxScaler()
    other_scaler = MinMaxScaler()

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

    # save scaler for possible use of inverse-transform later on
    joblib.dump(temp_scaler, temp_scaler_name)
    joblib.dump(other_scaler, other_scaler_name)


    # create data sequences
    X, y = create_sequences(scaled_df, target_col=target_column, lookback=lookback, forecast_horizon=forecast_horizon)

    # split data into train, validation and test datasets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, shuffle=False)

    print(f"rows in df: {len(scaled_df)}")
    print(f"all samples: {len(X)}")
    print(f"train samples: {len(X_train)}")
    print(f"validation samples: {len(X_val)}")
    print(f"test samples: {len(X_test)}")

    # assuming X shape is (samples, timesteps, features)
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, X, X_test, temp_scaler


def create_sequences(data, target_col="Temperature", lookback=12, forecast_horizon=6):

    print("\nCreating sequences...\n")

    X, y = [], []
    target_idx = data.columns.get_loc(target_col)
    for i in range(len(data) - lookback - forecast_horizon):
        seq_x = data.iloc[i:i + lookback].values
        seq_y = data.iloc[i + lookback : i + lookback + forecast_horizon, target_idx]
        X.append(seq_x)
        y.append(seq_y)
    
    return np.array(X), np.array(y)  # y shape: (samples, 6)