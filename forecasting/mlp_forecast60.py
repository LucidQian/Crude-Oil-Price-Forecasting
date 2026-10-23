from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import SelectKBest, f_regression
from models.mlp_model import train_mlp


def run_forecast():
    # 1. Read train/test split.
    train_path = ROOT / "data" / "train.csv"
    test_path = ROOT / "data" / "test.csv"
    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing data file: {path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train["Date"] = pd.to_datetime(train["Date"])
    test["Date"] = pd.to_datetime(test["Date"])
    train = train.sort_values("Date").reset_index(drop=True)
    test = test.sort_values("Date").reset_index(drop=True)
    assert train["Date"].max() < test["Date"].min(), "Train/test dates overlap or are out of order"

    # 2. Same feature engineering as the original MLP notebook.
    df_ml = pd.concat([train, test], ignore_index=True).copy()
    cols = ["brent_Price", "wti_Price", "XLE"]
    for col in cols:
        df_ml[f"{col}_diff"] = df_ml[col].diff()
        for lag in [1, 2, 5, 10, 20]:
            df_ml[f"{col}_lag{lag}"] = df_ml[col].diff().shift(lag)
        for w in [5, 20, 60]:
            df_ml[f"{col}_ma{w}"] = df_ml[col].rolling(w).mean()
            df_ml[f"{col}_vol{w}"] = df_ml[col].diff().rolling(w).std()
        for w in [5, 20, 60]:
            df_ml[f"{col}_mom{w}"] = df_ml[col] - df_ml[col].shift(w)
    df_ml["spread"] = df_ml["brent_Price"] - df_ml["wti_Price"]
    df_ml = df_ml.dropna().reset_index(drop=True)

    # 3. target and feature selection
    horizons = [1, 5, 20, 60]
    for h in horizons:
        df_ml[f"target_{h}"] = df_ml["brent_Price"].shift(-h) - df_ml["brent_Price"]
    features = [col for col in df_ml.columns if col != "Date" and not col.startswith("target_")]
    train_end = train["Date"].max()
    train_data_20 = df_ml[df_ml["Date"] <= train_end].iloc[:-20].copy()
    X_reference = train_data_20[features]
    y_reference = train_data_20["target_20"]
    selector = SelectKBest(score_func=f_regression, k=30)
    selector.fit(X_reference, y_reference)
    selected_features = X_reference.columns[selector.get_support()].tolist()
    print("Selected Features:")
    for name in selected_features:
        print(name)

    # 4. Train 60 independent direct-forecast MLPs, fixed hyperparameters.
    df_ml = df_ml.sort_values("Date").reset_index(drop=True)
    last_date = df_ml["Date"].iloc[-1]
    last_price = float(df_ml["brent_Price"].iloc[-1])
    latest = df_ml[selected_features].iloc[[-1]].to_numpy(dtype=np.float32)
    forecast_prices = []
    for h in range(1, 61):
        target = df_ml["brent_Price"].shift(-h) - df_ml["brent_Price"]
        valid = target.notna()
        X_train = df_ml.loc[valid, selected_features].to_numpy(dtype=np.float32)
        y_train = target.loc[valid].to_numpy(dtype=np.float32)
        torch.manual_seed(42)
        model, scaler, y_mean, y_std = train_mlp(
            X_train, y_train, hidden=32, lr=0.0001, epochs=200
        )
        model.eval()
        with torch.no_grad():
            X_latest = torch.tensor(scaler.transform(latest), dtype=torch.float32)
            pred_scaled = model(X_latest).item()
        forecast_prices.append(last_price + float(pred_scaled * y_std + y_mean))
        if h % 10 == 0:
            print(f"Completed {h}/60 horizons", flush=True)

    # 5. Save the forecasts
    future_dates = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=60)
    forecast_df = pd.DataFrame({
        "Horizon": np.arange(1, 61),
        "MLP": forecast_prices,
    })
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "mlp_forecast60.csv"
    forecast_df.to_csv(output_path, index=False)
    print("\nFULL 60-DAY BRENT PRICE FORECAST")
    print(forecast_df.round(4).to_string(index=False))
    print(f"\nSaved forecast CSV to: {output_path}")
    return forecast_df


if __name__ == "__main__":
    run_forecast()
