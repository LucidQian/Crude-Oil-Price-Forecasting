from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import ParameterGrid

import torch
import torch.nn as nn


# MLP model: original architecture, unchanged
class MLP(nn.Module):
    def __init__(self, n_features, hidden=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# Training function: original training behavior, unchanged
def train_mlp(X, y, hidden=32, lr=0.001, epochs=200):
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Standardize target
    y_mean = y.mean()
    y_std = max(y.std(), 1e-6)
    y_scaled = (y - y_mean) / y_std

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y_scaled, dtype=torch.float32)

    model = MLP(X.shape[1], hidden)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()

    return model, scaler, y_mean, y_std


# Reusable helper for future forecast: does not train or run backtest
def predict_change(fitted, X):
    model, scaler, y_mean, y_std = fitted
    X_tensor = torch.tensor(scaler.transform(X), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_tensor).cpu().numpy()
    return pred_scaled * y_std + y_mean


# Reusable helper for future forecast: same SelectKBest calculation
def select_features(X, y, feature_names, k=30):
    selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
    selector.fit(X, y)
    selected = list(np.asarray(feature_names)[selector.get_support()])
    return selector, selected


def run_backtest():
    # Load datasets
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")

    # Date processing
    train["Date"] = pd.to_datetime(train["Date"])
    test["Date"] = pd.to_datetime(test["Date"])

    train = train.sort_values("Date").reset_index(drop=True)
    test = test.sort_values("Date").reset_index(drop=True)

    # Combine for feature engineering
    df = pd.concat([train, test], ignore_index=True)

    print("Train:", train.shape)
    print("Test:", test.shape)
    print("WTI minimum:", df["wti_Price"].min())

    assert train["Date"].max() < test["Date"].min()

    # Feature Engineering
    df_ml = df.copy()
    cols = ["brent_Price", "wti_Price", "XLE"]

    for col in cols:
        # Daily price difference
        df_ml[f"{col}_diff"] = df_ml[col].diff()

        # Lagged differences
        for lag in [1, 2, 5, 10, 20]:
            df_ml[f"{col}_lag{lag}"] = df_ml[col].diff().shift(lag)

        # Moving average and volatility
        for w in [5, 20, 60]:
            df_ml[f"{col}_ma{w}"] = df_ml[col].rolling(w).mean()
            df_ml[f"{col}_vol{w}"] = df_ml[col].diff().rolling(w).std()

        # Momentum
        for w in [5, 20, 60]:
            df_ml[f"{col}_mom{w}"] = df_ml[col] - df_ml[col].shift(w)

    # Brent-WTI spread
    df_ml["spread"] = df_ml["brent_Price"] - df_ml["wti_Price"]

    # Remove initial missing values
    df_ml = df_ml.dropna().reset_index(drop=True)

    print("Dataset shape:", df_ml.shape)
    print("Missing values:", df_ml.isna().sum().sum())
    print("Number of features:", df_ml.shape[1] - 1)

    # Target Variables
    horizons = [1, 5, 20, 60]

    # Future Brent price changes
    for h in horizons:
        df_ml[f"target_{h}"] = (
            df_ml["brent_Price"].shift(-h)
            - df_ml["brent_Price"]
        )

    # Feature columns
    features = [
        col for col in df_ml.columns
        if col != "Date" and not col.startswith("target_")
    ]

    # Original train/test boundary
    train_end = train["Date"].max()
    test_start = test["Date"].min()

    train_data = {}
    test_data = {}

    for h in horizons:
        # Training labels must be realized before test starts
        train_data[h] = (
            df_ml[df_ml["Date"] <= train_end]
            .iloc[:-h]
            .copy()
        )

        # Testing labels must be available
        test_data[h] = (
            df_ml[df_ml["Date"] >= test_start]
            .dropna(subset=[f"target_{h}"])
            .copy()
        )

        print(
            f"{h}-day | "
            f"Train: {len(train_data[h])} | "
            f"Test: {len(test_data[h])}"
        )

    print("Number of features:", len(features))

    # Feature Selection: original fixed 20-day selection
    h = 20

    # Training data only
    X = train_data[h][features]
    y = train_data[h][f"target_{h}"]

    # Select top 30 features
    selector = SelectKBest(
        score_func=f_regression,
        k=30
    )

    selector.fit(X, y)

    # Selected features
    selected_features = X.columns[
        selector.get_support()
    ].tolist()

    print("Selected Features:")
    for feature in selected_features:
        print(feature)

    # MLP HYPERPARAMETER TUNING
    torch.manual_seed(42)
    np.random.seed(42)

    # Hyperparameter search
    param_grid = list(ParameterGrid({
        "hidden": [32, 64, 128],
        "lr": [0.0001, 0.0003, 0.0005]
    }))

    print("Parameter combinations:", len(param_grid))

    best_mlp_params = {}

    for h in horizons:
        data = train_data[h]

        X = data[selected_features].to_numpy(dtype=np.float32)
        y = data[f"target_{h}"].to_numpy(dtype=np.float32)

        tscv = TimeSeriesSplit(n_splits=3, gap=h)

        best_rmse = float("inf")
        best_param = None

        for params in param_grid:
            scores = []

            for tr_idx, val_idx in tscv.split(X):
                # Reproducible initialization
                torch.manual_seed(42)

                model, scaler, y_mean, y_std = train_mlp(
                    X[tr_idx],
                    y[tr_idx],
                    hidden=params["hidden"],
                    lr=params["lr"],
                    epochs=200
                )

                # Validation prediction
                X_val = scaler.transform(X[val_idx])
                X_val = torch.tensor(
                    X_val, dtype=torch.float32
                )

                model.eval()

                with torch.no_grad():
                    pred_scaled = model(X_val).numpy()

                pred = pred_scaled * y_std + y_mean

                rmse = np.sqrt(
                    mean_squared_error(y[val_idx], pred)
                )

                scores.append(rmse)

            avg_rmse = np.mean(scores)

            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_param = params.copy()

        best_mlp_params[h] = best_param

        print(f"\n{h}-day Forecast")
        print("Best Parameters:", best_param)
        print("CV RMSE:", round(best_rmse, 4))

    # MLP OUT-OF-SAMPLE BACKTESTING
    mlp_results = []
    mlp_predictions = {}
    mlp_models = {}

    step = 10
    max_h = max(horizons)

    # forecast origins
    origins = np.arange(
        0,
        len(test) - max_h + 1,
        step
    )

    # Forecast origin dates
    origin_dates = (
        [train["Date"].iloc[-1]]
        + [test["Date"].iloc[i - 1] for i in origins[1:]]
    )

    # Features at forecast origins
    origin_data = df_ml.set_index("Date").loc[origin_dates]

    X_test = origin_data[selected_features].to_numpy(
        dtype=np.float32
    )

    origin_prices = origin_data[
        "brent_Price"
    ].to_numpy()

    for h in horizons:
        print(f"Training MLP: {h}-day forecast")

        # Training data
        X_train = train_data[h][selected_features].to_numpy(
            dtype=np.float32
        )

        y_train = train_data[h][f"target_{h}"].to_numpy(
            dtype=np.float32
        )

        # Best hyperparameters
        params = best_mlp_params[h]

        torch.manual_seed(42)

        # Train final MLP
        model, scaler, y_mean, y_std = train_mlp(
            X_train,
            y_train,
            hidden=params["hidden"],
            lr=params["lr"],
            epochs=200
        )

        mlp_models[h] = (model, scaler, y_mean, y_std)

        # Standardize test features
        X_scaled = scaler.transform(X_test)

        X_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32
        )

        # Predict price changes
        model.eval()

        with torch.no_grad():
            predicted_scaled = model(X_tensor).numpy()

        predicted_changes = (
            predicted_scaled * y_std + y_mean
        )

        # Convert to Brent prices
        predicted_prices = (
            origin_prices + predicted_changes
        )

        # Actual future Brent prices
        actual_prices = test["brent_Price"].iloc[
            origins + h - 1
        ].to_numpy()

        # Evaluation
        error = actual_prices - predicted_prices

        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error ** 2))

        actual_change = actual_prices - origin_prices
        predicted_change = predicted_prices - origin_prices

        da = np.mean(
            np.sign(actual_change)
            == np.sign(predicted_change)
        ) * 100

        valid = actual_prices != 0

        mape = np.mean(
            np.abs(error[valid] / actual_prices[valid])
        ) * 100

        # Store metrics
        mlp_results.append({
            "Horizon": h,
            "MAE": mae,
            "RMSE": rmse,
            "DA": da,
            "MAPE": mape
        })

        # Store predictions
        mlp_predictions[h] = pd.DataFrame({
            "Origin_Date": origin_dates,
            "Origin_Price": origin_prices,
            "Actual": actual_prices,
            "Predicted": predicted_prices
        })

        print(f"{h}-day forecast completed!\n")

    # Final performance
    performance_mlp = pd.DataFrame(mlp_results)

    print("Forecast origins:", len(origins))

    print("\nMLP OUT-OF-SAMPLE PERFORMANCE")
    print(performance_mlp.round(4))

    # Export original backtest results (does not alter predictions or metrics)
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    performance_mlp.to_csv(results_dir / "mlp_performance.csv", index=False)
    for horizon, prediction_table in mlp_predictions.items():
        prediction_table.to_csv(results_dir / f"mlp_backtest_h{horizon}.csv", index=False)
    print("Saved backtest CSV files to:", results_dir)


if __name__ == "__main__":
    run_backtest()
