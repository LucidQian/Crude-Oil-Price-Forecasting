# Brent Crude Oil Price Forecasting

**Multi-horizon forecasting with ARIMAX, a Multilayer Perceptron (MLP), and a Random Walk benchmark**

This project explores Brent crude oil price forecasting using classical time-series econometrics and a neural-network model. It combines data preparation, feature engineering, model estimation, out-of-sample evaluation, saved prediction results, and 60-step forecasting workflows.

> **Scope note:** This is a research/coursework repository, not a trading system. Forecasts are uncertain and are not investment advice.

## Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Data](#data)
- [Methods](#methods)
- [Backtesting and Metrics](#backtesting-and-metrics)
- [Reported Results](#reported-results)
- [Setup](#setup)
- [Reproduce the Analysis](#reproduce-the-analysis)
- [Forecast Output](#forecast-output)
- [Important Evaluation Notes](#important-evaluation-notes)
- [Limitations and Further Work](#limitations-and-further-work)
- [Authentication and Credentials](#authentication-and-credentials)
- [Disclaimer](#disclaimer)

## Project Overview

The objective is to assess whether information from related energy markets and engineered price features can help forecast Brent crude oil prices across different horizons.

The repository currently contains three approaches:

1. **ARIMAX** — a time-series model for Brent daily price changes with exogenous market variables and autoregressive errors.
2. **MLP** — a feed-forward neural network using engineered features derived from Brent, WTI, and XLE.
3. **Random Walk** — a no-drift benchmark whose forecast equals the latest observed Brent price.

The saved workflows evaluate 1-, 5-, 20-, and 60-trading-day horizons and include separate scripts to produce 60-step forecasts.

## Repository Structure

```text
Crude-Oil-Price-Forecasting/
├── data/
│   ├── data.csv                         # Aligned historical series
│   ├── train.csv                        # Chronological training split
│   └── test.csv                         # Chronological testing split
├── figures/
│   ├── historical_price_comparison.png
│   ├── historical_price_change_comparison.png
│   ├── model_performance_comparison.png
│   └── future_60day_model_comparison.png
├── forecasting/
│   ├── arimax_forecast60.R              # ARIMAX 60-step price forecast
│   └── mlp_forecast60.py                # MLP 60-horizon forecasts
├── models/
│   ├── ARIMAX_model.R                   # ARIMAX estimation and rolling backtest
│   ├── Random_Walk.R                    # Random Walk rolling benchmark
│   └── mlp_model.py                     # MLP model, feature engineering, tuning and backtest
├── preprocessing/
│   └── Data_Preparation.R               # Download, align, and split data
├── results/
│   ├── arimax_backtest_h1.csv
│   ├── arimax_backtest_h5.csv
│   ├── arimax_backtest_h20.csv
│   ├── arimax_backtest_h60.csv
│   ├── arimax_performance.csv
│   ├── mlp_backtest_h1.csv
│   ├── mlp_backtest_h5.csv
│   ├── mlp_backtest_h20.csv
│   ├── mlp_backtest_h60.csv
│   ├── mlp_performance.csv
│   ├── random_walk_performance.csv
│   ├── arimax_forecast60.csv
│   └── mlp_forecast60.csv
├── visualization/
│   ├── historical_comparison.ipynb
│   └── forecast_60.ipynb
└── requirements.txt                    # Python dependencies
```

The `figures/` images and `results/` CSV files are saved project outputs. The notebooks under `visualization/` are intended for plotting and comparing historical data, model performance, and future forecasts.

## Data

The R data-preparation script downloads closing prices from Yahoo Finance via `quantmod` and aligns the series by date.

| Series | Yahoo Finance symbol | Use |
|---|---|---|
| Brent crude oil futures | `BZ=F` | Forecast target |
| WTI crude oil futures | `CL=F` | Related crude-oil market feature |
| Energy Select Sector SPDR Fund | `XLE` | Energy-sector market feature |

The data-preparation script currently requests data from **1 September 2007 through 17 October 2026**. The actual last observation depends on data-provider availability and the download date.

Observations are aligned on common dates, sorted chronologically, and split into 70% training data and 30% testing data. The generated CSV files have the following roles:

- `data/data.csv`: aligned full sample.
- `data/train.csv`: earlier observations used for model development.
- `data/test.csv`: later observations used for out-of-sample evaluation.

The CSV files are already included, so running the existing modeling scripts does not require downloading the data again.

## Methods

### 1. ARIMAX

The main ARIMAX workflow models daily changes in Brent prices using:

- Daily changes in WTI prices.
- Daily changes in XLE.
- A lagged 60-trading-day Brent price change.
- AR(3) dynamics in the model's error component.

The current specification uses an ARIMAX(3,0,0) form for the Brent-change target. Auxiliary AR models forecast WTI changes with AR(2) and XLE changes with AR(1). These forecasts are used as exogenous inputs in the recursive 60-step Brent forecast.

Relevant files:

- `models/ARIMAX_model.R`
- `forecasting/arimax_forecast60.R`

### 2. Multilayer Perceptron (MLP)

The Python implementation uses a feed-forward neural network:

```text
Input features
      |
Linear(n_features, hidden)
      |
ReLU
      |
Linear(hidden, 16)
      |
ReLU
      |
Linear(16, 1)
      |
Predicted Brent price change
```

The feature-engineering pipeline uses Brent, WTI, and XLE price levels to construct daily changes, lagged changes (1, 2, 5, 10, and 20 periods), moving averages, rolling volatility measures, momentum features, and the Brent–WTI spread.

The backtest script selects 30 features using `SelectKBest(f_regression)`, searches hidden-layer size and learning-rate combinations using `TimeSeriesSplit`, and evaluates 1-, 5-, 20-, and 60-day targets. The standalone future-forecast script fixes the MLP configuration at `hidden=32`, `lr=0.0001`, and 200 epochs, and writes a 60-row forecast table.

Relevant files:

- `models/mlp_model.py`
- `forecasting/mlp_forecast60.py`

### 3. Random Walk benchmark

The benchmark uses a no-drift ARIMA(0,1,0) model. At every forecast origin, all future price forecasts equal the last observed Brent price:

```text
Forecast(t + h | t) = Brent price at t
```

Relevant file: `models/Random_Walk.R`.

## Backtesting and Metrics

The ARIMAX and Random Walk R scripts use the following rolling-origin settings:

| Setting | Value |
|---|---:|
| Forecast horizons | 1, 5, 20, 60 trading days |
| Rolling estimation window | 1,000 observations |
| Forecast-origin step | 10 observations |
| Parameter re-estimation | Every 20 observations |
| Evaluation data | Chronological test set |

At each origin, only history available up to that point is used to generate a forecast. The scripts store actual and predicted prices for each horizon in separate CSV files.

The repository reports four metrics:

- **MAE (Mean Absolute Error):** average absolute difference between predicted and actual prices. Lower is better.
- **RMSE (Root Mean Squared Error):** square root of the average squared prediction error. Lower is better and larger errors receive greater weight.
- **MAPE (Mean Absolute Percentage Error):** average absolute percentage error. Lower is better, provided actual values are non-zero.
- **DA (Directional Accuracy):** percentage of forecast origins for which the predicted direction matches the realized direction over the selected horizon. Higher is better than a lower DA, although results near 50% indicate limited directional signal.

The Random Walk has `DA = NA` in the saved benchmark table because its predicted price change is exactly zero, so its directional prediction is undefined under the implementation's convention.

## Reported Results

The metrics below are read from the currently committed CSV result files. They are presented as stored outputs, not as a guarantee that rerunning code in a different environment will reproduce them exactly.

### RMSE comparison

| Horizon | ARIMAX RMSE | MLP RMSE | Random Walk RMSE | Lowest stored RMSE |
|---:|---:|---:|---:|---|
| 1 day | 2.7611 | 2.8095 | 2.8058 | ARIMAX |
| 5 days | 4.6059 | 4.5533 | 4.6008 | MLP |
| 20 days | 8.5990 | 8.7076 | 8.6696 | ARIMAX |
| 60 days | 14.2787 | 13.7497 | 14.1581 | MLP |

### MLP directional accuracy

| Horizon | MLP DA |
|---:|---:|
| 1 day | 53.28% |
| 5 days | 57.66% |
| 20 days | 51.82% |
| 60 days | 67.15% |

In the currently saved results, ARIMAX has the lowest RMSE at the 1-day and 20-day horizons, while MLP has the lowest RMSE at the 5-day and 60-day horizons. The MLP's stored directional accuracy is highest at the 60-day horizon. These are empirical results on the project's existing evaluation outputs; they do not establish that one model will remain superior in future markets.

For the model-by-model metric tables, see:

- [ARIMAX performance](results/arimax_performance.csv)
- [MLP performance](results/mlp_performance.csv)
- [Random Walk performance](results/random_walk_performance.csv)

## Setup

### R environment

Install R, then install the packages used by the scripts:

```r
install.packages(c("quantmod", "fBasics", "forecast", "tseries"))
```

The data-preparation script uses Yahoo Finance connectivity. If you only want to run the modeling scripts on the CSV files already included in the repository, you can skip data downloading.

### Python environment

Use Python 3.10 or a compatible recent Python version with the following packages:

```bash
python -m pip install -r requirements.txt
```

A virtual environment is recommended to keep project dependencies separate from other Python projects.

## Reproduce the Analysis

Run commands from the **repository root** so the relative R paths such as `data/train.csv` and `results/` resolve correctly.

### Step 1 — Prepare or refresh the data (optional)

In R, set the working directory to the repository's `data/` folder and source the script using its relative path:

```r
setwd("/path/to/Crude-Oil-Price-Forecasting/data")
source("../preprocessing/Data_Preparation.R")
```

Replace `/path/to/Crude-Oil-Price-Forecasting` with your local repository path. This downloads the three market series, aligns common dates, and creates `data.csv`, `train.csv`, and `test.csv` in the `data/` folder. The committed data files can be used without this step.

### Step 2 — Run ARIMAX rolling backtesting

In R:

```r
source("models/ARIMAX_model.R")
```

This estimates the model, performs rolling-origin evaluation for the four horizons, and writes the ARIMAX performance and prediction CSV files to `results/`.

### Step 3 — Run the Random Walk benchmark

In R:

```r
source("models/Random_Walk.R")
```

This writes `results/random_walk_performance.csv`.

### Step 4 — Run the MLP backtest and hyperparameter search

From the repository root:

```bash
python models/mlp_model.py
```

This trains the MLP configurations, performs time-series cross-validation for hyperparameter selection, and writes the MLP metric and backtest CSV files to `results/`. Neural-network training may take longer than the R scripts, depending on hardware.

### Step 5 — Generate the ARIMAX 60-step forecast

In R:

```r
source("forecasting/arimax_forecast60.R")
```

This reads `data/data.csv` and writes `results/arimax_forecast60.csv`.

### Step 6 — Generate the MLP 60-horizon forecasts

From the repository root:

```bash
python forecasting/mlp_forecast60.py
```

This writes `results/mlp_forecast60.csv`. It fits a separate direct-horizon MLP for each forecast horizon from 1 through 60, using the fixed configuration documented above.

### Step 7 — Inspect visualizations

Open the notebooks in `visualization/` with Jupyter Notebook or JupyterLab:

```bash
python -m pip install jupyter
jupyter notebook
```

Use `visualization/historical_comparison.ipynb` for historical comparisons and model evaluation plots, and `visualization/forecast_60.ipynb` for future forecast visualization. Existing PNGs are available in `figures/`.

## Forecast Output

| Output | Description |
|---|---|
| `results/arimax_forecast60.csv` | 60 ARIMAX forecast values |
| `results/mlp_forecast60.csv` | 60 MLP forecast values |
| `results/arimax_backtest_h{1,5,20,60}.csv` | ARIMAX origin date, target date, origin price, actual price, and prediction |
| `results/mlp_backtest_h{1,5,20,60}.csv` | MLP forecast-origin prices and predictions |
| `results/*_performance.csv` | Model-level metrics by horizon |

The forecast scripts currently export the horizon number and predicted price, rather than a calendar date column. The MLP forecasting code calculates a set of future business dates internally but does not include those dates in the saved CSV. Therefore, interpret the output `Horizon` as a forecast step, not a guaranteed exchange trading date.

## Limitations and Further Work

Possible next steps include:

- Refit feature selection and scaling inside each time-series cross-validation fold.
- Standardize forecast origins, information sets, rolling windows, and tuning protocols across models before drawing formal performance conclusions.
- Add confidence or prediction intervals and evaluate forecast calibration.
- Evaluate performance across different oil-market regimes.
- Add transaction costs and a clearly specified trading rule only if the project is extended from price forecasting to strategy testing.

## Authentication and Credentials

This repository is a local data-analysis project. The inspected source files do not implement application login, user sessions, API-token exchange, OAuth, or a model-serving authentication layer. GitHub authentication is handled by GitHub itself when a contributor clones or pushes to the repository; it is not part of the forecasting model's request flow.

The data-preparation script retrieves market data from Yahoo Finance through `quantmod` and does not define an API token in the project source. Do not put personal access tokens, passwords, or other secrets into the repository. If an authenticated external data provider is added later, credentials should be stored outside source control (for example, in environment variables or a local ignored configuration file).

## Disclaimer

This project is for educational and quantitative research purposes only. It is not investment advice, and model outputs should not be used as the sole basis for investment decisions.
