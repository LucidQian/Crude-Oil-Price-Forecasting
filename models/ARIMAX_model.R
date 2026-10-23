library(tseries)
library(stats)
library(forecast)
#=============================
# ARIMAX MODEL
#=============================

rt <- read.csv("data/train.csv",header = T)
tail(rt)
nrow(rt)

r1 = diff(rt$brent_Price)
r2 = diff(rt$wti_Price)
r3 = diff(rt$XLE)
r4 = diff(rt$brent_Price,lag=60)

y <- r1[61:length(r1)]
x <- cbind(
  WTI = r2[61:length(r2)],
  XLE = r3[61:length(r3)],
  Brent_60_Lag1 = r4[-length(r4)]
)

m1 <- lm(y~-1+x)
summary(m1)

# =====================================
# FIT ARIMAX
# =====================================
plot(m1$residuals,type='l')
acf(m1$residuals)
pacf(m1$residuals)

m2 <- arima(
  y,
  order = c(3,0,0),
  xreg = x,
  include.mean = FALSE,
)

m2

Box.test(m2$residuals,lag=9,type='Ljung',fitdf = 3)

#Test model
# Forecast WTI
pacf(r2)
acf(r2)

wti_model <- arima(
  r2,
  order = c(2,0,0),
  include.mean = FALSE
)
wti_model

Box.test(wti_model$residuals ,fitdf=2, type='Ljung', lag=10)

# Forecast XLE
pacf(r3)
acf(r3)

xle_model <- arima(
  r3,
  order = c(1,0,0),
  include.mean = FALSE
)
xle_model

Box.test(xle_model$residuals, fitdf=1, type='Ljung', lag=10)

# =====================================
# ROLLING-WINDOW ARIMAX BACKTEST
# PERIODIC PARAMETER RE-ESTIMATION
# =====================================

rt <- read.csv("data/train.csv", header=T)
test <- read.csv("data/test.csv", header=T)

cols <- c("brent_Price", "wti_Price", "XLE")

train_data <- as.matrix(rt[, cols])
test_data <- as.matrix(test[, cols])

storage.mode(train_data) <- "double"
storage.mode(test_data) <- "double"

stopifnot(
  all(is.finite(train_data)),
  all(is.finite(test_data))
)

# SETTINGS

horizons <- c(1, 5, 20, 60)

step <- 10
window_size <- 1000
refit_every <- 20
max_h <- max(horizons)

stopifnot(refit_every %% step == 0)

origins <- seq(
  1,
  nrow(test_data) - max_h + 1,
  by = step
)

cat("Forecast origins:", length(origins), "\n")

# =====================================
# CONSTRUCT MODEL DATA
# =====================================

make_model_data <- function(window) {
  
  b <- window[, 1]
  w <- window[, 2]
  e <- window[, 3]
  
  n <- length(b)
  
  # Daily changes
  dy <- diff(b)
  dw <- diff(w)
  dx <- diff(e)
  
  # Target dates: 62, ..., n
  t <- 62:n
  
  # Lagged 60-day Brent change
  brent_lag60 <- b[t - 1] - b[t - 61]
  
  y <- dy[t - 1]
  
  x <- cbind(
    WTI = dw[t - 1],
    XLE = dx[t - 1],
    Brent_60_Lag1 = brent_lag60
  )
  
  stopifnot(length(y) == nrow(x))
  
  list(
    y = y,
    x = x,
    dw = dw,
    dx = dx
  )
}

# =====================================
# RECURSIVE BRENT FORECAST
# =====================================

forecast_brent_recursive <- function(
    model_brent,
    model_data,
    brent_history,
    pred_wti,
    pred_xle,
    h = 60
) {
  
  cf <- coef(model_brent)
  
  beta_w <- unname(cf["WTI"])
  beta_x <- unname(cf["XLE"])
  gamma <- unname(cf["Brent_60_Lag1"])
  
  phi <- unname(cf[c("ar1", "ar2", "ar3")])
  
  # Historical regression errors
  u <- as.numeric(
    model_data$y -
      model_data$x %*% c(beta_w, beta_x, gamma)
  )
  
  # Forecast AR(3) regression errors
  u_all <- c(u, rep(NA_real_, h))
  n_u <- length(u)
  
  for (k in seq_len(h)) {
    
    t <- n_u + k
    
    u_all[t] <- (
      phi[1] * u_all[t - 1] +
        phi[2] * u_all[t - 2] +
        phi[3] * u_all[t - 3]
    )
  }
  
  pred_u <- tail(u_all, h)
  
  # Recursive Brent price forecast
  b <- as.numeric(brent_history)
  n <- length(b)
  
  all_prices <- c(b, rep(NA_real_, h))
  
  for (k in seq_len(h)) {
    
    t <- n + k
    
    brent_lag60 <- (
      all_prices[t - 1] -
        all_prices[t - 61]
    )
    
    pred_change <- (
      beta_w * pred_wti[k] +
        beta_x * pred_xle[k] +
        gamma * brent_lag60 +
        pred_u[k]
    )
    
    all_prices[t] <- (
      all_prices[t - 1] + pred_change
    )
  }
  
  tail(all_prices, h)
}

# =====================================
# STORAGE
# =====================================

actual <- matrix(
  NA_real_,
  nrow = length(origins),
  ncol = length(horizons)
)

predicted <- actual
origin_price <- numeric(length(origins))

model_brent <- NULL
model_wti <- NULL
model_xle <- NULL

# =====================================
# ROLLING BACKTEST
# =====================================

for (j in seq_along(origins)) {
  
  i <- origins[j]
  
  # Available historical observations
  if (i == 1) {
    
    history <- train_data
    
  } else {
    
    history <- rbind(
      train_data,
      test_data[1:(i-1), , drop = FALSE]
    )
  }
  
  n_hist <- nrow(history)
  
  # Rolling window
  window <- tail(history, window_size)
  
  md <- make_model_data(window)
  
  # Refit every 20 days
  if ((j - 1) %% (refit_every / step) == 0) {
    
    # WTI AR(2)
    model_wti <- forecast::Arima(
      md$dw,
      order = c(2, 0, 0),
      include.mean = FALSE
    )
    
    # XLE AR(1)
    model_xle <- forecast::Arima(
      md$dx,
      order = c(1, 0, 0),
      include.mean = FALSE
    )
    
    # Brent ARIMAX(3,0,0)
    model_brent <- forecast::Arima(
      md$y,
      order = c(3, 0, 0),
      xreg = md$x,
      include.mean = FALSE
    )
    
  } else {
    
    # Update model states without re-estimation
    model_wti <- forecast::Arima(
      md$dw,
      model = model_wti
    )
    
    model_xle <- forecast::Arima(
      md$dx,
      model = model_xle
    )
    
    model_brent <- forecast::Arima(
      md$y,
      xreg = md$x,
      model = model_brent
    )
  }
  
  # Forecast WTI and XLE daily changes
  pred_wti <- as.numeric(
    forecast(model_wti, h = max_h)$mean
  )
  
  pred_xle <- as.numeric(
    forecast(model_xle, h = max_h)$mean
  )
  
  # Forecast Brent prices
  pred_price <- forecast_brent_recursive(
    model_brent = model_brent,
    model_data = md,
    brent_history = window[, 1],
    pred_wti = pred_wti,
    pred_xle = pred_xle,
    h = max_h
  )
  
  # Store actual and predicted prices
  origin_price[j] <- history[n_hist, 1]
  
  for (k in seq_along(horizons)) {
    
    h <- horizons[k]
    
    predicted[j, k] <- pred_price[h]
    actual[j, k] <- test_data[i + h - 1, 1]
  }
  
  if (j %% 10 == 0) {
    cat(
      "Completed", j, "of",
      length(origins), "origins\n"
    )
  }
}

# =====================================
# CHECK COMPLETENESS
# =====================================

stopifnot(
  all(is.finite(actual)),
  all(is.finite(predicted)),
  all(is.finite(origin_price))
)

cat("\nRolling ARIMAX Backtest Completed!\n")

# =====================================
# EVALUATION METRICS
# =====================================

performance_arimax <- data.frame(
  Horizon = horizons,
  MAE = NA_real_,
  RMSE = NA_real_,
  DA = NA_real_,
  MAPE = NA_real_
)

for (k in seq_along(horizons)) {
  
  error <- actual[, k] - predicted[, k]
  
  actual_change <- actual[, k] - origin_price
  predicted_change <- predicted[, k] - origin_price
  
  # MAE
  performance_arimax$MAE[k] <- mean(abs(error))
  
  # RMSE
  performance_arimax$RMSE[k] <- sqrt(
    mean(error^2)
  )
  
  # Directional Accuracy
  valid <- actual_change != 0
  
  performance_arimax$DA[k] <- if (any(valid)) {
    mean(
      sign(actual_change[valid]) ==
        sign(predicted_change[valid])
    ) * 100
  } else {
    NA_real_
  }
  
  # MAPE
  valid_mape <- actual[, k] != 0
  
  performance_arimax$MAPE[k] <- if (any(valid_mape)) {
    mean(
      abs(
        error[valid_mape] /
          actual[valid_mape, k]
      )
    ) * 100
  } else {
    NA_real_
  }
}

# =====================================
# FINAL RESULTS & CSV EXPORT
# =====================================

# Set output directory
results_dir <- "oil_price_forecast/results"

dir.create(
  results_dir,
  showWarnings = FALSE,
  recursive = TRUE
)

# =====================================
# 1. EXPORT PERFORMANCE METRICS
# =====================================

cat("\nARIMAX ROLLING BACKTEST PERFORMANCE\n")

print(round(performance_arimax, 4))

write.csv(
  performance_arimax,
  file.path(results_dir, "arimax_performance.csv"),
  row.names = FALSE
)

# =====================================
# 2. EXPORT BACKTEST PREDICTIONS
# =====================================

# Convert dates
rt$Date <- as.Date(rt$Date)
test$Date <- as.Date(test$Date)

# Forecast origin dates
origin_dates <- c(
  tail(rt$Date, 1),
  test$Date[origins[-1] - 1]
)

# Export predictions for each horizon
for (k in seq_along(horizons)) {
  
  h <- horizons[k]
  
  # Actual target dates
  target_dates <- test$Date[origins + h - 1]
  
  # Historical forecast results
  forecast_df <- data.frame(
    Origin_Date = origin_dates,
    Target_Date = target_dates,
    Origin_Price = origin_price,
    Actual = actual[, k],
    Predicted = predicted[, k]
  )
  
  # Save CSV
  file_path <- file.path(
    results_dir,
    paste0("arimax_backtest_h", h, ".csv")
  )
  
  write.csv(
    forecast_df,
    file_path,
    row.names = FALSE
  )
  
  cat("Saved:", file_path, "\n")
}

cat("\nAll ARIMAX backtest CSV files exported successfully!\n")
