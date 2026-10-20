library(forecast)
library(stats)

# =====================================
# RANDOM WALK BENCHMARK
# =====================================

# Load data
rt <- read.csv(
  "C:/Users/qkq/Desktop/Oil Price Forecast/train.csv",
  header = TRUE
)

test <- read.csv(
  "C:/Users/qkq/Desktop/Oil Price Forecast/test.csv",
  header = TRUE
)

# Brent price levels
train_y <- as.numeric(rt$brent_Price)
test_y <- as.numeric(test$brent_Price)

stopifnot(
  all(is.finite(train_y)),
  all(is.finite(test_y))
)

# =====================================
# SETTINGS
# =====================================

horizons <- c(1, 5, 20, 60)

step <- 10
window_size <- 1000
refit_every <- 20

max_h <- max(horizons)

# Same forecast origins as ARIMAX
origins <- seq(
  from = 1,
  to = length(test_y) - max_h + 1,
  by = step
)

cat("Forecast origins:", length(origins), "\n")


# =====================================
# RANDOM WALK ROLLING BACKTEST
# =====================================

actual_rw <- matrix(
  NA_real_,
  nrow = length(origins),
  ncol = length(horizons)
)

predicted_rw <- actual_rw
origin_price_rw <- numeric(length(origins))

for (j in seq_along(origins)) {
  
  i <- origins[j]
  
  # -----------------------------------
  # 1. AVAILABLE HISTORICAL DATA
  # -----------------------------------
  
  if (i == 1) {
    
    history <- train_y
    
  } else {
    
    history <- c(
      train_y,
      test_y[seq_len(i - 1)]
    )
  }
  
  # -----------------------------------
  # 2. ROLLING WINDOW
  # -----------------------------------
  
  window <- tail(
    history,
    window_size
  )
  
  # -----------------------------------
  # 3. FIT RANDOM WALK
  # -----------------------------------
  
  rw_model <- forecast::Arima(
    window,
    order = c(0, 1, 0),
    include.drift = FALSE
  )
  
  # -----------------------------------
  # 4. FORECAST FUTURE 60 DAYS
  # -----------------------------------
  
  pred_price <- as.numeric(
    forecast(
      rw_model,
      h = max_h
    )$mean
  )
  
  # -----------------------------------
  # 5. STORE RESULTS
  # -----------------------------------
  
  origin_price_rw[j] <- tail(history, 1)
  
  for (k in seq_along(horizons)) {
    
    h <- horizons[k]
    
    predicted_rw[j, k] <- pred_price[h]
    
    actual_rw[j, k] <- test_y[i + h - 1]
  }
  
  if (j %% 10 == 0) {
    cat(
      "Completed",
      j,
      "of",
      length(origins),
      "origins\n"
    )
  }
}

# =====================================
# CHECK COMPLETENESS
# =====================================

stopifnot(
  all(is.finite(actual_rw)),
  all(is.finite(predicted_rw)),
  all(is.finite(origin_price_rw))
)

# Verify Random Walk property
stopifnot(
  max(abs(
    predicted_rw -
      matrix(
        origin_price_rw,
        nrow = length(origins),
        ncol = length(horizons)
      )
  )) < 1e-6
)

cat("\nRandom Walk Backtest Completed!\n")


# =====================================
# RANDOM WALK EVALUATION METRICS
# =====================================

performance_rw <- data.frame(
  Horizon = horizons,
  MAE = NA_real_,
  RMSE = NA_real_,
  DA = NA_real_,
  MAPE = NA_real_
)

for (k in seq_along(horizons)) {
  
  error <- actual_rw[, k] - predicted_rw[, k]
  
  actual_change <- actual_rw[, k] - origin_price_rw
  predicted_change <- predicted_rw[, k] - origin_price_rw
  
  # MAE
  performance_rw$MAE[k] <- mean(
    abs(error)
  )
  
  # RMSE
  performance_rw$RMSE[k] <- sqrt(
    mean(error^2)
  )
  
  # Directional Accuracy
  # No-drift RW predicts zero change
  performance_rw$DA[k] <- NA_real_
  
  # MAPE
  valid_mape <- actual_rw[, k] != 0
  
  performance_rw$MAPE[k] <- if (any(valid_mape)) {
    mean(
      abs(
        error[valid_mape] /
          actual_rw[valid_mape, k]
      )
    ) * 100
  } else {
    NA_real_
  }
}

cat("\nRANDOM WALK ROLLING BACKTEST PERFORMANCE\n")

print(round(performance_rw, 4))