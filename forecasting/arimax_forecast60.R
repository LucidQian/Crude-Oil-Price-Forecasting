library(stats)
library(forecast)
df <- read.csv("data/data.csv",header = T)
tail(df)

d1 = diff(df$brent_Price)
d2 = diff(df$wti_Price)
d3 = diff(df$XLE)

# ====================================
# Forecast WTI for 60 days
# ====================================
plot(d2,type='l')
acf(d2)
pacf(d2)

wti_model <- arima(
  d2,
  order = c(2,0,0),
  include.mean = F
)
wti_model

Box.test(wti_model$residuals ,fitdf=2, type='Ljung', lag=10)

wti_60 <- predict(
  wti_model,
  n.ahead = 60
)$pred

# ====================================
# Forecast XLE for 60 days
# ====================================
plot(d3,type='l')
acf(d3)
pacf(d3)

xle_model <- arima(
  d3,
  order = c(1,0,0),
  include.mean = F
)
xle_model

Box.test(xle_model$residuals ,fitdf=1, type='Ljung', lag=10)

xle_60 <- predict(
  xle_model,
  n.ahead = 60
)$pred

# =====================================
# ARIMAX FINAL 60-DAY FORECAST
# =====================================
h <- 60

brent <- as.numeric(df$brent_Price)
n <- length(brent)

# Construct ARIMAX variables
t <- 62:n

y <- d1[t - 1]

x <- cbind(
  WTI = d2[t - 1],
  XLE = d3[t - 1],
  Brent_60_Lag1 = brent[t - 1] - brent[t - 61]
)

# Fit ARIMAX(3,0,0)
brent_model <- arima(
  y,
  order = c(3,0,0),
  xreg = x,
  include.mean = FALSE
)

print(brent_model)

# Extract coefficients
cf <- coef(brent_model)

beta_w <- unname(cf["WTI"])
beta_x <- unname(cf["XLE"])
gamma <- unname(cf["Brent_60_Lag1"])

# Forecast AR(3) regression errors
u <- as.numeric(
  y - x %*% c(beta_w, beta_x, gamma)
)

error_model <- arima(
  u,
  order = c(3,0,0),
  include.mean = FALSE,
  fixed = unname(cf[c("ar1", "ar2", "ar3")]),
  transform.pars = FALSE
)

pred_u <- as.numeric(
  forecast(error_model, h = h)$mean
)

# Recursive Brent price forecast
all_prices <- c(brent, rep(NA_real_, h))

for (k in 1:h) {
  
  pos <- n + k
  
  brent_lag60 <- (
    all_prices[pos - 1] -
      all_prices[pos - 61]
  )
  
  pred_change <- (
    beta_w * wti_60[k] +
      beta_x * xle_60[k] +
      gamma * brent_lag60 +
      pred_u[k]
  )
  
  all_prices[pos] <- (
    all_prices[pos - 1] + pred_change
  )
}

brent_forecast <- tail(all_prices, h)

# Display forecast prices
forecast_results <- data.frame(
  Day = 1:h,
  Brent_Forecast = brent_forecast
)

print(round(forecast_results, 4))

# =====================================
# EXPORT ARIMAX 60-DAY FORECAST TO CSV
# =====================================

# Output directory
results_dir <- "results"

dir.create(
  results_dir,
  showWarnings = FALSE,
  recursive = TRUE
)

# Create forecast table
arimax_forecast_df <- data.frame(
  Horizon = 1:60,
  ARIMAX = as.numeric(brent_forecast)
)

# Save CSV
write.csv(
  arimax_forecast_df,
  file.path(results_dir, "arimax_forecast60.csv"),
  row.names = FALSE
)

cat("\nARIMAX 60-day forecast saved successfully!\n")
