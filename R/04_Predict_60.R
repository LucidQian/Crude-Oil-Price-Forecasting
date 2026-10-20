library(stats)
library(forecast)
df <- read.csv("C:/Users/qkq/Desktop/Oil Price Forecast/data.csv",header = T)
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
# PLOT FUTURE 60 DAYS
# =====================================
h <- 60

# ARIMAX future 60-day forecasts
arimax_60 <- as.numeric(brent_forecast)

# Random Walk
# ARIMA(0,1,0): E(P_T+h | F_T) = P_T
last_price <- tail(df$brent_Price, 1)

rw_60 <- rep(last_price, h)

stopifnot(
  length(arimax_60) == h,
  all(is.finite(arimax_60)),
  all(is.finite(rw_60))
)

# Future trading days
future_days <- 1:h

# Plot range
ylim_range <- range(arimax_60, rw_60)
padding <- max(diff(ylim_range) * 0.15, 0.5)

# ARIMAX forecast
plot(
  future_days,
  arimax_60,
  type = "l",
  col = "red",
  lwd = 2.5,
  ylim = ylim_range + c(-padding, padding),
  xlab = "Forecast Horizon (Trading Days)",
  ylab = "Brent Price",
  main = "Future 60-Day Brent Price Forecast",
  xaxt = "n"
)

axis(1, at = c(1, 10, 20, 30, 40, 50, 60))

# Random Walk forecast
lines(
  future_days,
  rw_60,
  col = "blue",
  lwd = 2.5,
  lty = 2
)

grid(col = "gray85", lty = "dotted")

legend(
  "topleft",
  legend = c(
    "ARIMAX",
    "Random Walk"
  ),
  col = c("red", "blue"),
  lty = c(1, 2),
  lwd = 2.5,
  bty = "n"
)