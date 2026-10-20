# ==========================================
# PROJECT VISUALIZATIONS
# ==========================================

# Run from the project root directory

dir.create("figures", showWarnings = FALSE)

# Load datasets
df <- read.csv("C:/Users/qkq/Desktop/oil_price_forecast/data/data.csv")
arimax <- read.csv("results/arimax_performance.csv")
rw <- read.csv("results/random_walk_performance.csv")

stopifnot(
  identical(arimax$Horizon, rw$Horizon)
)

# ==========================================
# FIGURE 1: HISTORICAL PRICE TRENDS
# ==========================================

df$Date <- as.Date(df$Date)

png("figures/historical_prices.png",
    width = 1400, height = 850, res = 150)

par(mar = c(5, 5, 4, 2))

plot(df$Date, df$brent_Price,
     type = "l",
     col = "steelblue",
     lwd = 2,
     ylim = range(c(df$brent_Price,
                    df$wti_Price,
                    df$XLE)),
     xlab = "Date",
     ylab = "Price (USD)",
     main = "Historical Energy Market Prices")

lines(df$Date, df$wti_Price,
      col = "tomato", lwd = 1.5)

lines(df$Date, df$XLE,
      col = "darkgreen", lwd = 1.5)

legend("topright",
       legend = c("Brent", "WTI", "XLE"),
       col = c("steelblue", "tomato",
               "darkgreen"),
       lty = 1, lwd = 2,
       bty = "n")

grid()
dev.off()


# ==========================================
# FIGURE 2: MAE COMPARISON
# ==========================================

mae_values <- rbind(arimax$MAE, rw$MAE)

png("figures/mae_comparison.png",
    width = 1400, height = 850, res = 150)

par(mar = c(5, 5, 4, 2))

barplot(mae_values,
        beside = TRUE,
        names.arg = arimax$Horizon,
        col = c("steelblue", "tomato"),
        ylim = c(0, max(mae_values) * 1.2),
        main = "ARIMAX vs Random Walk: MAE",
        xlab = "Forecast Horizon (Trading Days)",
        ylab = "Mean Absolute Error")

legend("topleft",
       legend = c("ARIMAX", "Random Walk"),
       fill = c("steelblue", "tomato"),
       bty = "n")

dev.off()


# ==========================================
# FIGURE 3: RELATIVE RMSE IMPROVEMENT
# ==========================================

improvement <- (rw$RMSE - arimax$RMSE) /
  rw$RMSE * 100

png("figures/rmse_improvement.png",
    width = 1400, height = 850, res = 150)

par(mar = c(5, 5, 4, 2))

barplot(
  improvement,
  names.arg = arimax$Horizon,
  col = ifelse(improvement > 0,
               "steelblue", "tomato"),
  ylim = range(c(improvement, 0)) * 1.5,
  main = "ARIMAX Relative RMSE Improvement",
  xlab = "Forecast Horizon (Trading Days)",
  ylab = "RMSE Improvement (%)"
)

abline(h = 0, lty = 2, lwd = 1.5)

legend("topright",
       legend = c("ARIMAX Better",
                  "Random Walk Better"),
       fill = c("steelblue", "tomato"),
       bty = "n")

cat("First three figures generated successfully!\n")

dev.off()
