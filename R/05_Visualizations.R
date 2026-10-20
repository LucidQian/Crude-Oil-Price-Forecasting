# ==========================================
# PROJECT VISUALIZATIONS
# ==========================================

# Run from the project root directory

dir.create("figures", showWarnings = FALSE)

# Load datasets
df <- read.csv("data/data.csv")
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

arimax <- read.csv("results/arimax_performance.csv")
rw <- read.csv("results/random_walk_performance.csv")

# Forecast horizons
horizons <- c(1, 5, 20, 60)

# Match results by horizon
arimax <- arimax[match(horizons, arimax$Horizon), ]
rw <- rw[match(horizons, rw$Horizon), ]

stopifnot(
  !anyNA(arimax$MAE),
  !anyNA(rw$MAE)
)

# MAE comparison matrix
mae_values <- rbind(
  ARIMAX = arimax$MAE,
  "Random Walk" = rw$MAE
)

# Create output folder
dir.create("figures", showWarnings = FALSE)

# Save figure
png(
  "figures/mae_comparison.png",
  width = 1600,
  height = 950,
  res = 160
)

par(mar = c(5, 5, 4, 2))

# Grouped bar chart
bp <- barplot(
  mae_values,
  beside = TRUE,
  col = c("steelblue", "tomato"),
  names.arg = paste(horizons, "Days"),
  ylim = c(0, max(mae_values) * 1.2),
  main = "ARIMAX vs Random Walk: MAE Comparison",
  xlab = "Forecast Horizon (Trading Days)",
  ylab = "Mean Absolute Error",
  border = NA,
  cex.main = 1.3,
  cex.lab = 1.1
)

# Add MAE values above bars
text(
  x = bp,
  y = mae_values,
  labels = sprintf("%.4f", mae_values),
  pos = 3,
  cex = 0.8
)

# Legend
legend(
  "topleft",
  legend = c("ARIMAX", "Random Walk"),
  fill = c("steelblue", "tomato"),
  bty = "n"
)

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

# =====================================
# FIGURE 4: RMSE COMPARISON
# =====================================

# Load backtesting results
arimax <- read.csv("results/arimax_performance.csv")
rw <- read.csv("results/random_walk_performance.csv")

# Forecast horizons
horizons <- c(1, 5, 20, 60)

# Match results by horizon
arimax <- arimax[match(horizons, arimax$Horizon), ]
rw <- rw[match(horizons, rw$Horizon), ]

stopifnot(
  !anyNA(arimax$RMSE),
  !anyNA(rw$RMSE)
)

# RMSE comparison matrix
rmse_values <- rbind(
  ARIMAX = arimax$RMSE,
  "Random Walk" = rw$RMSE
)

# Create output folder
dir.create("figures", showWarnings = FALSE)

# Save figure
png(
  "figures/rmse_comparison.png",
  width = 1600,
  height = 950,
  res = 160
)

par(mar = c(5, 5, 4, 2))

# Grouped bar chart
bp <- barplot(
  rmse_values,
  beside = TRUE,
  col = c("steelblue", "tomato"),
  names.arg = paste(horizons, "Days"),
  ylim = c(0, max(rmse_values) * 1.2),
  main = "ARIMAX vs Random Walk: RMSE Comparison",
  xlab = "Forecast Horizon (Trading Days)",
  ylab = "Root Mean Squared Error",
  border = NA,
  cex.main = 1.3,
  cex.lab = 1.1
)

# Add RMSE values above bars
text(
  x = bp,
  y = rmse_values,
  labels = sprintf("%.4f", rmse_values),
  pos = 3,
  cex = 0.8
)

# Legend
legend(
  "topleft",
  legend = c("ARIMAX", "Random Walk"),
  fill = c("steelblue", "tomato"),
  bty = "n"
)

dev.off()
