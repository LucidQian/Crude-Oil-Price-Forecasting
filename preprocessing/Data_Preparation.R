#===============================================================================
# 1.DATA Preparation
#===============================================================================
library(quantmod)
library(fBasics)

#brent oil price

getSymbols("BZ=F",
           src = "yahoo",
           from = "2007-09-01",
           to = "2026-10-17")
head(`BZ=F`)
tail(`BZ=F`)

brt <- data.frame(
  Date = as.Date(index(`BZ=F`)),
  brent_Price = as.numeric(Cl(`BZ=F`))
)

basicStats(brt[,2])

# Remove missing values
brt <- na.omit(brt)

#wti oil price from 2007.9.1

getSymbols("CL=F",
           src = "yahoo",
           from = "2007-09-01",
           to = "2026-10-17")
head(`CL=F`)
tail(`CL=F`)

wti <- data.frame(
  Date = index(`CL=F`),
  wti_Price = as.numeric(Cl(`CL=F`))
)
head(wti)
tail(wti)

basicStats(wti[,2])

wti <- na.omit(wti)

#S&P 500 XLE data

getSymbols("XLE",
           src = "yahoo",
           from = "2007-09-01",
           to = "2026-10-17")

head(XLE)
tail(XLE)

xle <- data.frame(
  Date = index(XLE),
  XLE = as.numeric(Cl(XLE))
)

head(xle)
basicStats(xle[,2])

#merge brent, wti and SP in same date

xle$Date <- as.Date(xle$Date)


oil <- merge(brt, wti, by = "Date")

df <- merge(oil , xle, by="Date")



df <- df[order(df$Date), ] #Arrange the data by date

head(df)
tail(df)

basicStats(df[2:4])

brt <- df$brent_Price
wti <- df$wti_Price
xle <- df$XLE

#divide train and test dataset
n <- nrow(df)

n_train <- floor(0.7 * n)

train <- df[1:n_train, ]
test <- df[(n_train + 1):n, ]

nrow(train)
nrow(test)

head(train)
tail(train)

head(test)
tail(test)

#output csv
write.csv(df, "data.csv", row.names = FALSE)

write.csv(train, "train.csv", row.names = FALSE)

write.csv(test, "test.csv", row.names = FALSE)
