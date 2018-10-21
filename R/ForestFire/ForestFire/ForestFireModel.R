library(kableExtra)
library(corrplot)
library(psych)
library(ggplot2)
library(randomForest)   
library(caret)

#Load Data
url <- "http://www.dsi.uminho.pt/~pcortez/forestfires/forestfires.csv"
fire_data <- read.csv(url)

#Print head of the data as kable table
head(fire_data) %>% kable() %>% kable_styling()

#Convert month and day to numerical
month2Number <- function(x) match(tolower(x), tolower(month.abb))
fire_data$month <- month2Number(fire_data$month)

weekday.abb <- c("mon", "tue", "wed", "thu", "fri", "sat", "sun")
fire_data$day <- match(fire_data$day, weekday.abb)

#Review data after day and month change
head(fire_data) %>% kable() %>% kable_styling()

# double check if any missing values present
sapply(fire_data, function(x) sum(is.na(x)))

#Data summary
summary(fire_data)

#Correlation matrix
cor_fire_data <- cor(fire_data)
cor_fire_data_sorted <- as.matrix(sort(cor_fire_data[,'area'], decreasing = TRUE))
cor_fire_data <- cor_fire_data[rownames(cor_fire_data_sorted), rownames(cor_fire_data_sorted)]
corrplot.mixed(cor_fire_data, tl.col="black", addCoef.col = 'black',tl.pos = "lt", tl.cex = 0.7, cl.cex = .7, number.cex=.7)

#correlation sorted by area
cor_fire_data_sorted

#Plot mostly correlated features to area 
ggplot(data=fire_data, aes(x=temp, y=area))+
  geom_point(col='blue') +
  scale_y_continuous(breaks= seq(0, 1100, by=100))

ggplot(data=fire_data, aes(x=DMC, y=area))+
  geom_point(col='blue') +
  scale_y_continuous(breaks= seq(0, 1100, by=100))

#Skewness of area
skew(fire_data$area)
qqnorm(fire_data$area)
qqline(fire_data$area)

ggplot(data=fire_data, aes(x=area)) +
  geom_histogram(fill="blue", binwidth = 5) +
  scale_x_continuous(breaks= seq(0, 1100, by=100))

#Modelling
#Cross validation
my_control <- trainControl(method="cv", number = 5)

#Tuning random forest
tunegrid <- expand.grid(.mtry=c(1:12))

rf_fire <-train(area ~., 
                data=fire_data,
                method ="rf",              # randomForest
                metric = "RMSE",
                tuneGrid=tunegrid,
                ntree = 500,
                trControl=my_control  # cross-validation strategy
)

#Model performance
ggplot(rf_fire)


## save this model
saveRDS(rf_fire , "fireforest_model.rds")
