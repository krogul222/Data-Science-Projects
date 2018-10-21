#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

test <- x <- data.frame("X" = as.numeric(args[1]), "Y" = as.numeric(args[2]), "month" = as.numeric(args[3]), "day" = as.numeric(args[4]), "FFMC" = as.numeric(args[5]), "DMC" = as.numeric(args[6]), "DC" = as.numeric(args[7]), "ISI" =  as.numeric(args[8]), "temp" = as.numeric(args[9]), "RH" = as.numeric(args[10]), "wind" = as.numeric(args[11]), "rain" = as.numeric(args[12]))

## load model
model <- readRDS("fireforest_model.rds")

#print forest instance
test

#predict
pred_result <- predict(model,
                        test,
                        type = "raw")
#View prediction
print(paste0("Predicted burned area of new forest fire instance: ", pred_result))