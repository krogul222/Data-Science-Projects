suppressMessages(library(lattice))
suppressMessages(library(ggplot2))
suppressMessages(library(tfdeploy))

# for describe the data
suppressMessages(library(psych))

# caret for the split
suppressMessages(library(caret))

# Keras
suppressMessages(library(keras))
suppressMessages(library(kerasR))

# load train and test dataset
train <- read.csv('train.csv', stringsAsFactors = FALSE, na.strings = c("NA", ""))
test <- read.csv('test.csv', stringsAsFactors = FALSE, na.strings = c("NA", ""))

# DATA OVERVIEW

# Dataset samples 
head(train, n=1L)
head(test, n=1L)

# Labels summary table
t(as.data.frame(table(as.factor(train$label))))

# Labels summary plot
ggplot(train, aes(x=label, fill=as.factor(label))) +
    geom_histogram(stat = "count") +
    scale_x_discrete(name = "Numbers", limits = c(0,1,2,3,4,5,6,7,8,9)) +
    scale_y_discrete(name = "Count", limits = c(0, 1000, 2000, 3000, 4000,5000))

#28x28 pixel digit image
drawDigit <- function(row) {
  image(t(apply(
    matrix(unlist(train[row, -1]),
    nrow = 28, byrow = TRUE),
    2, rev)),
    col = gray.colors(255))
}

# create a grid
par(mfrow=c(2,4))

# draw sample digits
sampleDigits <- sapply(sample(1:42000, 8, replace = T), drawDigit) 

# PREPARING DATA
splitTrainData <- createDataPartition(train$label, p = 0.9, list = FALSE)
kerasTrain <- train[c(splitTrainData), ]
kerasTest <- train[c(-splitTrainData), ]

X_kerasTrain <- kerasTrain[, -c(1)]
Y_kerasTrain <- kerasTrain[, c(1)]

X_kerasTest <- kerasTest[, -c(1)]
Y_kerasTest <- kerasTest[, c(1)]

# reshape

X_kerasTrain <- matrix(as.numeric(unlist(X_kerasTrain)), nrow = nrow(X_kerasTrain))
X_kerasTest <- matrix(as.numeric(unlist(X_kerasTest)), nrow = nrow(X_kerasTest))

# rescale
X_kerasTrain <- X_kerasTrain / 255
X_kerasTest <- X_kerasTest / 255
dim(X_kerasTrain) <- c(nrow(X_kerasTrain), 28, 28, 1)
dim(X_kerasTest) <- c(nrow(X_kerasTest), 28, 28, 1)

Y_kerasTrain <- to_categorical(Y_kerasTrain, 10)
Y_kerasTest <- to_categorical(Y_kerasTest, 10)
# Modelling

model <- keras_model_sequential()

# model %>% 
#  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
#  layer_dropout(rate = 0.4) %>% 
#  layer_dense(units = 128, activation = 'relu') %>%
#  layer_dropout(rate = 0.3) %>%
#  layer_dense(units = 10, activation = 'softmax')

model %>%
  layer_conv_2d(filters = 32, kernel_size = c(5,5), padding = 'same', activation = 'relu', input_shape = c(28,28,1))  %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), padding = 'same', activation = 'relu')  %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), padding = 'same', activation = 'relu')  %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), padding = 'same', activation = 'relu')  %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = 'softmax')
  
summary(model)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

reduceLearningRate <- callback_reduce_lr_on_plateau(monitor = "val_acc", factor = 0.5,
                              patience = 3, verbose = 0, mode = c("max"),
                              min_delta = 1e-04, cooldown = 0, min_lr = 1e-05)

dataGenerator <- image_data_generator(featurewise_center = FALSE, samplewise_center = FALSE,
                     featurewise_std_normalization = FALSE,
                     samplewise_std_normalization = FALSE, zca_whitening = FALSE,
                     zca_epsilon = 1e-06, rotation_range = 15, width_shift_range = 0.1,
                     height_shift_range = 0.1, brightness_range = NULL, shear_range = 0,
                     zoom_range = 0.1, channel_shift_range = 0, fill_mode = "nearest",
                     cval = 0, horizontal_flip = FALSE, vertical_flip = FALSE,
                     rescale = NULL, preprocessing_function = NULL, data_format = NULL,
                     validation_split = 0)

dataGenerator %>% fit_image_data_generator(X_kerasTrain)

history <- model %>% fit_generator(flow_images_from_data(X_kerasTrain, Y_kerasTrain, dataGenerator, batch_size = 128),
                                   steps_per_epoch = nrow(X_kerasTrain)/128, epochs = 1, callbacks = c(reduceLearningRate), validation_data = list(X_kerasTest, Y_kerasTest) )

model %>% evaluate(X_kerasTest, Y_kerasTest)

export_savedmodel(model, "digitrecognizer2", , remove_learning_phase = FALSE)

save_model_hdf5(model, 'my_model.h5')

# Save entire model to a HDF5 file
keras_save(model, path = "digitrecognizer.h5")
#saveRDS(model, "./DigitRecognizer.rds")

# reshape
test <- matrix(as.numeric(unlist(test)),nrow=nrow(test))
dim(test) <- c(nrow(test), 28, 28, 1)
# rescale
test <- test / 255

# predict 
predictFinal <- model %>% predict_classes(test,batch_size=100)
# create the final dst
destinationSubmit <- data.frame(ImageId=1:nrow(test),Label=predictFinal)

# check some data
head(destinationSubmit)

# write the submit csv file
write.csv(destinationSubmit,file = "DigitSubmit.csv",row.names = FALSE)