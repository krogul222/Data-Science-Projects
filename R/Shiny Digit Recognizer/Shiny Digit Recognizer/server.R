server <- function(input, output, session) {
  library(magick)
  library(broman)
  library(keras)
  
  model <- readRDS("./DigitRecognizer.rds")
  test <- read.csv('test.csv', stringsAsFactors = FALSE, na.strings = c("NA", ""))
  # reshape
  test <- matrix(as.numeric(unlist(test)),nrow=nrow(test))
  dim(test) <- c(nrow(test), 28, 28, 1)
  # rescale
  test <- test / 255
  #model <- load(file = "./DigitRecognizer.rds")
  # Start with placeholder img
 # image <- image_read("https://images-na.ssl-images-amazon.com/images/I/81fXghaAb3L.jpg")
  observeEvent(input$upload, {
    if (length(input$upload$datapath))
      image <- image_read(input$upload$datapath)
   # info <- image_info(image)
    #updateCheckboxGroupInput(session, "effects", selected = "")
    #updateTextInput(session, "size", value = paste(info$width, info$height, sep = "x"))
    
    image_data(image)
    
    output$img <- renderImage({
      # Numeric operators
      tmpfile <- image %>%
        image_resize( geometry_size_pixels(28)) %>%
        image_resize( geometry_size_pixels(300)) %>%
        image_convert(type = 'grayscale') %>%
        image_negate() %>%
        image_write(tempfile(fileext='jpg'), format = 'jpg')
      
      # Return a list
      list(src = tmpfile, contentType = "image/jpeg")
    })
    
    imgAnalysis <- image %>% 
    image_resize(geometry_size_pixels(28)) %>%
    image_negate() %>%
    image_data(channels = "gray") %>%
    hex2dec()
    
    imgAnalysis <- imgAnalysis/255
    
    dim(imgAnalysis) <- c(28, 28, 1)
    
    
    
    #output$Data <- renderText(imgAnalysis)
    #predict <- model %>% predict_classes(test, batch_size = 1)
    #predict(imgAnalysis,model)
    # create the final dst
    
    output$Data <- renderText(test)
    
    
  })
}