imgAnalysis <- 1
dataLoaded <- 0

server <- function(input, output, session) {
  library(magick)
  library(broman)
  library(keras)

  model <- load_model_hdf5('digitrecognizer30epochs.h5')
  
  observeEvent(input$button, {
    if(dataLoaded == 1){
      res <- model %>% predict_classes(imgAnalysis,batch_size=1)
      output$PredictedLabel <- renderText("Prediction")
      output$Predicted <- renderText(res)   
    } else {
      output$PredictedLabel <- renderText("No Data")   
    }
  })

  observeEvent(input$loadImage, {
    
    #assign uploaded file to a variable
  #  File <- input$imageLink   
    
    #catches null exception
  #  if (is.null(File))
   #  return(NULL)
    
    output$Original <- renderText("Original")
    output$Phase1 <- renderText("Phase 1")
    output$Phase2 <- renderText("Phase 2")
    output$Phase3 <- renderText("Phase 3")
       
#    if (length(input$upload$datapath))
 #     image <- image_read(input$upload$datapath)
    
    if (is.null(input$imageLink)) return(NULL)
    
  #  if (length(input$imageLink))

    tryCatch({
      image <- image_read(input$imageLink)

      
    output$img1 <- renderImage({
      tmpfile <- image %>%
      image_resize("200x200!") %>%
      image_write(tempfile(fileext='jpg'), format = 'jpg')
      # Return a list
      list(src = tmpfile, contentType = "image/jpeg")
    })
    
    output$img2 <- renderImage({
      # Numeric operators
      tmpfile <- image %>%
        image_resize("28x28!") %>%
        image_resize( geometry_size_pixels(200)) %>%
        image_write(tempfile(fileext='jpg'), format = 'jpg')
      
       #Return a list
      list(src = tmpfile, contentType = "image/jpeg")
    })
    
    output$img3 <- renderImage({
      # Numeric operators
      tmpfile <- image %>%
        image_resize("28x28!") %>%
        image_resize( geometry_size_pixels(200)) %>%
        image_convert(type = 'grayscale') %>%
        image_write(tempfile(fileext='jpg'), format = 'jpg')
      
      # Return a list
      list(src = tmpfile, contentType = "image/jpeg")
    })
    
    output$img4 <- renderImage({
      # Numeric operators
      tmpfile <- image %>%
        image_resize("28x28!") %>%
        image_resize( geometry_size_pixels(200)) %>%
        image_convert(type = 'grayscale') %>%
        image_negate() %>%
        image_write(tempfile(fileext='jpg'), format = 'jpg')
      
      # Return a list
      list(src = tmpfile, contentType = "image/jpeg")
    })
    
    imgAnalysis <<- image %>% 
    image_resize( "28x28!") %>%
    image_negate() %>%
    image_data(channels = "gray") %>%
    hex2dec()
    
    imgAnalysis <<- imgAnalysis/255
    dim(imgAnalysis) <<- c(1, 28, 28, 1)
    dataLoaded <<- 1
    },
    error = function(e) {
      output$PredictedLabel <- renderText("Link not compatible. Image loading error.") 
      dataLoaded <<- 0
      return(NULL)
    })
    output$Predicted <- renderText("")  
  })
}