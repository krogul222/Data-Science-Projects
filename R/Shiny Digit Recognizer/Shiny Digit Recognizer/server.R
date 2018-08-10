imgAnalysis <- 1

server <- function(input, output, session) {
  library(magick)
  library(broman)
  library(keras)
  library(tools)

  model <- load_model_hdf5('my_model.h5')
  
  observeEvent(input$button, {
    if(length(imgAnalysis) == 784){
      res <- model %>% predict_classes(imgAnalysis,batch_size=1)
      output$Predicted <- renderText(res)   
    } else {
      output$Predicted <- renderText("No Data")   
    }
  })

  observeEvent(input$upload, {
    
    #assign uploaded file to a variable
    File <- input$upload   
    
    #catches null exception
    #if (is.null(File))
     # return(NULL)
    
#    validate(
#      need(file_ext(File$name) %in% c(
#        'image/png', 
#        'image/jpeg'
#      ), "Wrong File Format try again!"))
    
    if (length(input$upload$datapath))
      image <- image_read(input$upload$datapath)
    image_data(image)
    
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
      
      # Return a list
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
    #res <- model %>% predict_classes(imgAnalysis,batch_size=1)
  })
}