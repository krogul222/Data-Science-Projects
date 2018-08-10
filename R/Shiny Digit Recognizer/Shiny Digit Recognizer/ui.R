
ui <- fluidPage(
  titlePanel("Magick Shiny Demo"),
  
  sidebarLayout(
    
    sidebarPanel(
      fileInput("upload", "Upload new image", accept = c('image/png', 'image/jpeg'))
    ),
    mainPanel(
      imageOutput("img"),
      textOutput("Data")
    )
  )
)