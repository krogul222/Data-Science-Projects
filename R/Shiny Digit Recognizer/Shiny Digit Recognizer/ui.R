ui <- fluidPage(
  titlePanel("Digit Recognizer"),
  
  sidebarLayout(
    
    sidebarPanel(
      fileInput("upload", "Upload new image", accept = c('image/png', 'image/jpeg')),
      actionButton("button", "Predict"),
      textOutput("Predicted"),
      tags$head(tags$style("#Predicted{color: blue;
                                 font-size: 80px;
            font-style: bold;
            }"))
    ),
    mainPanel(
      fluidRow(
        column(6, imageOutput("img1")),
        column(6, imageOutput("img2"))
      ),
      fluidRow(
        column(6, imageOutput("img3")),
        column(6, imageOutput("img4"))
      )
     # imageOutput("img")
    )
  )
)