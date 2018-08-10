ui <- fluidPage(
  titlePanel("Digit Recognizer"),
  
  sidebarLayout(
    
    sidebarPanel(
      fileInput("upload", "Upload new image", accept = c('image/png', 'image/jpeg')),
      actionButton("button", "Predict"),
      fluidRow(
        column(width = 6, offset = 0, style='padding:10px;')
      ),
      textOutput("PredictedLabel"),
      textOutput("Predicted"),
      tags$head(tags$style("#Predicted{
            display: inline-block;
            width: 220px;
            height: 220px;
            border: 1px solid #000;
            text-align: center;
            font-size: 150px;
            background-color: white;
            }
            #PredictedLabel{color: blue;
                            font-size: 20px;
                            font-style: bold;
                            }
                           "))
      
    ),
    mainPanel(
      fluidRow(
        column(6,  textOutput("Original"), imageOutput("img1",height = "220px", width = "220px")),
        column(6, textOutput("Phase1"), imageOutput("img2",height = "220px", width = "220px"))
      ),
      fluidRow(
        column(width = 6, offset = 0, style='padding:10px;')
      ),
      fluidRow(
        column(6, textOutput("Phase2"), imageOutput("img3",height = "220px", width = "220px")),
        column(6, textOutput("Phase3"), imageOutput("img4",height = "220px", width = "220px"))
      ),
      tags$head(tags$style("#Original,#Phase1,#Phase2,#Phase3{color: blue;
                                font-size: 20px;
                                font-style: bold;
                                }
                                #img1,#img2,#img3,#img4{
                                border: 1px solid black;
                                vertical-align: middle;
                                text-align: center;
                                }            
            "))
     # imageOutput("img")
    )
  )
)