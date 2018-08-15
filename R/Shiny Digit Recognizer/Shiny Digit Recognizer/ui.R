library(shiny)

ui <- fluidPage(
  titlePanel("Digit Recognizer"),
  
  sidebarLayout(
    
    sidebarPanel(
      textInput("imageLink", "Link to digit image:", "https://www.researchgate.net/profile/Jose_Sempere/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png"),
      fluidRow(
        column(12, align="center",
               actionButton("loadImage", "Load Image")
        )),
      # fileInput("upload", "Upload new image", accept = c('image/png', 'image/jpeg')),
      fluidRow(
        column(width = 6, offset = 0, style='padding:10px;')
      ),
      fluidRow(
      column(12, align="center",
      textOutput("PredictedLabel"),
      textOutput("Predicted"),
      fluidRow(
        column(12, align="center",
               actionButton("button", "Predict")
        )),
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
                            text-align: center;
            }
            #button{
            display: inline-block;
            text-align: center;
            }")),
      textOutput("keepAlive")
      )
      )
      
    ),
    mainPanel(
      fluidRow(
        column(6, align="center", textOutput("Original"), imageOutput("img1",height = "220px", width = "220px")),
        column(6, align="center", textOutput("Phase1"), imageOutput("img2",height = "220px", width = "220px"))
      ),
      fluidRow(
        column(width = 6, offset = 0, style='padding:10px;')
      ),
      fluidRow(
        column(6, align="center", textOutput("Phase2"), imageOutput("img3",height = "220px", width = "220px")),
        column(6, align="center", textOutput("Phase3"), imageOutput("img4",height = "220px", width = "220px"))
      ),
      tags$head(tags$style("#Original,#Phase1,#Phase2,#Phase3{color: blue;
                                font-size: 20px;
                                font-style: bold;
                                }
                                #img1,#img2,#img3,#img4{
                                border: 1px solid black;
                                vertical-align: middle;
                                }            
            "))
     # imageOutput("img")
    )
  )
)