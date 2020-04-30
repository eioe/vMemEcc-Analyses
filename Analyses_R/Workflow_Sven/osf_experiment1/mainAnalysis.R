#--------------------------------------------------------------------------
# Run main script for analysis for:
# Saccadic selection of stabilized items in visuospatial working memory
# Ohl and Rolfs
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# Clear workspace
rm(list=ls())

# set your directory to the location of this script
# see:
?setwd()
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# Define pathes
path_wd 	 <- getwd()

path_data 	 <- paste(path_wd,"/_datfiles/",sep="")
path_figures <- paste(path_wd,"/_Figures/",sep="")
path_script  <- paste(path_wd,"/_RScripts/",sep="")
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# load packages
source(paste(path_script,"loadPackages.R",sep=""))
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# define colors
source(paste(path_script,"loadColors.R",sep=""))
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# load colors to compute confidence intervals
source(paste(path_script,"loadFunctions.R",sep=""))
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# load theme
source(paste(path_script,"loadTheme.R",sep=""))
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# load data, assign names for columns and define variable type
source(paste(path_script,"loadData.R",sep=""))
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# run analysis 1
# Is there an effect of manual reaction times
source(paste(path_script,"runAnalysis1.R",sep=""))
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# run analysis 2
# Is there a speed accuracy tradeoff
# Are saccade latency and amplitude dependent on experimental condition
source(paste(path_script,"runAnalysis2.R",sep=""))
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# run analysis 3 - main analysis
# Is there an effect of saccades on visual memory performance
# output is Figure 2a of the manuscript
source(paste(path_script,"runAnalysis3.R",sep=""))
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# run analysis 4
# Is the saccadic influence spatially specific
# output is Figure 2b of the manuscript
source(paste(path_script,"runAnalysis4.R",sep=""))
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# Save combined plot
pdf(paste(path_figures,"Figure2.pdf",sep=""), width = 6.2/2.54, height = 8/2.54)
	grid.newpage()
	pushViewport(viewport(layout = grid.layout(100, 100)))
		vplayout <- function(x, y) 
  		viewport(layout.pos.row = x, layout.pos.col = y)
		print(figa, vp = vplayout(1:70, 1:100))
		print(figb, vp = vplayout(64:100, 1:100))		
dev.off()
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# run analysis 5
# Does saccade latency affect memory performance?
source(paste(path_script,"runAnalysis5.R",sep=""))
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# run analysis 6
# Does saccade latency affect memory performance differently than in 
# previous study from Ohl & Rolfs, 2017 JEPLMC
source(paste(path_script,"runAnalysis6.R",sep=""))
#--------------------------------------------------------------------------