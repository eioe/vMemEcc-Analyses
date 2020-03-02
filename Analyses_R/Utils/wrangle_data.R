
# Author         : Felix Klotzsche    ---    eioe
# Date           : 11 July 2019

###############################################################################
#                                                                             #
# Data wranglers                                                              #
#                                                                             #
###############################################################################




convCortMagFac <- function(dataFrame) {
  idxT <- (dataFrame$c_CortMag == TRUE)
  idxF <- (dataFrame$c_CortMag == FALSE)
  
  dataFrame$c_CortMag[idxT] <- "2"
  dataFrame$c_CortMag[idxF] <- "0"
  dataFrame <- mutate_at(dataFrame, vars(c_CortMag), as.numeric)
  return(dataFrame)
}



change_ppid <- function(dataFrame, newLabel) {
  dataFrame$ppid <- newLabel
  return(dataFrame)
}
