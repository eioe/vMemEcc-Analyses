
library(tidyverse)
library(magrittr)

data_annot <- read_csv('C:/Users/Felix/Seafile/Experiments/vMemEcc/Data/SubjectData/VME_S08/EyeTracking/Block3x/000/exports/000/')
data_gaze <- read_csv('C:/Users/Felix/Seafile/Experiments/vMemEcc/Data/SubjectData/VME_S08/EyeTracking/Block4x/000/exports/000/gaze_positions.csv ')

data_trialtype <- data_annot %>% 
  mutate(ttype = parse_number(label) - 150) %>% 
  filter(ttype < 24 & ttype >= 0) #%>% 

data_stimon <- data_annot %>%  
  mutate(ttype = parse_number(label)) %>% 
  filter(ttype == 2) 


# in case of aborted trials these DFs might have different lengths and need repair:
if (nrow(data_trialtype) > nrow(data_stimon)) {
  # check for subsequent rows in data_trialtype with same ttype:
  n_rows <- nrow(data_trialtype)
  dbl_idx <- which(data_trialtype$ttype[1 : n_rows-1] == data_trialtype$ttype[2 : n_rows])
  
  # now check for all candidates whether there was a stimulus onset between these two trial starts
  # (if there was not, then this was an aborted and repeated trial)
  idx_kickme <- vector(length = 0)
  for (ii in dbl_idx) {
    val <- (any(data_stimon$timestamp > data_trialtype$timestamp[ii] & data_stimon$timestamp < data_trialtype$timestamp[ii+1])) 
    if (!val) {
      idx_kickme <- ii
    }
  }
  
  # clean data_trialtype:
  data_trialtype <- data_trialtype[-idx_kickme, ] 
}

# Add the ttype info to the stimonset df:
data_stimon <- data_stimon %>% 
  mutate(ttype = data_trialtype$ttype, 
         CueDir =      ifelse(mod(ttype, 2) < 1, 'Left', 'Right'), 
         ChangeTrial = ifelse(mod(ttype, 4) < 2, TRUE, FALSE), 
         c_StimN =     ifelse(mod(ttype, 8) < 4, 2, 4), 
         c_Ecc =       ifelse(mod(ttype, 24) < 8, 4, 
                              ifelse(mod(ttype, 24) > 15, 14, 9))) 

