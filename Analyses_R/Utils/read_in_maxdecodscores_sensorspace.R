
# Read in highest decoding score per sibject and ecc condition from MNE export and save as .RDS: 

library(tidyverse)
library(here)

# Define pathes
path_global 	    <- here('../..')
path_r_data       <- file.path(path_global, 'Data/DataR')

path_in = file.path(path_global, 
                    'Data2022', 
                    'DataMNE', 
                    'EEG', 
                    '06_decoding', 
                    'sensorspace',
                    'stimon',
                    'collapsed',
                    'LoadLow_vs_LoadHigh',
                    'roc_auc',
                    'All',
                    'summaries')

fname_in <- file.path(path_in, 'allsubjects-maxDecodScore_sensorspace.csv')
df <- read_csv(fname_in)

data_decodscore <- df %>%  
  mutate(c_Ecc = as_factor(c_Ecc),
         c_Ecc  = recode(c_Ecc, 
                         'EccS' = 4, 
                         'EccM' = 9, 
                         'EccL' = 14),
         BlockStyle = 'experiment') %>% 
  dplyr::rename(ppid = subID) 

# Save as RDS: 
fname = file.path(path_r_data, 'data_maxDecodScore_sensorspace.rds')
saveRDS(data_decodscore, fname)
