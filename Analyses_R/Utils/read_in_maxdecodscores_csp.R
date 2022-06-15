
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
                    'csp',
                    'stimon',
                    'collapsed',
                    'LoadLow_vs_LoadHigh',
                    'roc_auc',
                    'shrinkage0.4',
                    'summaries')

fname_in <- file.path(path_in, 'allsubjects-maxDecodScore_csp.csv')
df <- read_csv(fname_in)

data_decodscore <- df %>%  
  mutate(c_Ecc = as_factor(c_Ecc),
         c_Ecc  = recode(c_Ecc, 
                         'S' = 4, 
                         'M' = 9, 
                         'L' = 14),
         BlockStyle = 'experiment') %>% 
  dplyr::rename(ppid = subID) 

# Save as RDS: 
fname = file.path(path_r_data, 'data_maxDecodScore_csp.rds')
saveRDS(data_decodscore, fname)
