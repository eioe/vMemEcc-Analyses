
# Read in mean CDA amplitude from MNE export and save as .RDS: 

library(tidyverse)
library(here)

# Define pathes
path_global 	    <- here('../..')
path_r_data       <- file.path(path_global, 'Data/DataR')

CDA_dfs = list()

for (timestyle in c('cluster_times', 'fixed_times')) {
  path_in = file.path(path_global, 
                    'Data2022', 
                    'DataMNE', 
                    'EEG', 
                    '04_evokeds',
                    'CDA', 
                    'summaries', 
                    timestyle)

  fname_in <- file.path(path_in, 'allsubjects-mean_amp_CDA.csv')
  df <- read_csv(fname_in) %>%  
    select(!X1)                 # remove col with index 
  
  meancda_colname <- str_c("CDA_amp_", str_replace(timestyle, '_', ''))

  data_CDA <- df %>%  
    mutate(c_StimN = as_factor(c_StimN), 
           c_Ecc = as_factor(c_Ecc), 
           c_StimN = recode(c_StimN, 
                            'LoadLow'  = 2,
                            'LoadHigh' = 4), 
           c_Ecc  = recode(c_Ecc, 
                           'EccS' = 4, 
                           'EccM' = 9, 
                           'EccL' = 14),
           # add offset to account for training and perception trials:
           trial_num = trial_num + 92) %>% 
           dplyr::rename(ppid = subID,
                         !!meancda_colname := cda_mean_amp) 
           

  CDA_dfs[[timestyle]] = data_CDA
}

data_CDA <- left_join(CDA_dfs$cluster_times, CDA_dfs$fixed_times)

# Save as RDS: 
fname = file.path(path_r_data, 'data_CDA.rds')
saveRDS(data_CDA, fname)
