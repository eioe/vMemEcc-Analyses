

# Read in mean PNP amplitude from MNE export and save as .RDS: 

library(tidyverse)
library(here)

# Define pathes
path_global 	    <- here('../..')
path_r_data       <- file.path(path_global, 'Data/DataR')

PNP_dfs = list()

for (timestyle in c('cluster_times', 'fixed_times')) {
  path_in = file.path(path_global, 
                      'Data2022', 
                      'DataMNE', 
                      'EEG', 
                      '04_evokeds',
                      'PNP', 
                      'summaries', 
                      timestyle)
  
  fname_in <- file.path(path_in, 'allsubjects-mean_amp_PNP.csv')
  df <- read_csv(fname_in) %>%  
    select(!X1)                 # remove col with index 
  
  meanpnp_colname <- str_c("PNP_amp_", str_replace(timestyle, '_', ''))
  
  data_PNP <- df %>%  
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
                  !!meanpnp_colname := pnp_mean_amp) 
  
  
  PNP_dfs[[timestyle]] = data_PNP
}

data_PNP <- left_join(PNP_dfs$cluster_times, PNP_dfs$fixed_times)

# Save as RDS: 
fname = file.path(path_r_data, 'data_PNP.rds')
saveRDS(data_PNP, fname)

