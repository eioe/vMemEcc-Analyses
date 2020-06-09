
# Read in mean CDA amplitude from MNE export and save as .RDS: 

library(tidyverse)

path_in = file.path(path_global, 
                    'Data', 
                    'DataMNE', 
                    'EEG', 
                    '07_evokeds', 
                    'summaries', 
                    'PNP', 
                    'stimon', 
                    'global_summary')

fname_in <- file.path(path_in, 'allsubjects-mean_amp_PNP.csv')
df <- read_csv(fname_in)

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
         PNP_amp = pnp_mean_amp) 

# Save as RDS: 
fname = file.path(path_r_data, 'data_PNP.rds')
saveRDS(data_PNP, fname)
