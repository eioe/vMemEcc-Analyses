
# Read in mean differences in alpha power from MNE export and save as .RDS: 

library(tidyverse)
library(here)

# Define pathes
path_global 	    <- here('../..')
path_r_data       <- file.path(path_global, 'Data/DataR')

path_in = file.path(path_global, 
                    'Data2022', 
                    'DataMNE', 
                    'EEG', 
                    '05_tfrs', 
                    'summaries', 
                    'induced', 
                    'retention', 
                    'global_summary')

fname_in <- file.path(path_in, 'allsubjects-mean_pwrdiff_retention.csv')
df <- read_csv(fname_in)

data_pwrdiff <- df %>%  
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
                alphapwr_diff_retent = mean_pwr_diff) 

# Save as RDS: 
fname = file.path(path_r_data, 'data_alphapwr_diff_retent_CDAroi.rds')
saveRDS(data_pwrdiff, fname)
