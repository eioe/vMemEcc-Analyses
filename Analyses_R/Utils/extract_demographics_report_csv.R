## Extract demographic report CSV

# Extract a publishable CSV file with the relevant and well formatted 
# demographic information about the subjects in the vMemEcc study.

library(tidyverse)
library(here)

# get data from virtual drive (set FALSE if you're not Felix)
data_from_seafile <- TRUE

if (data_from_seafile) {
  path_data <- file.path('S:', 
                         'Meine Bibliotheken', 
                         'Experiments', 
                         'vMemEcc', 
                         'Data',  
                         'Demographics')
} else {
  path_data <- here('Data', 
                    'Demographics')
}

data_demog <- read_csv(file.path(path_data, "data_demographic_clean.csv"))

data_demog <- data_demog %>% mutate(gender = recode(.$gender, 
                                                    '1' = "male", 
                                                    '2' = "female", 
                                                    '3' = "diverse", 
                                                    .default = NA_character_)) %>% 
  mutate(handedness = recode(.$handedness, 
                             '1' = 'left', 
                             '2' = 'right', 
                             '3' = 'both', 
                             .default = NA_character_)) %>% 
  dplyr::rename(sub_id = participant_number) %>% 
  select(sub_id, 
         gender, 
         age, 
         handedness, 
         contains('ssq_')) %>% 
  rename_at(.vars = vars(matches('_base$')), 
            .funs = ~str_replace(., '_base', '_pre'))

fname <- 'data_demographic_report.csv'
path_file <- file.path(path_data, fname)
write_csv(data_demog, path_file)
