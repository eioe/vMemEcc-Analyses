
data_from_hdd <- TRUE

library(tidyverse)
library(glue)
library(here)
source("EyeTracking/resample_gaze_data.R")


if (data_from_hdd) {
  path_data <- file.path('E:', 'vMemEcc', 'SubjectData_extern', 'SubjectData')
} else {
  path_data <- here('..', '..', 'Data', 'SubjectData')
}



# load subject list and exclude bad subjects:
sub_ids <- str_c('VME_S', str_pad(setdiff(1:27, c(11,14,19)), 
                                  2, 'left', 0))


##### Preprocessing ###########################################

### STEP 1
## crop and resample gaze data
## Crops to [-1.1;2.2] and resamples to 50Hz (choosing the closest sample with the highest confidence). 
## Change the function if you want to change these values. Sorry. 
## Saves the output to disk (one DF/file per subject). If wanted returns one huge DF with all data.
## This is slow. Only run if you want to change the resampling or so. 

# resample_gaze_data_w2disk(sub_ids, path_data, return_df = FALSE)

### STEP 2
## Load resampled data, combine with data about detected saccades

holder <- list()
for (sub_id in sub_ids) {
  path_data_sub <- file.path(path_data, sub_id, 'EyeTracking', 'R_data')
  fname <- file.path(path_data_sub, glue("dataresampled-{sub_id}.rds"))
  data <- readRDS(fname)
  saccdat <- readRDS(file.path(path_data_sub, glue('saccdata_{sub_id}.rds')))
  saccdat <- saccdat %>% 
    select(sub_id, block, trial, reject_eeg) %>% 
    mutate(block_nr = block) %>% 
    group_by(sub_id, block_nr, trial) %>% 
    summarise(reject_eeg = any(reject_eeg)) %>% 
    ungroup() 
  data_joined <- data %>% 
    ungroup() %>% 
    left_join(saccdat) %>% 
    mutate(reject_eeg = if_else(is.na(reject_eeg), FALSE, reject_eeg))
  holder[[sub_id]] <- data_joined 
}

# Merge to one DF:
data_gaze <- bind_rows(holder, .id = 'sub_id')


### STEP 3
## Clean DF for further processing

data_gaze <- data_gaze %>% 
  filter(block_nr != "Block2", # exclude perceptual block
         !reject_eeg) %>%      # exclude trials with saccades >2dva
  select(time_resampled, 
         block_nr, 
         trial, 
         sub_id, 
         gaze_dev_hor, 
         gaze_dev_vert, 
         confidence, 
         eye) %>% 
  # adapt vars
  mutate(block_num = parse_number(block_nr), 
         trial_num_in_block = as.integer(trial), 
         ppid = sub_id,
         .keep = "unused") 

### STEP 4
## Combine with behavioral & experimental data

data_behav <- readRDS(file.path(here(), '..', '..', 'Data', 'DataR', "fulldat_behav.rds")) %>% 
  select(ppid, 
         trial_num_in_block, 
         block_num, c_CueDir, 
         c_ResponseCorrect, 
         c_StimN, 
         c_Ecc) %>% 
  mutate(c_Ecc = as_factor(c_Ecc), 
         c_StimN = as_factor(c_StimN), 
         c_CueDir = as_factor(c_CueDir))

data_gaze <- data_gaze %>% left_join(data_behav) 


### STEP 5 
# Calculate baseline value per trial (mean value in the interval [-1.1; -0.8])

bl <- data_gaze %>% 
  dplyr::filter(((time_resampled > -1.1) & (time_resampled < -0.8))) %>% 
  group_by(ppid, block_num, trial_num_in_block, eye) %>% 
  summarise(bl_val_hor = mean(gaze_dev_hor, na.rm = FALSE), 
            bl_val_vert = mean(gaze_dev_vert, na.rm = FALSE)) %>% 
  ungroup() 

data_gaze <- data_gaze %>% 
  left_join(bl) %>% 
  mutate(bl_val_hor = if_else(is.na(bl_val_hor), 0, bl_val_hor), 
         bl_val_vert = if_else(is.na(bl_val_vert), 0, bl_val_vert), 
         gaze_dev_hor = (gaze_dev_hor - bl_val_hor), 
         gaze_dev_vert = (gaze_dev_vert - bl_val_vert))


### STEP 6
# 
data_plot <- data_gaze %>% 
  filter(eye == 1) %>% 
  mutate(gaze_dev_hor_pooled = if_else(c_CueDir == -1, gaze_dev_hor*-1, gaze_dev_hor)) %>% 
  # Subject means:
  group_by(c_StimN, time_resampled, ppid) %>% 
  summarise(gaze_dev_hor = mean(gaze_dev_hor_pooled), na.rm = FALSE) %>% 
  # summarize across subjects:
  summarise(dev_hor = mean(gaze_dev_hor, na.rm = FALSE), 
            sem = sd(gaze_dev_hor, na.rm = FALSE)/sqrt(21), 
            min = dev_hor - sem, 
            max = dev_hor + sem) %>% 
  ungroup()
  

data_plot %>%  ggplot(aes(x = time_resampled, col = c_StimN)) +  
  geom_line(aes(y = dev_hor)) + 
  geom_line(aes(y = min)) + geom_ribbon(aes(ymin=min, ymax=max, fill = c_StimN), alpha = 0.1) +
  ylim(c(-0.5,0.5)) + 
  xlim(c(-1.1, 2.2)) +
  # facet_grid(rows = 'c_CueDir') +
#  stat_summary(geom="ribbon", fun.min=min, fun.max="max", aes(fill=c_CueDir), alpha=0.3) +
  theme_bw()





