#--------------------------------------------------------------------------
# Load prepared .RData
#
#--------------------------------------------------------------------------

#TODO: make function and explicitely return specific df

# excluded subjects:
excl_subs <- c('VME_S11', 'VME_S14', 'VME_S19', # incomplete data
               'VME_S22', 'VME_S07', 'VME_S12')  # bad EEG

fname <- file.path(path_r_data, 'fulldat_behav.rds')
data_full <- readRDS(fname) %>% 
  filter(!ppid %in% excl_subs)
rm(fname)

#TODO: Implement filters: DroppedFrames

data_behav <- data_full %>% 
  filter(BlockStyle %in% c('perception', 'experiment')) %>% 
  select(ppid, 
         trial_num, 
         block_num, 
         c_StimN, 
         c_Ecc, 
         c_ResponseCorrect, 
         c_ResponseTime, 
         BlockStyle)


# Filter out trials in which a (large) saccade was detected:
# TODO: implement also for perception block

block_style = 'experiment'

path_ET_rej_trials <- file.path(path_global, 
                                'Data', 
                                'DataMNE', 
                                'EEG', 
                                '05.1_rejepo', 
                                'CSV_rejEpos_ET')
files <- list.files(path_ET_rej_trials)

rej_epos_per_sub <- list()
for (file in files) {
  sub_id <- str_split(file, '-')[[1]][1]
  fpath <- file.path(path_ET_rej_trials, file)
  # Extracted trial numbers of the to-be-rejected trials are indices relative to the task 
  # (perception: 1-72, vSTM-task=='experiment': 1:720).
  # `trial_num`in `data_behav` are indices relative to all trials incl. training (1:812).
  # We need to add an offset to compensate for earlier trials in the exp (training trials, trials in perception task):
  trial_num_offset <- if_else(block_style == 'experiment', 92, 10)
  rej_epos <- read_csv(fpath, col_names='trial_num') %>% 
    mutate(trial_num = trial_num + trial_num_offset)
  rej_epos_per_sub[[sub_id]] <- rej_epos
}
rej_epos_df <- bind_rows(rej_epos_per_sub, .id='ppid') %>% mutate(BlockStyle = block_style) 

data_behav <- anti_join(data_behav, rej_epos_df, by = c('ppid', 'trial_num', 'BlockStyle'))



##-----------------------------------------------------------------------
# Read in CDA mean amplitudes:

fname <- file.path(path_r_data, 'data_CDA.rds')
data_CDA <- readRDS(fname)
# convert to uV:
data_CDA <- data_CDA %>% 
  mutate(CDA_amp = CDA_amp * 1e6)
rm(fname)

# Bind to behavioral data: 

data_behav <- left_join(data_behav, 
                    data_CDA, 
                    by = c('ppid', 
                           'trial_num', 
                           'c_StimN', 
                           'c_Ecc'))


##-----------------------------------------------------------------------
# Read in PNP mean amplitudes:

fname <- file.path(path_r_data, 'data_PNP.rds')
data_PNP <- readRDS(fname)
# convert to uV:
data_PNP <- data_PNP %>% 
  mutate(PNP_amp = PNP_amp * 1e6)
rm(fname)

# Bind to behavioral data: 

data_behav <- left_join(data_behav, 
                        data_PNP, 
                        by = c('ppid', 
                               'trial_num', 
                               'c_StimN', 
                               'c_Ecc'))

##-----------------------------------------------------------------------
# Read in mean alpha power differences:

# retention intervall (CDA ROI):

fname <- file.path(path_r_data, 'data_alphapwr_diff_retent_CDAroi.rds')
data_apwr_retent <- readRDS(fname)
rm(fname)

# Bind to behavioral data: 
data_behav <- left_join(data_behav, 
                        data_apwr_retent[, c('ppid', 
                                             'trial_num', 
                                             'c_StimN', 
                                             'c_Ecc', 
                                             'alphapwr_diff_retent')], 
                        by = c('ppid', 
                               'trial_num', 
                               'c_StimN', 
                               'c_Ecc'))



####################################################
## OLD versions: ###################################

# 
# # For now: read in aggregated data
# fname <- file.path(path_r_data, 'data_CDA.rds')
# data_CDA <- readRDS(fname)
# rm(fname)
# 
# 
# # For now: read in aggregated data
# fname <- file.path(path_r_data, 'data_alpha.rds')
# data_alpha <- readRDS(fname)
# rm(fname)




