#--------------------------------------------------------------------------
# Load prepared .RData
#
#--------------------------------------------------------------------------

# excluded subjects:
excl_subs <- c('VME_S11', 'VME_S14', 'VME_S19', # incomplete data
               'VME_S12', 'VME_S13', 'VME_S22') # bad EEG or too many saccades

fname <- file.path(path_r_data, 'fulldat_behav.rds')
data_full <- readRDS(fname) %>% 
  filter(!ppid %in% excl_subs)
rm(fname)


data_behav <- data_full %>% 
  filter(BlockStyle %in% c('perception', 'experiment')) %>% 
  select(ppid, 
         trial_num, 
         block_num, 
         c_StimN, 
         c_Ecc, 
         c_ResponseCorrect, 
         c_ResponseTime, 
         BlockStyle) %>% 
  mutate(ppid = as_factor(ppid),
         c_StimN = as_factor(c_StimN),
         c_Ecc = as_factor(c_Ecc),
         BlockStyle = as_factor(BlockStyle)) 


# Filter out trials in which a (large) saccade was detected:

path_ET_rej_trials <- file.path(path_r_data,
                                'CSV_rejEpos_ET')

for (block_style in c('perception', 'experiment')) {
                             
  files <- list.files(file.path(path_ET_rej_trials, block_style))
  
  rej_epos_per_sub <- list()
  for (file in files) {
    sub_id <- str_split(file, '-')[[1]][1]
    fpath <- file.path(path_ET_rej_trials, block_style, file)
    # Extracted trial numbers of the to-be-rejected trials are indices relative to the task 
    # (perception: 1-72, vSTM-task=='experiment': 1:720).
    # `trial_num`in `data_behav` are indices relative to all trials incl. training (1:812).
    # We need to add an offset to compensate for earlier trials in the exp (training trials, trials in perception task):
    trial_num_offset <- if_else(block_style == 'experiment', 92, 10)
    rej_epos <- read_csv(fpath, col_names='trial_num') 
    if (nrow(rej_epos) == 0) { next }
    rej_epos <- rej_epos %>% 
      mutate(trial_num = trial_num + trial_num_offset)
    rej_epos_per_sub[[sub_id]] <- rej_epos
  }
  rej_epos_df <- bind_rows(rej_epos_per_sub, .id='ppid') %>% 
    mutate(BlockStyle = block_style,
           ppid = as_factor(ppid)) 
  
  data_behav <- anti_join(data_behav, rej_epos_df, by = c('ppid', 'trial_num', 'BlockStyle'))
  
  
  # Export info about rejected trials (ET) in vSTM block:
  if (block_style == 'experiment') {
      
    n_tot_trials_rej_ET <- rej_epos_df %>% nrow()
    summary_trials_rej_ET_per_sub <- rej_epos_df %>% 
      filter(!ppid %in% excl_subs, 
      ) %>% 
      group_by(ppid) %>% summarise(n = n(), 
                                   perc = n*100/720)
    mean_n_trials_rej_ET <- mean(summary_trials_rej_ET_per_sub$n)
    min_n_trials_rej_ET <- min(summary_trials_rej_ET_per_sub$n)
    max_n_trials_rej_ET <- max(summary_trials_rej_ET_per_sub$n)
    mean_perc_trials_rej_ET <- mean(summary_trials_rej_ET_per_sub$perc)
    min_perc_trials_rej_ET <- min(summary_trials_rej_ET_per_sub$perc)
    max_perc_trials_rej_ET <- max(summary_trials_rej_ET_per_sub$perc)
    
    extract_var("mean_n_trials_rej_ET",  mean_n_trials_rej_ET, exp_format="%0.1f")
    extract_var("min_n_trials_rej_ET", min_n_trials_rej_ET, exp_format="%i")
    extract_var("max_n_trials_rej_ET", max_n_trials_rej_ET, exp_format="%i")
    extract_var("mean_perc_trials_rej_ET", mean_perc_trials_rej_ET, exp_format="%0.1f")
    extract_var("min_perc_trials_rej_ET", min_perc_trials_rej_ET, exp_format="%0.1f")
    extract_var("max_perc_trials_rej_ET", max_perc_trials_rej_ET, exp_format="%0.1f")
  }
}



##-----------------------------------------------------------------------
# Read in CDA mean amplitudes:
source(here('Utils', 'read_in_cda.R'))

fname <- file.path(path_r_data, 'data_CDA.rds')
data_CDA <- readRDS(fname)
# convert to uV:
data_CDA <- data_CDA %>% 
  dplyr::mutate(across(contains("CDA_amp"),  ~ .x * 1e6)) %>% 
  mutate(ppid = as_factor(ppid),
         c_StimN = as_factor(c_StimN),
         c_Ecc = as_factor(c_Ecc)) 
rm(fname)

# Bind to behavioral data: 

data_behav <- left_join(data_behav, 
                    data_CDA, 
                    by = c('ppid', 
                           'trial_num', 
                           'c_StimN', 
                           'c_Ecc'))


# Export info about rejected trials (EEG):
n_tot_trials_rej_EEG <- data_behav %>% 
  filter(!ppid %in% excl_subs, 
         BlockStyle == 'experiment',
         is.na(CDA_amp_clustertimes)) %>% 
  group_by(ppid) %>% summarise(n = n(), 
                               perc = n*100/720)
  
#TODO: extract vars

##-----------------------------------------------------------------------
# Read in PNP mean amplitudes:

source(here('Utils', 'read_in_pnp.R'))

fname <- file.path(path_r_data, 'data_PNP.rds')
data_PNP <- readRDS(fname)
# convert to uV:
data_PNP <- data_PNP %>% 
  mutate(across(contains("PNP_amp"), ~.x * 1e6)) %>% 
  mutate(ppid = as_factor(ppid),
         c_StimN = as_factor(c_StimN),
         c_Ecc = as_factor(c_Ecc)) 
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

source(here('Utils', 'read_in_alphapwr_diff_retent.R'))

fname <- file.path(path_r_data, 'data_alphapwr_diff_retent_CDAroi.rds')
data_apwr_retent <- readRDS(fname) %>% 
  mutate(ppid = as_factor(ppid),
         c_StimN = as_factor(c_StimN),
         c_Ecc = as_factor(c_Ecc)) 
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





