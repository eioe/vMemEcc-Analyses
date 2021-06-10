

## Helper script to gather the trials in the perception block that need to be rejected due to ET analysis (saccades found)

# Should not be necessary in proper pipeline as its functionality is now implemented in `analyze_gaze.R`
# (c) 2021: Felix Klotzsche


path_hdd <- 'D:\\vMemEcc/SubjectData_extern/SubjectData/'
path_r_data <- here('..', '..', 'Data', 'DataR')
path_data_rejepo_out <- file.path(path_r_data, 'CSV_rejEpos_ET')

sub_ids <- list.files(path_hdd, pattern = '*VME_S*')

d_l <- list()
for (sub_id in sub_ids) {
  fpath <- file.path(path_hdd, sub_id, 'EyeTracking', 'R_data', str_c('saccdata_', sub_id, '.rds'))
  if (file.exists(fpath)) {
    sacc_info <- readRDS(fpath)
    
    rej_trials <- sacc_info %>% 
      ungroup() %>% 
      filter(block == "Block2", 
             reject_eeg == TRUE) %>% 
      select(trial) 
    
    rej_trials <- unique(rej_trials)
    fname <- file.path(path_hdd, sub_id, 'EyeTracking', 'R_data', str_c(sub_id, '-rejTrials-ET-perception.csv'))
    write_csv2(rej_trials, fname, col_names = F)
    
    fname <- file.path(path_data_rejepo_out, 'perception', str_c(sub_id, '-rejTrials-ET-perception.csv'))
    write_csv2(rej_trials, fname, col_names = F)
  }
}
r_all <- bind_rows(d_l, .id = 'ppid')
