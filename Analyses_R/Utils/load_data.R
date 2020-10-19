#--------------------------------------------------------------------------
# Load prepared .RData
#
#--------------------------------------------------------------------------

#TODO: make function and explicitely return specific df

# excluded subjects:
#QUESTION: exclude these before saving RDS?
excl_subs <- c('VME_S11', 'VME_S14', 'VME_S19', # incomplete data
               'VME_S22', 'VME_S7', 'VME_S12')  # bad EEG

fname <- file.path(path_r_data, 'fulldat_behav.rds')
data_full <- readRDS(fname) %>% 
  filter(!ppid %in% excl_subs)
rm(fname)

#TODO: Implement filters: DroppedFrames!, BadEEGTrials?, IncorrectTrials?

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




