#--------------------------------------------------------------------------
# Load prepared .RData
#
#--------------------------------------------------------------------------

#TODO: make function and explicitely return specific df

# excluded subjects:
#QUESTION: exclude these before saving RDS?
excl_subs <- c('VME_S11', 'VME_S14', 'VME_S19')

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

# For now: read in aggregated data
fname <- file.path(path_r_data, 'data_CDA.rds')
data_CDA <- readRDS(fname)
rm(fname)


# For now: read in aggregated data
fname <- file.path(path_r_data, 'data_alpha.rds')
data_alpha <- readRDS(fname)
rm(fname)




