

library(tidyverse)
library(eegUtils)
library(stringr)
library(RColorBrewer)
library(ggsci)
library(mne)
library(reticulate)

mpl <- import("matplotlib")
mpl$use("tkAgg")
plt <- import("matplotlib.pyplot")

subjects <- c('P06', 'P07', 'P08', 'P09')
data_path <- 'D:/Felix/Seafile/Experiments/vMemEcc/Data/PilotData/EEG/01_preprocessed'

# for development/debugging:
sub <- subjects[4]

read_matlab <- FALSE
read_py <- TRUE

for (sub in subjects) {
  if (read_matlab) {
    data_file_name <- str_c(sub, '_hp0.01_lp40_epo_blrem_rejepo_rejcomp.set')
    data_file_path <- file.path(data_path, data_file_name)
    eeg_dat <- import_set(data_file_path)
    
    eeg_data <- mne$read_epochs_eeglab(data_file_path)
    
    # create event vector (event info per sampling point):
    eves <- eeg_dat$events$event_type
    eves <- eves[eves < 200 & eves >= 150]
    event_epo <- rep(eves, each=length(eeg_dat$signals$F1)/length(eves))
    
    # filter:
    # eeg_dat <- eeg_filter(eeg_dat, high_freq = 10)
    
    # add to global df:
    if (!exists('eeg_df_tot')) eeg_df_tot <- data.frame()
    eeg_df <- as.data.frame(eeg_dat)
    eeg_df <- add_column(eeg_df,
                         event_epo = event_epo,
                         ID = sub)
    eeg_df_tot <- rbind(eeg_df_tot, eeg_df)
    rm(eeg_dat, eeg_df, event_epo)
  }
  
  else if (read_py) {
    epo <- mne$read_epochs(fname=file.path(data_path, str_c(sub,'CUE-rejComp-epo.fif')))
    
    ## filter?:
    #epo$filter(l_freq = NULL, h_freq = 7)
    
    epo_df <- epo$to_data_frame()
    
    # create event vector (event info per sampling point):
    eves <- epo$events[, 3]
    event_epo <- rep(eves, each=dim(epo_df)[1]/length(eves))
    # add to global df:
    if (!exists('eeg_df_tot')) eeg_df_tot <- data.frame()
    epo_df <- add_column(epo_df,
                         time = rep(epo$times, length(eves)), 
                         event_epo = event_epo,
                         ID = sub)
    eeg_df_tot <- rbind(eeg_df_tot, epo_df)
  }
  
}


eeg_df_tot %>%
  mutate(
    contralat = ifelse(
      ((eeg_df_tot$event_epo-150)%%2) == 1, 
      rowMeans(cbind(PO7, O1, PO3, P3, P7)), 
      rowMeans(cbind(PO8, O2, PO4, P4, P8))), 
    ipsilat = ifelse(
      ((eeg_df_tot$event_epo-150)%%2) == 0, 
      rowMeans(cbind(PO7, O1, PO3, P3, P7)), 
      rowMeans(cbind(PO8, O2, PO4, P4, P8)))) %>%#, 
    #CDA = contralat - ipsilat) %>%
  add_column(memLoad = factor(
               ifelse(
                 ((eeg_df_tot$event_epo-150)%%8) < 4, 2, 4))) %>%
  add_column(ecc = factor(
               ifelse(
                 ((eeg_df_tot$event_epo-150)%%24) < 8, 4, 
                  ifelse(
                    ((eeg_df_tot$event_epo-150)%%24) < 16, 9, 14)))) -> eeg_df_tot_2

eeg_df_tot_2 %>%
  select(time, contralat, ipsilat, memLoad, ecc, ID) %>%
  group_by(time, memLoad, ecc, ID) %>% 
  summarize(aCo = mean(contralat), 
            aIp = mean(ipsilat)) %>%
  mutate(CDA = aCo - aIp) %>%
  ggplot(aes(x = time, y = CDA, group = memLoad)) +
  #  geom_line(aes(group = electrode), alpha = 0.2) + 
  
  stat_summary(fun.y = mean,
               geom = "line",
               size = 1, 
               aes(colour = memLoad)) +
  geom_vline(xintercept = 0) +
  geom_vline(xintercept = -0.8, linetype = 'dotted') +
  geom_hline(yintercept = 0) +
  theme_classic() +
  facet_grid(ID~ecc) + 
  scale_color_jco()

  
    

  
  

eeg_tfr <- compute_tfr(eeg_dat, 
                       method = 'morlet', 
                       foi = c(1,15),
                       n_freq = 10, 
                       n_cycles = 3)
plot_tfr(eeg_tfr, baseline_type = 'db', baseline = c(-0.2, 0))

