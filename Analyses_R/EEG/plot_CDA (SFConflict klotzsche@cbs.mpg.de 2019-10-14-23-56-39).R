

data_path <- 'D:/Felix/Seafile/Experiments/vMemEcc/Data/PilotData/EEG/01_preprocessed'
data_file_name <- 'P07_hp0.01_lp45_epo_blrem_rejepo_rejcomp.set'
data_file_path <- file.path(data_path, data_file_name)

eeg_dat <- import_set(data_file_path)


epo_right <- select_epochs(eeg_dat, epoch_no = idx_right)

epo_left %>%
  mutate(CDA = (PO8 - PO7)) %>%
  select(CDA) %>%
  ggplot(aes(x = time, y = amplitude)) +
#  geom_line(aes(group = electrode), alpha = 0.2) + 

  stat_summary(aes(group = electrode),
               fun.y = mean,
               geom = "line",
               size = 2) +
  geom_vline(xintercept = 0) +
  geom_hline(yintercept = 0) +
  theme_classic()


epo_right %>%
  mutate(CDA = (PO7 - PO8)) %>%
  select(CDA) %>%
  ggplot(aes(x = time, y = amplitude)) +
  #  geom_line(aes(group = electrode), alpha = 0.2) + 
  
  stat_summary(aes(group = electrode),
               fun.y = mean,
               geom = "line",
               size = 2) +
  geom_vline(xintercept = 0) +
  geom_hline(yintercept = 0) +
  theme_classic()

load_epo <- 

eeg_dat %>%
  mutate(
    contralat = ifelse(
      ((event_epo-150)%%2) == 1, rowMeans(cbind(PO7,PO3,P3)), rowMeans(cbind(PO8,PO4,P4))), 
    ipsilat = ifelse(
      ((event_epo-150)%%2) == 0, rowMeans(cbind(PO7,PO3,P3)), rowMeans(cbind(PO8,PO4,P4))), 
    CDA = contralat - ipsilat) %>%
  select(CDA, contralat, ipsilat) %>%
  ggplot(aes(x = time, y = amplitude, color = electrode)) +
  #  geom_line(aes(group = electrode), alpha = 0.2) + 
  
  stat_summary(fun.y = mean,
               geom = "line",
               size = 1) +
  geom_vline(xintercept = 0) +
  geom_hline(yintercept = 0) +
  theme_classic()



eeg_df <- as.data.frame(eeg_filter(eeg_dat, high_freq = 15))
eeg_df %>%
  mutate(
    contralat = ifelse(
      ((event_epo-150)%%2) == 1, 
      rowMeans(cbind(PO7, O1, PO3, P3, P7)), 
      rowMeans(cbind(PO8, O2, PO4, P4, P8))), 
    ipsilat = ifelse(
      ((event_epo-150)%%2) == 0, 
      rowMeans(cbind(PO7, O1, PO3, P3, P7)), 
      rowMeans(cbind(PO8, O2, PO4, P4, P8))), 
    CDA = contralat - ipsilat) %>%
  add_column(memLoad = 
               ifelse(
                 ((event_epo-150)%%8) < 4, 2, 4)) -> eeg_df

eeg_df %>%
  select(time, CDA, memLoad) %>%
  ggplot(aes(x = time, y = CDA, group = memLoad)) +
  #  geom_line(aes(group = electrode), alpha = 0.2) + 
  
  stat_summary(fun.y = mean,
               geom = "line",
               size = 1, 
               aes(colour = memLoad)) +
  geom_vline(xintercept = 0) +
  geom_hline(yintercept = 0) +
  theme_classic()