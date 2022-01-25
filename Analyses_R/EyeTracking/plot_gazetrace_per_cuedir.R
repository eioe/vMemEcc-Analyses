

holder <- list()
for (sub_id in sub_ids) {
  path_data_sub <- file.path(path_data, sub_id, 'EyeTracking', 'R_data')
  fname <- file.path(path_data_sub, glue("dataresampled-{sub_id}.rds"))
  data <- readRDS(fname)
  saccdat <- readRDS(file.path(path_data_sub, glue('saccdata_{sub_id}.rds')))
  saccdat <- saccdat %>% 
    ungroup() %>%  
    mutate(block = as.character(block)) %>% 
    select(sub_id, block, trial, reject_eeg) %>% 
    mutate(block_nr = as.character(block), 
           trial = as.character(trial)) %>% 
    group_by(sub_id, block_nr, trial) %>% 
    summarise(reject_eeg = any(reject_eeg)) %>% 
    mutate(rej_eeg = as.character(reject_eeg)) %>% 
    ungroup() %>% 
    distinct(sub_id, block_nr, trial, .keep_all = TRUE)
  data_joined <- data %>% mutate(trial = as.character(trial)) %>% ungroup() %>% left_join(saccdat) %>% 
    mutate(reject_eeg = if_else(is.na(reject_eeg), FALSE, reject_eeg))
  holder[[sub_id]] <- data_joined 
}



monster <- bind_rows(holder, .id = 'sub_id')

# both eyes to common time
# resample
# long format
# throw out bad trials (conf < 0.65 for > 33%) 


holder2 <- list()
for (sub_id in sub_ids) {
  ouut<- monster %>% 
    # slice(which(row_number() %% 5 == 1)) %>% 
    filter(sub_id == sub_id) %>% 
    select(sub_id, gaze_timestamp, trial, block_nr, eye, confidence, gaze_dev0_hor, gaze_dev1_hor) %>% 
    group_by(sub_id,
             trial,
             block_nr,
             eye) %>% 
    et_resample("gaze_timestamp", 
                c("gazedev_", "confidence"), 
                srate = 50, 
                tmax = 2.2, 
                tmin = -1.1) 
  fname <- file.path(path_data_sub, glue("dataresampled-{sub_id}.rds"))
  saveRDS(ouut, fname)
  print("##########################\n")
  print(glue("Done with {sub_id}!"))
  print("##########################\n")
  
}

#saccdat <- readRDS(file.path(path_data_sub, glue('saccdata_{sub_id}.rds')))

# discard bad motherfuckers trials
monster %>% 
  # downsample
  # slice(which(row_number() %% 5 == 1)) %>% 
  # get exp trials only:
  filter(block_nr != "Block2" & !reject_eeg) %>% 
  # 
  group_by(sub_id, 
           block_nr,
           trial) %>% 
  summarize(n = n(),
            n_lowconf_sampels = sum(confidence < 0.6),
            perc_lowconf_samples = n_lowconf_sampels / n) %>% 
  mutate(bmf_trial = perc_lowconf_samples > 0.33) -> shw
  




tmax_ <- max(me$gaze_timestamp, na.rm = T)
tmin_ <- min(me$gaze_timestamp, na.rm = T)
tdiff <- tmax - tmin




alldf_0 <- monster %>% 
  #left_join(shw) %>% 
  #choose eye & exp blocks
  filter(block_nr != "Block2", 
         #!bmf_trial,
         #eye == 1, 
         !reject_eeg) %>%
  # downsample
  # slice(which(row_number() %% 2 == 1)) %>% 
  # choose cols
  select(time_resampled, 
         block_nr, 
         trial, 
         sub_id, 
         gaze_dev_hor, 
         confidence, 
         eye) %>% 
  # adapt vars
  mutate(block_num = parse_number(block_nr), 
         trial_num_in_block = as.integer(trial), 
         ppid = sub_id,
         .keep = "unused") #%>% 

bl <- alldf_0 %>% 
  dplyr::filter(((time_resampled > -1.1) & (time_resampled < -0.8))) %>% 
  group_by(ppid, block_num, trial_num_in_block, eye) %>% 
  summarise(bl_val = mean(gaze_dev_hor, na.rm = FALSE)) %>% 
  ungroup() 

cdat <- alldf_0 %>% left_join(d_behv) 
cdat <- cdat %>%  left_join(bl)


tmp_plot <- cdat %>% 
  #filter(!((block_num == 12) & (sub_id == 'VME_S21'))) %>% 
  mutate(bl_val = if_else(is.na(bl_val), 0, bl_val)) %>% 
  mutate(gaze_dev_hor = (gaze_dev_hor - bl_val), 
         c_Ecc = as_factor(c_Ecc), 
         c_StimN = as_factor(c_StimN)) %>%
  mutate(gaze_dev_hor = if_else(c_CueDir == -1, gaze_dev_hor*-1, gaze_dev_hor)) %>% 
  group_by(c_StimN, time_resampled, ppid) %>% 
  summarise(gaze_dev_hor = mean(gaze_dev_hor), na.rm = FALSE) %>% 
  summarise(dev_hor = mean(gaze_dev_hor, na.rm = FALSE), 
            sem = sd(gaze_dev_hor, na.rm = FALSE)/sqrt(21), 
            min = dev_hor - sem, 
            max = dev_hor + sem) %>% 
  #mutate(c_CueDir = as_factor(c_CueDir)) %>% 
  ungroup()# , 
    #dev_hor = rollmean(na.spline(dev_hor), 5, 
    #       na.pad = TRUE))
  

tmp_plot %>%  ggplot(aes(x = time_resampled, col = c_StimN)) +  
  geom_line(aes(y = dev_hor)) + 
  geom_line(aes(y = min)) + geom_ribbon(aes(ymin=min, ymax=max, fill = c_StimN), alpha = 0.1) +
  ylim(c(-0.5,0.5)) + 
  xlim(c(-1.1, 2.2)) +
  # facet_grid(rows = 'c_CueDir') +
#  stat_summary(geom="ribbon", fun.min=min, fun.max="max", aes(fill=c_CueDir), alpha=0.3) +
  theme_bw()


data_behav <- readRDS(file.path(here(), '..', '..', 'Data', 'DataR', "fulldat_behav.rds"))

d_behv <- data_behav %>% 
 # filter(ppid == "VME_S02") %>% 
  select(ppid, trial_num_in_block, block_num, c_CueDir, c_ResponseCorrect, c_StimN, c_Ecc)


