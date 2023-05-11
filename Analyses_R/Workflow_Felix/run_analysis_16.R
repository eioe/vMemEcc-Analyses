#--------------------------------------------------------------------------
# Is there an effect of workload or eccentricity on the number of rejected trials?
#
#--------------------------------------------------------------------------
# Authors: Sven Ohl & Felix Klotzsche, 2020
#--------------------------------------------------------------------------
# ... confidence interval for repeated measures design - based on Coussineau Morey

func_analysis_16<- function() {
  
  c1.aov <- data_behav %>% 
    filter(BlockStyle == 'experiment', !is.na(CDA_amp_clustertimes)) %>% 
    group_by(ppid, c_StimN, c_Ecc) %>%   
    summarise(prop_trials_rej = (120-n()) / 120) %>% 
    ungroup() %>% 
    select("prop_trials_rej", "c_StimN", "c_Ecc", "ppid") 
  
  c1.reduced <- c1.aov %>% 
    pivot_wider(names_from = c(c_StimN, c_Ecc), 
                values_from = prop_trials_rej, 
                names_prefix = 'cond_') %>% 
    select(., contains('cond_'))
    
  ci_cm <- cm.ci(data.frame=c1.reduced,
                 conf.level=2*(pnorm(1,0,1)-0.5),
                 difference=TRUE) #1sem or 1.96sem
  
  #--------------------------------------------------------------------------
  ## Run ANOVA
  aov.srt <- aov(prop_trials_rej ~ c_StimN*c_Ecc + Error(ppid/(c_StimN* c_Ecc)),data=c1.aov)
  
  results = list()
  results[["aov.srt"]] <- aov.srt
  
  print_header(str_c('Summary ANOVA \nprop remaining trials'))
  print(summary(aov.srt))
  
  
  #--------------------------------------------------------------------------
  ## Run post-hoc t tests:
  
  # main effect Eccentricity:
  res_ttest <- data_behav %>% 
    filter(BlockStyle == 'experiment', !is.na(CDA_amp_clustertimes)) %>% 
    group_by(ppid, c_Ecc) %>% 
    summarise(prop_trials_rej = (240-n())/240) %>% 
    ungroup() %>% 
    pairwise_t_test(
      prop_trials_rej ~ c_Ecc, paired = TRUE, 
      p.adjust.method = "bonferroni",
      detailed = TRUE
    )

  print_header(str_c('Results post-hoc t test\nprop remaining trials'))
  print(res_ttest)
  results[["res_ttest"]] <- res_ttest
  
  means <- data_behav %>% 
    filter(BlockStyle == 'experiment', !is.na(CDA_amp_clustertimes)) %>% 
    group_by(ppid, c_StimN) %>% 
    summarise(prop_trials_rej = (360-n())/360,
              n_trials_rej = 360 - n()) %>%
    select(c_StimN, ppid, prop_trials_rej) %>% 
    group_by(c_StimN, ppid) %>%
    pivot_wider(id_cols =  ppid, names_from = c_StimN, values_from = prop_trials_rej) %>% 
    ungroup() %>% 
    select(!ppid) %>% 
    summarise_all(.funs = c(mean))
  print(means)
  
  means <- data_behav %>% 
    filter(BlockStyle == 'experiment', !is.na(CDA_amp_clustertimes)) %>% 
    group_by(ppid, c_Ecc) %>% 
    summarise(prop_trials_rej = (240-n())/240,
              n_trials_rej = 240 - n()) %>%
    select(c_Ecc, ppid, prop_trials_rej) %>% 
    group_by(c_Ecc, ppid) %>%
    pivot_wider(id_cols =  ppid, names_from = c_Ecc, values_from = prop_trials_rej) %>% 
    ungroup() %>% 
    select(!ppid) %>% 
    summarise_all(.funs = c(mean))
  print(means)
  
  #--------------------------------------------------------------------------
  ## Plot
  
  c1.plt <- c1.aov %>% 
    group_by(c_StimN, c_Ecc) %>% 
    summarise(prop_trials_rej = mean(prop_trials_rej)) %>% 
    mutate(cond = str_c('cond', c_StimN, c_Ecc, sep = '_'), 
           c_Ecc = as.numeric(as.character(c_Ecc))) %>% 
    left_join(as_tibble(ci_cm) %>% 
                add_column(cond = rownames(ci_cm)), 
              by = 'cond')
  
  txt_title <- 'VSTM Task: remaining trials'
  
  figa <- ggplot(c1.plt, 
                 aes(x = c_Ecc, 
                     y = prop_trials_rej,
                     ymin = lower,
                     ymax = upper, 
                     colour = as_factor(c_StimN))) #+ facet_wrap(~cond)
  #figa <- figa + geom_line(size=0.2358491)  + geom_point(shape=15,size=2)
  figa <- figa + geom_line(size=0.2358491) + geom_point(shape=15,size=0.8)
  figa <- figa + scale_colour_manual(values=c(col_LoadLow, col_LoadHigh, defgrey))  # values=c(defblue,deforange,defgrey))
  figa <- figa + scale_fill_manual(values=c(defblue,deforange,defgrey))
  figa <- figa + geom_linerange(size=0.2358491)
  figa <- figa + scale_x_continuous(breaks=c(4,9,14))
  figa <- figa + scale_y_continuous(limits=c(0.0,0.15))
  figa <- figa + mytheme
  figa <- figa + ylab("Proportion of rejected trials") + xlab("Eccentricity")
  # figa <- figa + labs(title = txt_title, color = "Size Memory Array")
  figa <- figa + theme(legend.position = c(0.9, 0.9))
  
  plot(figa)
  
  fname = file.path(path_global, 'Plots2022', 'Else', str_c('rejected_trials_per_ecc.pdf'))
  ggsave(plot = figa,
         width = 10/2.54,
         height = 2*3.7/2.54,
         dpi = 300,
         filename = fname)


  #--------------------------------------------------------------------------
  n_trials_rej_overall <- data_behav %>% 
    filter(BlockStyle == 'experiment', !is.na(CDA_amp_clustertimes)) %>% 
    group_by(ppid, c_StimN, c_Ecc) %>%   
    summarise(n_trials_rej = 120-n()) %>% 
    ungroup() %>% 
    select("n_trials_rej", "c_StimN", "c_Ecc", "ppid") 
  
  ## n_trials_rej_ET 
  
  path_in <- file.path(path_r_data, "CSV_rejEpos_ET", "experiment")
  files <- list.files(path_in)
  
  trials_ETrej <- tibble()
  
  for (f in files) {
    fname <- file.path(path_in, f)
    dat <- read_csv(fname, col_names=FALSE, progress = FALSE, col_types = c("d"))
    colnames(dat) <- c("trial_num")
    dat$ppid <- str_split(f, '-')[[1]][1]
    
    trials_ETrej <- rbind(trials_ETrej, dat)
  }
  
  trials_ETrej <- trials_ETrej %>% 
    filter(ppid %in% unique(data_behav$ppid))  # discard invalid participants
  
  # combine with info about ecc:
  db_f <- data_full %>% 
    filter(BlockStyle == 'experiment') %>% 
    select(ppid, trial_num, c_Ecc, c_StimN) %>% 
    mutate(trial_num = trial_num - 92)  # fix index to start at 1
  
  n_trials_rej_ET <- left_join(trials_ETrej, db_f, by=c("ppid", "trial_num")) %>% 
    group_by(ppid, c_Ecc, c_StimN) %>% 
    summarise(n_trials_rej_ET = n()) %>% 
    mutate(c_Ecc = as.factor(c_Ecc), 
           c_StimN = as.factor(c_StimN))
  
  df_n_trials_rej <- full_join(n_trials_rej_overall, n_trials_rej_ET, by=c("ppid", "c_Ecc", "c_StimN")) %>% 
    replace_na(list(n_trials_rej_ET = 0)) %>% 
    mutate(n_trials_rej_autorej = n_trials_rej - n_trials_rej_ET) %>% 
    select(-n_trials_rej) %>% 
    mutate(prop_trials_rej_ET = n_trials_rej_ET / 120, 
           prop_trials_rej_autorej = n_trials_rej_autorej / 120) %>% 
    pivot_longer(c(prop_trials_rej_ET, prop_trials_rej_autorej), names_prefix = "prop_trials_rej_") %>% 
    group_by(c_Ecc, name) %>% 
    summarise(value = mean(value))
    
  category_order <- c("ET", "autorej")
  df_n_trials_rej$name <- factor(df_n_trials_rej$name, levels = category_order, labels = c("saccade", "bad EEG"))
    
  figa <- ggplot(df_n_trials_rej, aes(x = c_Ecc, y = value, fill = name)) + 
    geom_col() + theme_minimal() + scale_fill_manual(values = c("darkgrey", "black")) 
figa
  
  figa <- figa + scale_y_continuous(limits=c(0.0,0.15))
  figa <- figa + mytheme
  figa <- figa + ylab("Proportion of rejected trials") + xlab("Eccentricity")
  figa <- figa + theme(legend.position = c(0.5, 0.7))
  
  plot(figa)
  
  fname = file.path(path_global, 'Plots2022', 'Else', str_c('rejected_trials_per_ecc_per_reason.pdf'))
  ggsave(plot = figa,
         width = 0.4 * 10/2.54,
         height = 0.4 * 2*3.7/2.54,
         dpi = 300,
         filename = fname)
  
  return(results)
}

