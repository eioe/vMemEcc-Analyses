#--------------------------------------------------------------------------
# Analysis 05:
# Is there an effect of workload or eccentricity on the max decoding performance (sensor space)?
#
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# ... confidence interval for repeated measures design - based on Coussineau Morey

func_analysis_12 <- function() {
  
  condition <- "experiment"  #Only EEG data for VSTM task. 
  
  c1.aov <- data_behav %>% 
    mutate(c_Ecc = as_factor(c_Ecc)) %>% 
    filter(BlockStyle == condition) %>% 
    group_by(ppid, c_Ecc) %>%   
    summarise(mean_decodscore = mean(maxDecodScore_sensorspace, na.rm = T)) %>% 
    ungroup() %>% 
    select("mean_decodscore", "c_Ecc", "ppid") 
  
  c1.reduced <- c1.aov %>% 
    pivot_wider(names_from = c(c_Ecc), 
                values_from = mean_decodscore, 
                names_prefix = 'cond_') %>% 
    select(., contains('cond_')) %>% 
    drop_na()
  
  ci_cm <- cm.ci(data.frame=c1.reduced,
                 conf.level=2*(pnorm(1,0,1)-0.5),
                 difference=TRUE) #1sem or 1.96sem
  
  #--------------------------------------------------------------------------
  ## Run ANOVA
  aov.srt <- aov(mean_decodscore ~ c_Ecc + Error(ppid/c_Ecc),data=c1.aov)
  
  print_header(str_c('Summary ANOVA \ntask: ', condition))
  print(summary(aov.srt))
  
  # # rstatix alternative (identical results:)
  # aov.res <- anova_test(data = c1.aov,
  #                       dv = meanCDA,
  #                       wid = ppid,
  #                       within = c(c_StimN, c_Ecc))
  # get_anova_table(aov.res)
  
  #--------------------------------------------------------------------------
  ## Run post-hoc t tests:
  
  # main effect Eccentricity:
  res_ttest <- c1.aov %>% 
    pairwise_t_test(
      mean_decodscore ~ c_Ecc, paired = TRUE, 
      p.adjust.method = "bonferroni"
    )
  
  print_header(str_c('Results post-hoc t test\ntask: ', condition))
  print(res_ttest)
  #--------------------------------------------------------------------------
  ## Plot
  
  c1.plt <- c1.aov %>% 
    group_by(c_Ecc) %>% 
    summarise(mean_decodscore = mean(mean_decodscore, na.rm = T)) %>% 
    mutate(cond = str_c('cond', c_Ecc, sep = '_'), 
           c_Ecc = as.numeric(as.character(c_Ecc))) %>% 
    left_join(as_tibble(ci_cm) %>% 
                add_column(cond = rownames(ci_cm)), 
              by = 'cond')
  
  txt_title <- ''
  
  figa <- ggplot(c1.plt, 
                 aes(x = c_Ecc, 
                     y = mean_decodscore,
                     ymin = lower,
                     ymax = upper), 
                     colour = 'blue') #+ facet_wrap(~cond)
  #figa <- figa + geom_jitter(position=position_jitter(width=100))
  figa <- figa + geom_line(size=0.2358491) + geom_point(shape=15,size=0.8)
  # figa <- figa + scale_colour_manual(values=c(col_LoadLow, col_LoadHigh))#deforange,defblue,defgrey))
  figa <- figa + scale_fill_manual(values=c(defblue,deforange,defgrey))
  figa <- figa + geom_linerange(size=0.2358491)
  figa <- figa + scale_x_continuous(breaks=c(4,9,14))
  figa <- figa + scale_y_continuous(limits=c(0.5, 1))
  figa <- figa + mytheme
  figa <- figa + ylab("highest decoding score (ROC-AUC)") + xlab("Eccentricity")
#   figa <- figa + labs(title = txt_title, color = "Size Memory Array")
  figa <- figa + theme(legend.position = c(0.85, 1))
  
  plot(figa)
  
  #--------------------------------------------------------------------------
}

