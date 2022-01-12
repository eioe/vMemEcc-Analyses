#--------------------------------------------------------------------------
# Is there an effect of workload or eccentricity on reaction time?
#
#--------------------------------------------------------------------------
# Authors: Sven Ohl & Felix Klotzsche, 2020
#--------------------------------------------------------------------------
# ... confidence interval for repeated measures design - based on Coussineau Morey

func_analysis_03 <- function(condition) {
  
  c1.aov <- data_behav %>% 
    filter(BlockStyle == condition, 
           c_ResponseTime < 3) %>%  #TODO: Implement proper outlier correction!
    group_by(ppid, c_StimN, c_Ecc) %>%   
    summarise(meanRT = mean(c_ResponseTime)) %>% 
    ungroup() %>% 
    select("meanRT", "c_StimN", "c_Ecc", "ppid") 
  
  c1.reduced <- c1.aov %>% 
    pivot_wider(names_from = c(c_StimN, c_Ecc), 
                values_from = meanRT, 
                names_prefix = 'cond_') %>% 
    select(., contains('cond_'))
  
  ci_cm <- cm.ci(data.frame=c1.reduced,
                 conf.level=2*(pnorm(1,0,1)-0.5),
                 difference=TRUE) #1sem or 1.96sem
  
  #--------------------------------------------------------------------------
  ## Run ANOVA
  aov.srt <- aov(meanRT ~ c_StimN*c_Ecc + Error(ppid/(c_StimN* c_Ecc)),data=c1.aov)
  
  print_header(str_c('Summary ANOVA \ntask: ', condition))
  print(summary(aov.srt))
  
  ## rstatix alternative (identical results:)
  # aov.res <- anova_test(data = c1.aov, 
  #                       dv = meanAcc, 
  #                       wid = ppid, 
  #                       within = c(c_StimN, c_Ecc))
  # get_anova_table(aov.res)
  
  #--------------------------------------------------------------------------
  ## Run post-hoc t tests:
  
  # main effect Eccentricity:
  res_ttest <- c1.aov %>% 
    group_by(ppid, c_Ecc) %>% 
    summarise(meanRT = mean(meanRT)) %>% 
    ungroup() %>% 
    pairwise_t_test(
      meanRT ~ c_Ecc, paired = TRUE, 
      p.adjust.method = "bonferroni"
    )
  
  print_header(str_c('Results post-hoc t test\ntask: ', condition))
  print(res_ttest)
  #--------------------------------------------------------------------------
  ## Plot
  
  c1.plt <- c1.aov %>% 
    group_by(c_StimN, c_Ecc) %>% 
    summarise(meanRT = mean(meanRT)) %>% 
    mutate(cond = str_c('cond', c_StimN, c_Ecc, sep = '_'), 
           c_Ecc = as.numeric(as.character(c_Ecc))) %>% 
    left_join(as_tibble(ci_cm) %>% 
                add_column(cond = rownames(ci_cm)), 
              by = 'cond')
  
  txt_title <- ifelse(condition == 'perception', 
                      'Perceptual Task', 
                      'VSTM Task')
  
  figa <- ggplot(c1.plt, 
                 aes(x = c_Ecc, 
                     y = meanRT,
                     ymin = lower,
                     ymax = upper, 
                     colour = as_factor(c_StimN))) #+ facet_wrap(~cond)
  #figa <- figa + geom_jitter(position=position_jitter(width=100))
  figa <- figa + geom_line(size=0.2358491) + geom_point(shape=15,size=0.8)
  figa <- figa + scale_colour_manual(values=c(defblue,deforange,defgrey))
  figa <- figa + scale_fill_manual(values=c(defblue,deforange,defgrey))
  figa <- figa + geom_linerange(size=0.2358491)
  figa <- figa + scale_x_continuous(breaks=c(4,9,14))
#  figa <- figa + scale_y_continuous(limits=c(0.5,1.0))
  figa <- figa + mytheme
  figa <- figa + ylab("Reaction Time") + xlab("Eccentricity")
  figa <- figa + labs(title = txt_title, color = "Size Memory Array")
  figa <- figa + theme(legend.position = 'right')
  
  
  figa
  
  #--------------------------------------------------------------------------
}

