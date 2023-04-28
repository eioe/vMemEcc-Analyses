#--------------------------------------------------------------------------
# Is there an effect of workload or eccentricity on d' (d prime)?
#
#--------------------------------------------------------------------------
# Authors: Sven Ohl & Felix Klotzsche, 2023
#--------------------------------------------------------------------------
# ... confidence interval for repeated measures design - based on Coussineau Morey

func_analysis_01.1 <- function(condition) {

  # Drop trials where we do not have EEG data:
  if (condition == "experiment") {
    data2analyze <- data_behav %>% 
      drop_na()
  } else {
    data2analyze <- data_behav
  }
  
  c1.aov <- data2analyze %>% 
    filter(BlockStyle == condition) %>% 
    group_by(ppid, c_StimN, c_Ecc) %>%   
    summarise(dprime = qnorm(mean(c(Hit, 0.5))) - qnorm(mean(c(FalseAlarm, 0.5)))) %>%  # applying the "log-linear" correction (e.g., Hautus, 1995; https://link.springer.com/article/10.3758/BF03203619)
    ungroup() %>% 
    select("dprime", "c_StimN", "c_Ecc", "ppid")
  
  c1.reduced <- c1.aov %>% 
    pivot_wider(names_from = c(c_StimN, c_Ecc), 
                values_from = dprime, 
                names_prefix = 'cond_') %>% 
    select(., contains('cond_')) 
    
  ci_cm <- cm.ci(data.frame=c1.reduced,
                 conf.level=2*(pnorm(1,0,1)-0.5),
                 difference=TRUE) #1sem or 1.96sem
  
  #--------------------------------------------------------------------------
  ## Run ANOVA
  aov.srt <- aov(dprime ~ c_StimN*c_Ecc + Error(ppid/(c_StimN* c_Ecc)),data=c1.aov)
  
  results = list()
  results[["aov.srt"]] <- aov.srt
  
  print_header(str_c('Summary ANOVA \ntask: ', condition))
  print(summary(aov.srt))
  
  ## rstatix alternative (identical results:)
  # aov.res <- anova_test(data = c1.aov, 
  #                       dv = meanAcc, 
  #                       wid = ppid, 
  #                       within = c(c_StimN, c_Ecc))
  # get_anova_table(aov.re  s)
  
  #--------------------------------------------------------------------------
  ## Run post-hoc t tests:
  
  # main effect Eccentricity:
  res_ttest <- data2analyze %>% 
    filter(BlockStyle == condition) %>% 
    group_by(ppid, c_Ecc) %>% 
    summarise(dprime = qnorm(mean(c(Hit, 0.5))) - qnorm(mean(c(FalseAlarm, 0.5)))) %>% 
    ungroup() %>% 
    pairwise_t_test(
      dprime ~ c_Ecc, paired = TRUE, 
      p.adjust.method = "bonferroni",
      detailed = TRUE
    )

  print_header(str_c('Results post-hoc t test\ntask: ', condition))
  print(res_ttest)
  results[["res_ttest"]] <- res_ttest
  

  means <- data2analyze %>%
    filter(BlockStyle == condition) %>%
    select(c_Ecc, c_StimN, ppid, Hit, FalseAlarm) %>%
    group_by(c_Ecc, c_StimN, ppid) %>%
    summarise(dprime = qnorm(mean(c(Hit, 0.5))) - qnorm(mean(c(FalseAlarm, 0.5)))) %>%
    pivot_wider(id_cols =  ppid, names_from = c_Ecc, values_from = dprime, values_fn = mean) %>%
    ungroup() %>%
    select(!ppid) %>%
    summarise_all(.funs = list(mean = mean, sd = sd))
  print(means)
  
  means <- data2analyze %>%
    filter(BlockStyle == condition) %>%
    select(c_Ecc, c_StimN, ppid, Hit, FalseAlarm) %>%
    group_by(c_StimN, c_Ecc, ppid) %>%
    summarise(dprime = qnorm(mean(c(Hit, 0.5))) - qnorm(mean(c(FalseAlarm, 0.5)))) %>%
    pivot_wider(id_cols =  ppid, names_from = c_StimN, values_from = dprime, values_fn = mean) %>%
    ungroup() %>%
    select(!ppid) %>%
    summarise_all(.funs = list(mean = mean, sd = sd)) 
  print(means)
  
  #--------------------------------------------------------------------------
  ## Plot
  
  c1.plt <- c1.aov %>% 
    group_by(c_StimN, c_Ecc) %>% 
    summarise(meandprime = mean(dprime)) %>% 
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
                     y = meandprime,
                     ymin = lower,
                     ymax = upper, 
                     colour = as_factor(c_StimN))) #+ facet_wrap(~cond)
  #figa <- figa + geom_jitter(position=position_jitter(width=100))
  figa <- figa + geom_line(size=0.2358491) + geom_point(shape=15,size=0.1)
  figa <- figa + scale_colour_manual(values=c(col_LoadLow, col_LoadHigh, defgrey))  # values=c(defblue,deforange,defgrey))
  figa <- figa + scale_fill_manual(values=c(defblue,deforange,defgrey))
  figa <- figa + geom_linerange(size=0.2358491)
  figa <- figa + scale_x_continuous(breaks=c(4,9,14))
  figa <- figa + scale_y_continuous(limits=c(0.0, 2.5))
  figa <- figa + mytheme
  figa <- figa + ylab("d'") + xlab("Eccentricity")
  # figa <- figa + labs(title = txt_title, color = "Size Memory Array")
  figa <- figa + theme(legend.position = c(1.85, 1.15))
  
  plot(figa)
  
  fname = file.path(path_global, 'Plots2022', 'Behavior', str_c('behav_dprime_perf_anova_', condition, '.pdf'))
  ggsave(plot = figa,   
         width = 7.7/2.54,
         height = 6.9/2.54,
         dpi = 300,
         filename = fname)


  #--------------------------------------------------------------------------
  
  return(results)
}

