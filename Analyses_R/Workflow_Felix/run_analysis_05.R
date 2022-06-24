#--------------------------------------------------------------------------
# Analysis 05:
# Is there an effect of workload or eccentricity on mean CDA amplitude?
#
#--------------------------------------------------------------------------
# Authors: Sven Ohl & Felix Klotzsche, 2020
#--------------------------------------------------------------------------
# ... confidence interval for repeated measures design - based on Coussineau Morey

func_analysis_05 <- function(dep_variable = "CDA_amp_clustertimes") {
  
  
  condition <- "experiment"  #Only EEG data for VSTM task. 

  c1.aov <- data_behav %>% 
    # drop_na() %>% 
    mutate(c_Ecc = as_factor(c_Ecc)) %>% 
    filter(BlockStyle == condition) %>% 
    group_by(ppid, c_StimN, c_Ecc) %>%   
    summarise(across(dep_variable, ~ mean(.x, na.rm = T), .names = c('meanCDA'))) %>% 
    ungroup() %>% 
    select("meanCDA", "c_StimN", "c_Ecc", "ppid") 
  
  c1.reduced <- c1.aov %>% 
    pivot_wider(names_from = c(c_StimN, c_Ecc), 
                values_from = meanCDA, 
                names_prefix = 'cond_') %>% 
    select(., contains('cond_')) %>% 
    drop_na()
  
  ci_cm <- cm.ci(data.frame=c1.reduced,
                 conf.level=2*(pnorm(1,0,1)-0.5),
                 difference=TRUE) #1sem or 1.96sem
  
  cda_overall_summary <- data_behav %>% 
                            mutate(c_Ecc = as_factor(c_Ecc)) %>% 
                            filter(BlockStyle == condition) %>% 
                            drop_na() %>% 
                            group_by(ppid) %>% 
                            summarise(across(CDA_amp_clustertimes, ~ mean(.x, na.rm = T), .names = c('meanCDA'))) %>%
                            ungroup() %>% 
                            summarise(meanCDA_mean = mean(meanCDA), meanCDA_sd = sd(meanCDA), meanCDA_ci95lower = ci95lower(meanCDA), meanCDA_ci95upper = ci95upper(meanCDA))
  
  extract_var("cda_sign_cluster_meanamp_mean", cda_overall_summary$meanCDA_mean)
  extract_var("cda_sign_cluster_meanamp_sd", cda_overall_summary$meanCDA_sd)
  extract_var("cda_sign_cluster_meanamp_cilower", cda_overall_summary$meanCDA_ci95lower)
  extract_var("cda_sign_cluster_meanamp_ciupper", cda_overall_summary$meanCDA_ci95upper)
  
  cda_summary_stimN <- data_behav %>% 
    mutate(c_Ecc = as_factor(c_Ecc)) %>% 
    filter(BlockStyle == condition) %>% 
    group_by(ppid, c_StimN) %>% 
    summarise(across(dep_variable, ~ mean(.x, na.rm = T), .names = c('meanCDA'))) %>%
    group_by(c_StimN) %>% 
    summarise(meanCDA_mean = mean(meanCDA), meanCDA_sd = sd(meanCDA))
  
  extract_var("cda_sign_cluster_meanamp_StimN_2_mean", cda_summary_stimN$meanCDA_mean[cda_summary_stimN$c_StimN == 2])
  extract_var("cda_sign_cluster_meanamp_StimN_2_sd", cda_summary_stimN$meanCDA_sd[cda_summary_stimN$c_StimN == 2])
  extract_var("cda_sign_cluster_meanamp_StimN_4_mean", cda_summary_stimN$meanCDA_mean[cda_summary_stimN$c_StimN == 4])
  extract_var("cda_sign_cluster_meanamp_StimN_4_sd", cda_summary_stimN$meanCDA_sd[cda_summary_stimN$c_StimN == 4])
  
  #--------------------------------------------------------------------------
  ## Run ANOVA
  aov.srt <- aov(meanCDA ~ c_StimN*c_Ecc + Error(ppid/(c_StimN* c_Ecc)),data=c1.aov)
  
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
    group_by(ppid, c_Ecc) %>% 
    summarise(meanCDA = mean(meanCDA)) %>% 
    ungroup() %>% 
    pairwise_t_test(
      meanCDA ~ c_Ecc, paired = TRUE, 
      p.adjust.method = "bonferroni",
      detailed = TRUE
    )
  
  print_header(str_c('Results post-hoc t test\ntask: ', condition))
  print('Contrast between the eccentricity conditions:\n')
  print(res_ttest)
  
  # interaction
  res_ttest_perEcc <- c1.aov %>% 
    group_by(ppid, c_Ecc, c_StimN) %>%
    summarise(meanCDA = mean(meanCDA)) %>% 

    group_by(c_Ecc) %>% 
    pairwise_t_test(
      meanCDA ~ c_StimN, paired = TRUE, 
      p.adjust.method = "bonferroni",
      detailed = TRUE
    )
  print('Contrast between MemoryLoad conditions per Eccentricity:\n')
  print(res_ttest_perEcc)
  
  extract_var("cda_exp_Ecc_4_StimN2vsStimN4_t", res_ttest_perEcc$statistic[res_ttest_perEcc$c_Ecc == 4])
  extract_var("cda_exp_Ecc_9_StimN2vsStimN4_t", res_ttest_perEcc$statistic[res_ttest_perEcc$c_Ecc == 9])
  extract_var("cda_exp_Ecc_14_StimN2vsStimN4_t", res_ttest_perEcc$statistic[res_ttest_perEcc$c_Ecc == 14])
  extract_var("cda_exp_Ecc_4_StimN2vsStimN4_p", res_ttest_perEcc$p[res_ttest_perEcc$c_Ecc == 4])
  extract_var("cda_exp_Ecc_9_StimN2vsStimN4_p", res_ttest_perEcc$p[res_ttest_perEcc$c_Ecc == 9])
  extract_var("cda_exp_Ecc_14_StimN2vsStimN4_p", res_ttest_perEcc$p[res_ttest_perEcc$c_Ecc == 14])
  extract_var("cda_exp_Ecc_4_StimN2vsStimN4_df", res_ttest_perEcc$df[res_ttest_perEcc$c_Ecc == 4])
  extract_var("cda_exp_Ecc_9_StimN2vsStimN4_df", res_ttest_perEcc$df[res_ttest_perEcc$c_Ecc == 9])
  extract_var("cda_exp_Ecc_14_StimN2vsStimN4_df", res_ttest_perEcc$df[res_ttest_perEcc$c_Ecc == 14])
  extract_var("cda_exp_Ecc_4_StimN2vsStimN4_diff", res_ttest_perEcc$estimate[res_ttest_perEcc$c_Ecc == 4])
  extract_var("cda_exp_Ecc_9_StimN2vsStimN4_diff", res_ttest_perEcc$estimate[res_ttest_perEcc$c_Ecc == 9])
  extract_var("cda_exp_Ecc_14_StimN2vsStimN4_diff", res_ttest_perEcc$estimate[res_ttest_perEcc$c_Ecc == 14])
  extract_var("cda_exp_Ecc_4_StimN2vsStimN4_ci95upper", res_ttest_perEcc$conf.high[res_ttest_perEcc$c_Ecc == 4])
  extract_var("cda_exp_Ecc_9_StimN2vsStimN4_ci95upper", res_ttest_perEcc$conf.high[res_ttest_perEcc$c_Ecc == 9])
  extract_var("cda_exp_Ecc_14_StimN2vsStimN4_ci95upper", res_ttest_perEcc$conf.high[res_ttest_perEcc$c_Ecc == 14])
  extract_var("cda_exp_Ecc_4_StimN2vsStimN4_ci95lower", res_ttest_perEcc$conf.low[res_ttest_perEcc$c_Ecc == 4])
  extract_var("cda_exp_Ecc_9_StimN2vsStimN4_ci95lower", res_ttest_perEcc$conf.low[res_ttest_perEcc$c_Ecc == 9])
  extract_var("cda_exp_Ecc_14_StimN2vsStimN4_ci95lower", res_ttest_perEcc$conf.low[res_ttest_perEcc$c_Ecc == 14])
  
  #--------------------------------------------------------------------------
  ## Plot
  
  c1.plt <- c1.aov %>% 
    group_by(c_StimN, c_Ecc) %>% 
    summarise(meanCDA = mean(meanCDA, na.rm = T)) %>% 
    mutate(cond = str_c('cond', c_StimN, c_Ecc, sep = '_'), 
           c_Ecc = as.numeric(as.character(c_Ecc))) %>% 
    left_join(as_tibble(ci_cm) %>% 
                add_column(cond = rownames(ci_cm)), 
              by = 'cond')
  
  txt_title <- ''
  
  figa <- ggplot(c1.plt, 
                 aes(x = c_Ecc, 
                     y = meanCDA,
                     ymin = lower,
                     ymax = upper, 
                     colour = as_factor(c_StimN))) #+ facet_wrap(~cond)
  #figa <- figa + geom_jitter(position=position_jitter(width=100))
  figa <- figa + geom_line(size=0.2358491) + geom_point(shape=15,size=0.8)
  figa <- figa + scale_colour_manual(values=c(col_LoadLow, col_LoadHigh))#deforange,defblue,defgrey))
  figa <- figa + scale_fill_manual(values=c(defblue,deforange,defgrey))
  figa <- figa + geom_linerange(size=0.2358491)
  figa <- figa + scale_x_continuous(breaks=c(4,9,14))
  figa <- figa + scale_y_continuous(limits=c(-1.0, 0))
  figa <- figa + mytheme
  figa <- figa + ylab("mean CDA amplitude (uV)") + xlab("Eccentricity")
  figa <- figa + labs(title = txt_title, color = "Size Memory Array")
  figa <- figa + theme(legend.position = c(0.85, 0.15))
  
  figa
  
  #--------------------------------------------------------------------------
  return(aov.srt)
}

