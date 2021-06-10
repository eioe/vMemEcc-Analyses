#--------------------------------------------------------------------------
# Is there an effect of workload or eccentricity on accuracy?
#
#--------------------------------------------------------------------------
#TODO: make function and return: summaries (aov, ttest), plot (figa)
#--------------------------------------------------------------------------
# ... confidence interval for repeated measures design - based on Coussineau Morey

func_analysis_01 <- function(condition) {
  
  c1.aov <- data_behav %>% 
    filter(BlockStyle == condition) %>% 
    group_by(ppid, c_StimN, c_Ecc) %>%   
    summarise(meanAcc = mean(c_ResponseCorrect)) %>% 
    ungroup() %>% 
    select("meanAcc", "c_StimN", "c_Ecc", "ppid") 
  
  c1.reduced <- c1.aov %>% 
    pivot_wider(names_from = c(c_StimN, c_Ecc), 
                values_from = meanAcc, 
                names_prefix = 'cond_') %>% 
    select(., contains('cond_'))
    
  ci_cm <- cm.ci(data.frame=c1.reduced,
                 conf.level=2*(pnorm(1,0,1)-0.5),
                 difference=TRUE) #1sem or 1.96sem
  
  #--------------------------------------------------------------------------
  ## Run ANOVA
  aov.srt <- aov(meanAcc ~ c_StimN*c_Ecc + Error(ppid/(c_StimN* c_Ecc)),data=c1.aov)
  
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
  res_ttest <- c1.aov %>% 
    group_by(ppid, c_Ecc) %>% 
    summarise(meanAcc = mean(meanAcc)) %>% 
    ungroup() %>% 
    pairwise_t_test(
      meanAcc ~ c_Ecc, paired = TRUE, 
      p.adjust.method = "bonferroni"
    )

  print_header(str_c('Results post-hoc t test\ntask: ', condition))
  print(res_ttest)
  #--------------------------------------------------------------------------
  ## Plot
  
  c1.plt <- c1.aov %>% 
    group_by(c_StimN, c_Ecc) %>% 
    summarise(meanAcc = mean(meanAcc)) %>% 
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
                     y = meanAcc,
                     ymin = lower,
                     ymax = upper, 
                     colour = as_factor(c_StimN))) #+ facet_wrap(~cond)
  #figa <- figa + geom_jitter(position=position_jitter(width=100))
  figa <- figa + geom_line(size=0.2358491) + geom_point(shape=15,size=0.8)
  figa <- figa + scale_colour_manual(values=c(defblue,deforange,defgrey))
  figa <- figa + scale_fill_manual(values=c(defblue,deforange,defgrey))
  figa <- figa + geom_linerange(size=0.2358491)
  figa <- figa + scale_x_continuous(breaks=c(4,9,14))
  figa <- figa + scale_y_continuous(limits=c(0.5,1.0))
  figa <- figa + mytheme
  figa <- figa + ylab("Proportion correct") + xlab("Eccentricity")
  figa <- figa + labs(title = txt_title, color = "Size Memory Array")
  figa <- figa + theme(legend.position = c(0.85, 0.15))
  
  
  figa

  #--------------------------------------------------------------------------
}


#--------------------------------------------------------------------------
# Is there an interaction effect of trial type (VSTM vs Perception) with 
# load or eccentricity on accuracy?
#
#--------------------------------------------------------------------------


func_analysis_01a <- function(dep_variable) {
  
  
  ## Select relevant data:
  data_filtered <- data_behav %>% 
    select(ppid, c_StimN, c_Ecc, c_ResponseCorrect, BlockStyle) %>% 
    mutate(c_Ecc = as_factor(c_Ecc), 
           c_BlockStyle = as_factor(BlockStyle)) 
  
  # Define contrast - baseline is Eccentricity = 9° and MemLoad = 2 and BlockStyle = experiment
  contrasts(data_filtered$c_Ecc) <- contr.treatment(3,base=2)
  data_filtered$load_contrast <- ifelse(data_filtered$c_StimN==2,1,0)
  data_filtered$BS_contrast <- ifelse(data_filtered$BlockStyle=='experiment',0,1)
  
  
  #--------------------------------------------------------------------------
  ## run glmer
  
  glmer.family = ifelse(dep_variable == 'c_ResponseCorrect', 
                        'binomial', 
                        'gaussian')
  
  m1 <- glmer(c_ResponseCorrect ~ load_contrast*c_Ecc*BS_contrast + (1|ppid), 
              data = data_filtered, 
              family = "binomial")
  
  print_header(str_c('Summary glmer\n', 
                     dep_variable, ' ~ MemLoad * Eccentricity * BlockStyle\n'))
  print(summary(m1))
  print(Anova(m1))
  
  
  
  
  
  ## Plot it:
  
  c1.aov <- data_behav %>% 
    mutate(BlockStyle = recode(BlockStyle, 'experiment' = 'memory')) %>% 
    group_by(ppid, c_StimN, c_Ecc, BlockStyle) %>%   
    summarise(meanAcc = mean(c_ResponseCorrect)) %>% 
    ungroup() %>% 
    select("meanAcc", "c_StimN", "c_Ecc", "ppid", "BlockStyle") 
  
  c1.reduced <- c1.aov %>% 
    pivot_wider(names_from = c(c_StimN, c_Ecc, BlockStyle), 
                values_from = meanAcc, 
                names_prefix = 'cond_') %>% 
    select(., contains('cond_'))
  
  ci_cm <- cm.ci(data.frame=c1.reduced,
                 conf.level=2*(pnorm(1,0,1)-0.5),
                 difference=TRUE) #1sem or 1.96sem
  
  
  c1.plt <- c1.aov %>% 
    mutate(c_Ecc = as.numeric(as.character(c_Ecc))) %>% 
    group_by(c_StimN, c_Ecc, BlockStyle) %>% 
    summarise(meanAcc = mean(meanAcc)) %>% 
    mutate(cond = str_c('cond', c_StimN, c_Ecc, BlockStyle, sep = '_')) %>% 
    left_join(as_tibble(ci_cm) %>% 
                add_column(cond = rownames(ci_cm)), 
              by = 'cond')
  
  txt_title <- 'Task comparison'
  
  figa <- ggplot(c1.plt, 
                 aes(x = c_Ecc, 
                     y = meanAcc,
                     ymin = lower,
                     ymax = upper, 
                     colour = as_factor(c_StimN))) + facet_wrap(~BlockStyle)
  #figa <- figa + geom_jitter(position=position_jitter(width=100))
  figa <- figa + geom_line(size=0.2358491) + geom_point(shape=15,size=0.8)
  figa <- figa + scale_colour_manual(values=c(defblue,deforange,defgrey))
  figa <- figa + scale_fill_manual(values=c(defblue,deforange,defgrey))
  figa <- figa + geom_linerange(size=0.2358491)
  figa <- figa + scale_x_continuous(breaks=c(4,9,14))
  figa <- figa + scale_y_continuous(limits=c(0.5,1.0))
  figa <- figa + mytheme
  figa <- figa + ylab("Proportion correct") + xlab("Eccentricity")
  figa <- figa + labs(title = txt_title, color = "Array Size")
  figa <- figa + theme(legend.position = c(0.85, 0.15))
  
  
  figa
  
  
  
}
