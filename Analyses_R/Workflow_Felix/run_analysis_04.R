#--------------------------------------------------------------------------
# The influence of Memory Load and Eccentricity on reaction time
#
#--------------------------------------------------------------------------
# Set parameters:
#
# condition: 
#     experiment = VSTM change detection task (overall: 720 trials)
#     perception = perceptual change detection task (overall: 72 trials)
# dependent variable (dv): 
#     c_ResponseCorrect
#     CDA_amp = mean amplitude CDA
#     alpha_pwr = mean alpha power

#--------------------------------------------------------------------------

func_analysis_04 <- function(condition, dep_variable) {
  ## Select relevant data:
  data_filtered <- data_behav %>% 
    filter(BlockStyle == condition) %>% 
    select(ppid, c_StimN, c_Ecc, dep_variable) %>% 
    mutate(c_Ecc = as_factor(c_Ecc))
  
  # Define contrast - baseline is Eccentricity = 9° and MemLoad = 2
  contrasts(data_filtered$c_Ecc) <- contr.treatment(3,base=2)
  
  #--------------------------------------------------------------------------
  ## run glmer
  
  glmer.family = ifelse(dep_variable == 'c_ResponseCorrect', 
                        'binomial', 
                        'gaussian')
  
  m1 <- lmer(c_ResponseTime ~ c_StimN*c_Ecc + (1|ppid), 
              data = data_filtered)
  print_header(str_c('Summary glmer\n', 
                     dep_variable, ' ~ MemLoad * Eccentricity\n', 
                     'task: ', condition))
  summary(m1)
  anova(m1)
}