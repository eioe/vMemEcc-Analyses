#--------------------------------------------------------------------------
# The influence of Memory Load and Eccentricity on CDA mean amplitude
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
# Authors: Sven Ohl & Felix Klotzsche, 2020
#--------------------------------------------------------------------------

func_analysis_07 <- function(condition, dep_variable) {
  
  condition <- 'experiment' # VSTM block only
  
  ## Select relevant data:
  data_filtered <- data_behav %>% 
    drop_na() %>% 
    filter(BlockStyle == condition) %>% 
    select(contains(c("ppid", "c_StimN", "c_Ecc", dep_variable))) %>% 
    mutate(c_Ecc = as_factor(c_Ecc))
  
  # Define contrast - baseline is Eccentricity = 9° and MemLoad = 2
  contrasts(data_filtered$c_Ecc) <- contr.treatment(3,base=2)
  data_filtered$load_contrast <- ifelse(data_filtered$c_StimN==2,1,0)
  
  #--------------------------------------------------------------------------
  ## run glmer
  
  glmer.family = ifelse(dep_variable == 'c_ResponseCorrect', 
                        'binomial', 
                        'gaussian')
  formula <- as.formula(str_c(dep_variable, '~ load_contrast*c_Ecc + (1|ppid)'))
  m1 <- lmer(formula, 
                data = data_filtered)
  
  print_header(str_c('Summary glmer\n', 
                     dep_variable, ' ~ MemLoad * Eccentricity\n', 
                     'task: ', condition))
  summary(m1)
}

