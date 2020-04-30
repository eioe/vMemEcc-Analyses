#--------------------------------------------------------------------------
# Run main script for analysis for:
#  vMemEcc
#--------------------------------------------------------------------------

library(here)

#--------------------------------------------------------------------------
# Define pathes
path_global 	    <- here('../..')
path_r_data       <- file.path(path_global, 'Data/DataR')
path_scripts_sven <- file.path(here('Workflow_Sven', 
                                    'osf_experiment1', 
                                    '_RScripts'))

#--------------------------------------------------------------------------
## load packages
source(here("Utils", "load_packages.R"))
source(here('Utils', 'print_output.R'))
source(file.path(path_scripts_sven,"loadPackages.R"))

#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
## define colors
source(file.path(path_scripts_sven, "loadColors.R"))
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
## load function to compute confidence intervals
source(file.path(path_scripts_sven,"loadFunctions.R"))
source(file.path(here('Utils', 'load_functions.R')))
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
## load theme
source(file.path(path_scripts_sven, "loadTheme.R"))
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
## load data, assign names for columns and define variable type
source(file.path(here('Utils', 'load_data.R')))
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
## Run analysis 01: Is there an effect of workload or eccentricity on 
##                  accuracy in the VSTM task and in the perception task?
##                  ANOVA & post hoc t test

# VSTM task:
func_analysis_01('experiment')

# perceptual Change Detection Task:
func_analysis_01('perception')


#--------------------------------------------------------------------------
## Run analysis 02: The influence of Memory Load and Eccentricity on 
##                  change detection performance:
##                  glmer

# VSTM task:
func_analysis_02('experiment', 'c_ResponseCorrect')

# perceptual Change Detection Task:
func_analysis_02('perception', 'c_ResponseCorrect')


#--------------------------------------------------------------------------
## Run analysis 03: Is there an effect of workload or eccentricity on 
##                  RT in the VSTM task and in the perception task?
##                  ANOVA & post hoc t test

# VSTM task:
func_analysis_03('experiment')

# perceptual Change Detection Task:
func_analysis_03('perception')


#--------------------------------------------------------------------------
## Run analysis 04: The influence of Memory Load and Eccentricity on 
##                  reaction time:
##                  glmer

# VSTM task:
func_analysis_04('experiment', 'c_ResponseTime')

# perceptual Change Detection Task:
func_analysis_04('perception', 'c_ResponseTime')


#--------------------------------------------------------------------------
## Run analysis 05: Is there an effect of workload or eccentricity on 
##                  mean CDA amplitude in the VSTM task?
##                  ANOVA & post hoc t test

func_analysis_05()


#--------------------------------------------------------------------------
## Run analysis 06: Is there an effect of workload or eccentricity on 
##                  mean alpha power in the VSTM task?
##                  ANOVA & post hoc t test

func_analysis_06()
