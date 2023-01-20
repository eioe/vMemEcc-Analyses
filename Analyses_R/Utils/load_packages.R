#--------------------------------------------------------------------------
library(tidyverse)
library(here)
library(Hmisc)
library(ggpubr)
library(rstatix)
library(lmerTest)
library(rjson)
library(corrplot)

library(parallel)
library(ggplot2)
library(reshape)
library(pracma)
library(grid)
library(ez)				# includes Greenhouse Geisser correction for ANOVA
#library(BayesFactor)
#library(viridis)
library(lme4)
library(nlme)
#library(patchwork)


source(here("Utils", "wrangle_data.R"))
source(here("Utils", "plot_behavior.R"))