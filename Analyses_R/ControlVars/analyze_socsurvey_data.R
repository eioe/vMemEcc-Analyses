
library(tidyverse)


dat <- read_tsv('C:/Users/Felix/Downloads/rdata_vmemecc_2020-05-19_23-15.csv')

dat <- dat %>%  select(contains('GI')) %>% 
  filter(str_detect(.$GI12_01, 'VME_S'), !is.na(GI01)) %>% 
  filter(!(.$GI12_01 %in% c('VME_S19', 'VME_S11', 'VME_S14')))

dat %>%  group_by(GI01) %>% summarise(n = n(), max_age = max(GI04_01), min_age = min(GI04_01))  

         