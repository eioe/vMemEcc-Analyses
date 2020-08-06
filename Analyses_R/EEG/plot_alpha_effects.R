
library(tidyverse)

##: for Stimon:

path_norm <- 'C:/Users/Felix/Seafile/Experiments/vMemEcc/Data/DataMNE/EEG/08_tfr/summaries/timeseries/stimon/averages' 
path_indu <- 'C:/Users/Felix/Seafile/Experiments/vMemEcc/Data/DataMNE/EEG/08_tfr/summaries/induc/timeseries/stimon/averages'

l_norm <- NULL
l_indu <- NULL

for (f in list.files(path_norm)) {

  subID <- str_split(f, '_cond')[[1]][1]
  df <- read_csv2(file.path(path_norm, f), 
                  col_types = cols(.default = col_character())) %>% 
    mutate_all(as.numeric)
  df_indu <- read_csv2(file.path(path_indu, f), 
                       col_types = cols(.default = col_character())) %>% 
    mutate_all(as.numeric)
  l_norm[[subID]] <- df
  l_indu[[subID]] <- df_indu
}
                       
df_norm <- bind_rows(l_indu, .id = 'subID')

cols <- RColorBrewer::brewer.pal(10, 'Spectral')
cols_load <- cols[8:9]
cols_ecc <- cols[c(1,3,5)]

# Plot: Main effect Load:
          
df_norm %>%
  pivot_longer(cols = c(LoadLow, LoadHigh), names_to = 'Condition') %>% 
  group_by(time, Condition) %>% 
  summarize_all(mean)  %>%  
  #filter(subID == 'VME_S08') %>% 
  ggplot(aes(x=as.numeric(time), 
             y=value, 
             color=Condition)) +
  geom_line() + 
  theme_classic() + 
  xlab('Time (s)') + 
  ylab('Alpha power (abs)') + 
  geom_vline(xintercept = c(0,0.2, 2.3)) + 
  scale_color_manual(values = cols_load)


## Plot main eff Ecc:

df_norm %>%
  pivot_longer(cols = c(EccS, EccM, EccL), names_to = 'Condition') %>% 
  group_by(time, Condition) %>% 
  summarize_all(mean)  %>%  
  #filter(subID == 'VME_S08') %>% 
  ggplot(aes(x=as.numeric(time), 
             y=value, 
             color=Condition)) +
  geom_line() + 
  theme_classic() + 
  xlab('Time (s)') + 
  ylab('Alpha power (abs)') + 
  geom_vline(xintercept = c(0,0.2, 2.3)) + 
  scale_color_manual(values = cols_ecc)




stat_smooth(method="loess",
              span=0.3, se=TRUE, alpha=0.3)
  

stat_summary(geom='ribbon', fun.min='min',
               fun.max = 'max')
  geom_line() + 
  geom_line(data=df_indu, aes(x=as.numeric(time), y=as.numeric(all)), 
            col = 'red')

