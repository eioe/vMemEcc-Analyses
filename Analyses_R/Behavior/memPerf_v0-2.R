

library(tidyverse)

dPath <- "D:/Felix/Seafile/Experiments/vMemEcc/Data/PilotData/P02/Unity/S001"
dat <- read_csv(file.path(dPath, "trial_results.csv"))

d2Path <- "D:/Felix/Seafile/Experiments/vMemEcc/Data/PilotData"

pilots <- c("P01/Unity/S001", "P01/Unity/S002", 
            "P02/Unity/S001", 
            "Micha/P01/S001", 
            "Sven")

datP01_1 <- read_csv(file.path(d2Path, pilots[1], "trial_results.csv"))
datP01_2 <- read_csv(file.path(d2Path, pilots[2], "trial_results.csv"))

datP01 <- bind_rows(datP01_1, datP01_2) %>% mutate(ppid = "P01")

datP02 <- read_csv(file.path(d2Path, pilots[3], "trial_results.csv"))
datP02 <- mutate(datP02, ppid = "P02")

datMicha <- read_csv(file.path(d2Path, pilots[4], "trial_results.csv"))
datMicha <- convCortMagFac(datMicha) %>% mutate(ppid = "P001", 
                                                BlockStyle = "experiment")

datSven <- read_csv(file.path(d2Path, pilots[5], "trial_results.csv"))
datSven <- convCortMagFac(datSven) %>% mutate(ppid = "P002", 
                                              BlockStyle = "experiment")

fulldat <- bind_rows(datP01, datP02, datMicha, datSven)
fulldat <- fulldat %>% 
             mutate_at(vars(ppid, 
                            c_Ecc,
                            c_CortMag,
                            c_StimN
                       ), 
                       factor) %>% 
  
             mutate_at(vars(c_ResponseCorrect), 
                       as.logical) %>% 
          
             drop_na(c_Response) %>% 
              
             filter(c_CortMag == 0) %>% 
             
             mutate(c_ResponseCorrect = (c_Response == c_Change))



plot_performance(fulldat, "perception")

mean_error_glob <- mean_error %>% 
                            group_by(c_StimN, c_Ecc) %>% 
                            summarise(meanErr = mean(meanErr))

df <- fulldat
  style <- "experiment"                                                                 


plot_performance <- function(df, style) {
  
  df_mean_err <- df %>%
    filter(BlockStyle == style) %>%
    group_by(c_StimN, c_Ecc, ppid) %>%
    summarise(meanErr = mean(c_ResponseCorrect,
                             na.rm = TRUE), 
              sumTrials = n())
  
  df_mean_err_tot <- df_mean_err %>%
    group_by(c_StimN, c_Ecc) %>%
    summarise(meanErr = mean(meanErr))
  
  ggplot(df_mean_err, aes(x = c_Ecc,
                          y = meanErr)) +
    geom_point(aes(
      col = factor(ppid),
      shape = factor(c_StimN),
      fill = ppid
    ),
    size = 3,
    alpha = 0.5) +
    geom_line(
      data = subset(df_mean_err, c_StimN == "2"),
      aes(col = factor(ppid),
          group = ppid),
      linetype = 1,
      alpha = 0.3
    ) +
    geom_line(
      data = subset(df_mean_err, c_StimN == "4"),
      aes(col = factor(ppid),
          group = ppid),
      linetype = 3,
      alpha = 0.9
    ) +
    geom_point(data = df_mean_err_tot,
               aes(shape = factor(c_StimN)),
               size = 4,
               fill = NA) +
    geom_line(data = subset(df_mean_err_tot),
              aes(
                x = c_Ecc,
                y = meanErr,
                group = c_StimN,
                linetype = c_StimN
              )) +
    scale_shape_manual(values = c(21, 22)) +
    scale_fill_discrete(na.value = NA, guide = "none") +
    ylim(0, 1) +
    labs(x = "mean Eccentricity (dva)",
         y = "% correct",
         title = ifelse(style == "experiment",
                        "Visual memory task", 
                        "Change detection task"), 
         col = "subject", 
         linetype = "memory load", 
         shape = "memory load")

  printf("This code is based on %i trials in total.", 3)
  
}