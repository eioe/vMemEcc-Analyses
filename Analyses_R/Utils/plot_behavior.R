
# Author         : Felix Klotzsche    ---    eioe
# Date           : 11 July 2019

###############################################################################
#                                                                             #
# Plotters for behavioral data                                                              #
#                                                                             #
###############################################################################

c_StimN.labs <- c("Load Low", "Load High")
names(c_StimN.labs) <- c("2", "4")


plot_performance <- function(df, style) {
  
  df_mean_err <- df %>%
    filter(BlockStyle == style, 
           c_Ecc %in% c(4, 9, 14)) %>%
    group_by(c_StimN, c_Ecc, ppid) %>%
    summarise(meanErr = mean(c_ResponseCorrect,
                             na.rm = TRUE), 
              numTrials = n())
  
  df_mean_err_tot <- df_mean_err %>%
    group_by(c_StimN, c_Ecc) %>%
    summarise(meanErr = mean(meanErr))
  
  ggplot(df_mean_err, aes(x = c_Ecc,
                          y = meanErr)) +
    geom_jitter(width = 0.1, height = 0.0) + 
    # geom_dotplot(binaxis='y', 
    #              stackdir='center', 
    #              dotsize=0.6, 
    #              binwidth = 0.01, 
    #              method = 'histodot') +
    stat_summary(fun.data=mean_sdl, 
                 fun.args = list(mult=1), 
                 geom="pointrange", 
                 aes(color=c_StimN), 
                 size = 0.6) + 
    facet_wrap(~c_StimN, 
               labeller = labeller(c_StimN = c_StimN.labs)) +
    #ylim(c(0.5,1.05)) + 
    labs(x = "mean Eccentricity (dva)",
         y = "% correct (+/-1SD)",
         title = ifelse(style == "experiment",
                        "Visual memory task",
                        "Change detection task"),
         col = "Memory Items",
         linetype = "memory load",
         shape = "memory load") +
    theme_light() #+ 
    #coord_fixed(ratio = 3.5)  #+
  #scale_color_jco() 
  
  # ggplot(df_mean_err, aes(x = c_Ecc,
  #                         y = meanErr)) +
  #   geom_point(aes(
  #     col = factor(ppid),
  #     shape = factor(c_StimN),
  #     fill = ppid
  #   ),
  #   size = 2,
  #   alpha = 0.3) +
  #   geom_line(
  #     data = subset(df_mean_err, c_StimN == "2"),
  #     aes(col = factor(ppid),
  #         group = ppid),
  #     linetype = 1,
  #     alpha = 0.3
  #   ) +
  #   geom_line(
  #     data = subset(df_mean_err, c_StimN == "4"),
  #     aes(col = factor(ppid),
  #         group = ppid),
  #     linetype = 3,
  #     alpha = 0.9
  #   ) +
  #   geom_point(data = df_mean_err_tot,
  #              aes(shape = factor(c_StimN)),
  #              size = 4,
  #              fill = NA) +
  #   geom_line(data = subset(df_mean_err_tot),
  #             aes(
  #               x = c_Ecc,
  #               y = meanErr,
  #               group = c_StimN,
  #               linetype = c_StimN
  #             )) +
  #   scale_shape_manual(values = c(21, 22)) +
  #   scale_fill_discrete(na.value = NA, guide = "none") +
  #   ylim(0, 1) +
  #   labs(x = "mean Eccentricity (dva)",
  #        y = "% correct",
  #        title = ifelse(style == "experiment",
  #                       "Visual memory task", 
  #                       "Change detection task"), 
  #        col = "subject", 
  #        linetype = "memory load", 
  #        shape = "memory load") +
  #   annotate("text", x = 3, y = 0, label = paste("# of obs. total:", 
  #                                                 as.character(sum(df_mean_err$numTrials))))
  
  #sprintf("This code is based on %i trials in total.", sum(df_mean_err$numTrials))
}