library(tidyverse)
library(here)

dat <- read_csv(file.path("../../Data/PilotData/Micha/P01/S001/trial_results.csv"))

# Dirty shit to clean problem with dec. sperator: 
idx <- dat$c_Ecc == "9"
dat2 <- dat
dat2[idx, 10:17] <- dat[idx, 11:18] 
i2 <- dat2$c_Ecc == "5"
dat2[i2, "c_Ecc"] <- "9.5"
i3 <- dat2$c_Ecc == "True"
dat2[i3, 11:19] <- dat2[i3, 10:18]
dat2[i3, "c_Ecc"] <- '4'
dat2$c_Ecc <- as.numeric(dat2$c_Ecc)

res <- dat2 %>% 
  mutate(corRes = (c_Change == c_Response)) %>%
  group_by(c_CortMag, c_Ecc)

resres <- res %>% 
  summarise(ha = mean(corRes, na.rm = T))

ggplot(resres, aes(x=c_Ecc, y=ha, col=c_CortMag)) + 
  geom_point() + geom_line()