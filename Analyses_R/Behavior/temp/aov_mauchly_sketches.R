

## Sketchbook for preliminary analysis tryouts:
# no standalone >>> needs data from fulldat_behav




## Mauchly test for sphericity:

wf <- sumstat %>% select(ppid, c_Ecc, mean_acc, c_StimN) %>% 
  pivot_wider(names_from = c(c_Ecc), values_from = mean_acc) %>% 
  select('4', '9', '14')

acc <- tibble(.rows = length(levels(sumstat$ppid)))
i <- 0
#for (aa in levels(sumstat$c_StimN)) {
for (bb in levels(sumstat$c_Ecc)) {
  i = i+1
  dd <- sumstat$mean_acc[sumstat$c_Ecc == bb & sumstat$c_StimN == aa] 
  acc[, i] <- dd
}
#}
# make mlm object:
mlm <- lm(as.matrix(wf)~1)
# test it:
mauchly.test(mlm, x = ~ 1)

summary(aov(mean_acc ~ c_StimN * c_Ecc + Error(ppid/(c_StimN * c_Ecc)), data = sumstat))

with(sumstat, pairwise.t.test(mean_acc, c_Ecc))