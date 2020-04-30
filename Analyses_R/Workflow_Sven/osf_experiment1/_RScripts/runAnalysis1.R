#--------------------------------------------------------------------------
# Is there an effect for manual reaction times
#
# optional: add Bayes Factors
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# ... confidence interval for repeated measures design - based on Coussineau Morey
m1 <- melt(data,id=c("testcond","vp"),measure=c("keyRT"))
c1 <- cast(m1,testcond  + vp ~ variable,mean)

c1$cond <-  as.factor((as.integer(c1$testcond)))
c1_reduced <- c1[,c("keyRT","cond")]
c1_reduced <- unstack(c1_reduced)

ci_cm <- cm.ci(data.frame=c1_reduced,conf.level=2*(pnorm(1,0,1)-0.5),difference=TRUE) #1sem or 1.96sem
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# Melt and reshape data
m2_manual <- melt(data,id=c("vp","testcond"),measure=c("keyRT"))
c2_manual <- cast(m2_manual,vp + testcond ~ variable,mean)

m3_manual <- melt(c2_manual,id=c("testcond"),measure=c("keyRT"))
c3_manual <- cast(m3_manual, testcond  ~ variable,mean)

c3_manual$upper <- ci_cm[,2]
c3_manual$lower <- ci_cm[,1]

m2_manual_diff <- melt(data,id=c("vp","testcond"),measure=c("keyRT"))
c2_manual_diff <- cast(m2_manual_diff,vp+ testcond ~ variable,mean)

m3_manual_diff <- melt(c2_manual_diff,id=c("testcond"),measure=c("keyRT"))
c3_manual_diff <- cast(m3_manual_diff,testcond ~ variable,mean)
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# Report t-test
t.test(c2_manual$keyRT[which(c2_manual$testcond==1)],c2_manual$keyRT[which(c2_manual$testcond==2)],paired=TRUE)

#c2_manual.aov             <- c2_manual
#c2_manual.aov$vp          <- as.factor(c2_manual.aov$vp)
#c2_manual.aov$cueTim    <- as.factor(c2_manual.aov$cueTim)
#c2_manual.aov$testcond        <- as.factor(c2_manual.aov$testcond)

## Run ANOVA
#aov.manual 		       <- aov(keyRT ~ testcond*cueTim + Error(vp/(testcond* cueTim)),data=c2_manual.aov)
#summary(aov.manual)
#--------------------------------------------------------------------------
