#--------------------------------------------------------------------------
# Is there an effect for saccade latency or saccade amplitude
#
# optional: add Bayes Factors
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# set dependent variable: "sacRT" vs. "sacAmp"
dv <- "sacRT"
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# can we use dapply for that to run faster?
mx0 <- melt(data,id=c("testcond","vp"),measure=c(dv))
cx0 <- cast(mx0,testcond + vp ~ variable,mean)
mx1 <- melt(cx0,id=c("testcond"),measure=c(dv))
cx1 <- cast(mx1,testcond ~ variable,mean)
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# ... confidence interval for repeated measures design - based on Coussineau Morey
m1 <- melt(data,id=c("testcond","cueTim","vp"),measure=c(dv))
c1 <- cast(m1,testcond + cueTim  + vp ~ variable,mean)

c1$cond <-  as.factor(paste(as.integer(c1$testcond),as.integer(c1$cueTim)))
c1_reduced <- c1[,c(dv,"cond")]
c1_reduced <- unstack(c1_reduced)

ci_cm <- cm.ci(data.frame=c1_reduced,conf.level=2*(pnorm(1,0,1)-0.5),difference=TRUE) #1sem or 1.96 sem
ci_cm <- ci_cm[c(1,4,5,2,3,6,9,10,7,8),]
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# Melt and reshape data
m2_srt <- melt(data,id=c("vp","testcond","cueTim"),measure=c(dv))
c2_srt <- cast(m2_srt,vp + testcond+cueTim ~ variable,mean)

m3_srt <- melt(c2_srt,id=c("testcond","cueTim"),measure=c(dv))
c3_srt <- cast(m3_srt, testcond + cueTim ~ variable,mean)

c3_srt$upper <- ci_cm[,2]
c3_srt$lower <- ci_cm[,1]

m2_srt_diff <- melt(data,id=c("vp","testcond"),measure=c(dv))
c2_srt_diff <- cast(m2_srt_diff, vp + testcond ~ variable,mean)

m3_srt_diff <- melt(c2_srt_diff,id=c("testcond"),measure=c(dv))
c3_srt_diff <- cast(m3_srt_diff, testcond ~ variable,mean)
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
#t.test(c2_manual$keyRT[which(c2_manual$cond=="congruent")],c2_manual$keyRT[which(c2_manual$cond=="incongruent")],paired=TRUE)
c2_srt.aov             <- c2_srt
c2_srt.aov$vp          <- as.factor(c2_srt.aov$vp)
c2_srt.aov$testcond    <- as.factor(c2_srt.aov$testcond)
c2_srt.aov$cueTim      <- as.factor(c2_srt.aov$cueTim)

# Run ANOVA
aov.srt 		       <- aov(sacRT ~ testcond*cueTim + Error(vp/(testcond* cueTim)),data=c2_srt.aov)
summary(aov.srt)
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# Report saccade latencies for different delays
num_srt <- melt(data,id=c("vp","cueTim"),measure=dv)
rep_srt <- cast(num_srt,vp + cueTim ~ variable,mean)

num2_srt <- melt(rep_srt,id=c("cueTim"),measure=dv)
rep2_srt <- cast(num2_srt,  cueTim ~ variable,mean)
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# Add Bayes Factor
#--------------------------------------------------------------------------
