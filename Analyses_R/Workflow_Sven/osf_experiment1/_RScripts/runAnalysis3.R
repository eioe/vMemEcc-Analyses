#--------------------------------------------------------------------------
# ... confidence interval for repeated measures design - based on Coussineau Morey
m1 <- melt(data,id=c("testcond","cueTim","vp"),measure=c("answer"))
c1 <- cast(m1,testcond + cueTim + vp ~ variable,mean)

c1$cond <-  as.factor(paste(as.integer(c1$testcond),as.integer(c1$cueTim)))
c1_reduced <- c1[,c("answer","cond")]
c1_reduced <- unstack(c1_reduced)

ci_cm <- cm.ci(data.frame=c1_reduced,conf.level=2*(pnorm(1,0,1)-0.5),difference=TRUE) # 1sem or 1.96 sem
ci_cm <- ci_cm[c(1,4,5,2,3,6,9,10,7,8),]
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# Melt and reshape
m1x <- melt(data,id=c("testcond","cueTim","vp"),measure=c("answer"))
c1x <- cast(m1x, cueTim+testcond+vp ~ variable,mean)

m2	<- melt(c1x,id=c("testcond","cueTim"),measure=c("answer"))
c2	<- cast(m2,testcond + cueTim ~ variable,mean)

c2$upper <- ci_cm[,2]
c2$lower <- ci_cm[,1]
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# Prepare ANOVA
c2.aov             <- c1x
c2.aov$vp          <- as.factor(c2.aov$vp)
c2.aov$cueTim      <- as.factor(c2.aov$cueTim)
c2.aov$testcond    <- as.factor(c2.aov$testcond)

c2.aov$answer_transformed <- asin(sqrt(c1x$answer))

# Run ANOVA
aov.m 		       <- aov(answer ~ testcond* cueTim + Error(vp/(testcond* cueTim)),data=c2.aov)
summary(aov.m)

# Run post-hoc t-tests
t.test(c2.aov$answer[which(c2.aov$testcond==1 & c2.aov$cueTim==100)],c2.aov$answer[which(c2.aov$testcond==2 & c2.aov$cueTim==100)],paired=TRUE) 
t.test(c2.aov$answer[which(c2.aov$testcond==1 & c2.aov$cueTim==400)],c2.aov$answer[which(c2.aov$testcond==2 & c2.aov$cueTim==400)],paired=TRUE)
t.test(c2.aov$answer[which(c2.aov$testcond==1 & c2.aov$cueTim==800)],c2.aov$answer[which(c2.aov$testcond==2 & c2.aov$cueTim==800)],paired=TRUE) 
t.test(c2.aov$answer[which(c2.aov$testcond==1 & c2.aov$cueTim==1600)],c2.aov$answer[which(c2.aov$testcond==2 & c2.aov$cueTim==1600)],paired=TRUE) 
t.test(c2.aov$answer[which(c2.aov$testcond==1 & c2.aov$cueTim==3200)],c2.aov$answer[which(c2.aov$testcond==2 & c2.aov$cueTim==3200)],paired=TRUE)
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# Plot
c2$testcond <- as.factor(c2$testcond)

figa <- ggplot(c2,aes(x=cueTim,y=answer,ymin=lower,ymax=upper,colour=testcond)) #+ facet_wrap(~cond)
#figa <- figa + geom_jitter(position=position_jitter(width=100))
figa <- figa + geom_line(size=0.2358491) + geom_point(shape=15,size=0.8)
figa <- figa + scale_colour_manual(values=c(defblue,deforange,defgrey))
figa <- figa + scale_fill_manual(values=c(defblue,deforange,defgrey))
figa <- figa + geom_linerange(size=0.2358491)
figa <- figa + scale_x_continuous(breaks=c(100,400,800,1600,3200))
figa <- figa + scale_y_continuous(limits=c(0.5,0.8))
figa <- figa + mytheme
figa <- figa + ylab("Proportion correct") + xlab("Set size")
#--------------------------------------------------------------------------