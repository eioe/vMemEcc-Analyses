#--------------------------------------------------------------------------
# Plot spatial specificity
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# ... confidence interval for repeated measures design - based on Coussineau Morey
m1 <- melt(data,id=c("DistToProbe","cueTim","vp"),measure=c("answer"))
c1 <- cast(m1, DistToProbe + cueTim + vp ~ variable,mean)

c1$cond <-  as.factor(paste(as.integer(c1$DistToProbe),as.integer(c1$cueTim)))
c1_reduced <- c1[,c("answer","cond")]
c1_reduced <- unstack(c1_reduced)

ci_cm <- cm.ci(data.frame=c1_reduced,conf.level=2*(pnorm(1,0,1)-0.5),difference=TRUE) #1sem or 1.96 sem
ci_cm <- ci_cm[c(1,4,5,2,3,6,9,10,7,8,11,14,15,12,13,16,19,20,17,18,21,24,25,22,23),]
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# Melt and reshape
# prepare for bootstrapping
m1 <- melt(data,id=c("DistToProbe","cueTim","vp"),measure=c("answer"))
c1 <- cast(m1,DistToProbe + cueTim + vp ~ variable,mean)

# prepare for anova, bf, and plot
m1x <- melt(data,id=c("DistToProbe","cueTim","vp"),measure=c("answer"))
c1x <- cast(m1x,DistToProbe + cueTim + vp ~ variable,mean)

m2<- melt(c1x,id=c("DistToProbe","cueTim"),measure=c("answer"))
c2 <- cast(m2,DistToProbe + cueTim ~ variable,c(mean,length,sd))

# add errorbars
c2$upper <- ci_cm[,2]
c2$lower <- ci_cm[,1]
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# Compute 2 way parametric anova

# Prepare ANOVA
c2.aov             <- c1
c2.aov$vp          <- as.factor(c2.aov$vp)
c2.aov$cueTim      <- as.factor(c2.aov$cueTim)
c2.aov$DistToProbe <- as.factor(c2.aov$DistToProbe)
c2.aov$answer_transformed <- asin(sqrt(c2.aov$answer))

# Run ANOVA
aov.m 		       <- aov(answer ~ DistToProbe* cueTim + Error(vp/(DistToProbe* cueTim)),data=c2.aov)
summary(aov.m)

# post-hoc anovas
aov.m100             <- aov(answer ~ DistToProbe + Error(vp/(DistToProbe)),data=c2.aov[which(c2.aov$cueTim ==100),])
aov.m400             <- aov(answer ~ DistToProbe + Error(vp/(DistToProbe)),data=c2.aov[which(c2.aov$cueTim ==400),])
aov.m800             <- aov(answer ~ DistToProbe + Error(vp/(DistToProbe)),data=c2.aov[which(c2.aov$cueTim ==800),])
aov.m1600            <- aov(answer ~ DistToProbe + Error(vp/(DistToProbe)),data=c2.aov[which(c2.aov$cueTim ==1600),])
aov.m3200            <- aov(answer ~ DistToProbe + Error(vp/(DistToProbe)),data=c2.aov[which(c2.aov$cueTim ==3200),])

# post-hoc t-tests
t.test(c2.aov$answer[which(c2.aov$DistToProbe==0)],c2.aov$answer[which(c2.aov$DistToProbe==1)],paired=TRUE) # distance 0 vs. 1
t.test(c2.aov$answer[which(c2.aov$DistToProbe==1)],c2.aov$answer[which(c2.aov$DistToProbe==2)],paired=TRUE) # distance 1 vs. 2
t.test(c2.aov$answer[which(c2.aov$DistToProbe==1)],c2.aov$answer[which(c2.aov$DistToProbe==3)],paired=TRUE) # distance 1 vs. 3
t.test(c2.aov$answer[which(c2.aov$DistToProbe==1)],c2.aov$answer[which(c2.aov$DistToProbe==4)],paired=TRUE) # distance 1 vs. 4

t.test(c2.aov$answer[which(c2.aov$DistToProbe==0 & c2.aov$cueTim==400)],c2.aov$answer[which(c2.aov$DistToProbe==1 & c2.aov$cueTim==400)],paired=TRUE) # distance 0 vs. 1
t.test(c2.aov$answer[which(c2.aov$DistToProbe==1 & c2.aov$cueTim==400)],c2.aov$answer[which(c2.aov$DistToProbe==2 & c2.aov$cueTim==400)],paired=TRUE) # distance 1 vs. 2
t.test(c2.aov$answer[which(c2.aov$DistToProbe==1 & c2.aov$cueTim==400)],c2.aov$answer[which(c2.aov$DistToProbe==3 & c2.aov$cueTim==400)],paired=TRUE) # distance 1 vs. 2
t.test(c2.aov$answer[which(c2.aov$DistToProbe==1 & c2.aov$cueTim==400)],c2.aov$answer[which(c2.aov$DistToProbe==4 & c2.aov$cueTim==400)],paired=TRUE) # distance 1 vs. 2
t.test(c2.aov$answer[which(c2.aov$DistToProbe==0 & c2.aov$cueTim==800)],c2.aov$answer[which(c2.aov$DistToProbe==1 & c2.aov$cueTim==800)],paired=TRUE) # distance 1 vs. 2
t.test(c2.aov$answer[which(c2.aov$DistToProbe==1 & c2.aov$cueTim==800)],c2.aov$answer[which(c2.aov$DistToProbe==2 & c2.aov$cueTim==800)],paired=TRUE) # distance 1 vs. 2
t.test(c2.aov$answer[which(c2.aov$DistToProbe==1 & c2.aov$cueTim==800)],c2.aov$answer[which(c2.aov$DistToProbe==3 & c2.aov$cueTim==800)],paired=TRUE) # distance 1 vs. 2
t.test(c2.aov$answer[which(c2.aov$DistToProbe==1 & c2.aov$cueTim==800)],c2.aov$answer[which(c2.aov$DistToProbe==4 & c2.aov$cueTim==800)],paired=TRUE) # distance 1 vs. 2
t.test(c2.aov$answer[which(c2.aov$DistToProbe==0 & c2.aov$cueTim==1600)],c2.aov$answer[which(c2.aov$DistToProbe==1 & c2.aov$cueTim==1600)],paired=TRUE) # distance 1 vs. 2
t.test(c2.aov$answer[which(c2.aov$DistToProbe==1 & c2.aov$cueTim==1600)],c2.aov$answer[which(c2.aov$DistToProbe==2 & c2.aov$cueTim==1600)],paired=TRUE) # distance 1 vs. 2
t.test(c2.aov$answer[which(c2.aov$DistToProbe==1 & c2.aov$cueTim==1600)],c2.aov$answer[which(c2.aov$DistToProbe==3 & c2.aov$cueTim==1600)],paired=TRUE) # distance 1 vs. 2
t.test(c2.aov$answer[which(c2.aov$DistToProbe==1 & c2.aov$cueTim==1600)],c2.aov$answer[which(c2.aov$DistToProbe==4 & c2.aov$cueTim==1600)],paired=TRUE) # distance 1 vs. 2
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# Start plotting

# Define parameters for plot
c2$cueTim <- as.factor(c2$cueTim) 
c2$col_ind <- as.factor(c(1,1,1,1,1,rep(2,times=20)))

figb <- ggplot(c2,aes(x= DistToProbe,y=answer_mean,ymin=lower,ymax=upper,colour=col_ind))+facet_wrap(~ cueTim,nrow=1)
figb <- figb + geom_point(shape=15,size=0.8)
figb <- figb + geom_linerange(size=0.2358491)
figb <- figb + scale_colour_manual(values=c(defblue,deforange,deforange,deforange,deforange))
#figb <- figb + scale_color_viridis(discrete=TRUE)
figb <- figb + scale_fill_manual(values=c(defblue,deforange,deforange,deforange,deforange))
figb <- figb + geom_line(size=0.2358491,col=deforange)
figb <- figb + scale_y_continuous(limits=c(0.5,0.8),breaks=c(0.5,0.8))
figb <- figb + mytheme
figb <- figb + theme(strip.background=element_blank())
figb <- figb + theme(axis.title.x=element_blank())
figb <- figb + ylab("Proportion correct") + xlab("Distance to probe")
#--------------------------------------------------------------------------