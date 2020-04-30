#--------------------------------------------------------------------------
# runAnalysis6 
#
# Compare influence of srt in this experiment and in Ohl & Rolfs 2017
#
# testcond2 <- set contrast
# sacRT2    <- z-transformed sacRT for each participant
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# open pfd for saving
pdf(paste(path_wd,"/_Figures/Figure3.pdf",sep=""), width = 17.6/2.54, height = 9/2.54)
par(mfrow=c(1,2),ps=9)
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# merge data from two experiments
dx       <- data[which(data$cueTim==100),c("vp","testcond2","sacRT2","answer")]
dx$expt  <- 0

dx2      <- vstm2data[which(vstm2data$cueTim==100),c("vp","testcond2","sacRT2","answer")]
dx2$vp   <- as.factor(dx2$vp+10)
dx2$expt <- 1

dxmerge <- rbind(dx,dx2)

m1 <- glmer(answer~testcond2*sacRT2*expt + (1|vp),data=dxmerge,family="binomial")
summary(m1)
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# merge data from two experiments
dx       <- data[which(data$cueTim==400),c("vp","testcond2","sacRT2","answer")]
dx$expt  <- 0

dx2      <- vstm2data[which(vstm2data$cueTim==400),c("vp","testcond2","sacRT2","answer")]
dx2$vp   <- as.factor(dx2$vp+10)
dx2$expt <- 1

dxmerge <- rbind(dx,dx2)
m2 <- glmer(answer~testcond2*sacRT2*expt + (1|vp),data=dxmerge,family="binomial")
summary(m2)
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# plot predictions from the two models
tc2 <- c(0,1)
Exp <- c(0,0)
RT  <- seq(-0.1,0.2,length.out=10)
fef <- fixef(m1)
fef2 <- fixef(m2)

colvec <- c(deforange,defblue)
plot(0,0,type="n",bty="l",xlim=c(-0.1,0.2),ylim=c(0,1),xlab="Centered saccade latency [in s]",ylab="Proportion correct")
for(i in 1:2){
pred <- fef[[1]] +tc2[i]*fef[[2]] +RT*fef[[3]] +Exp[i]*fef[[4]] +tc2[i]*RT*fef[[5]] +tc2[i]*Exp[i]*fef[[6]] +Exp[i]*RT*fef[[7]] +tc2[i]*Exp[i]*RT*fef[[8]]
lines(RT,exp(pred)/(1+exp(pred)),col=colvec[i],lty=1)
pred2 <- fef2[[1]] +tc2[i]*fef2[[2]] +RT*fef2[[3]] +Exp[i]*fef2[[4]] +tc2[i]*RT*fef2[[5]] +tc2[i]*Exp[i]*fef2[[6]] +Exp[i]*RT*fef2[[7]] +tc2[i]*Exp[i]*RT*fef2[[8]]
lines(RT,exp(pred2)/(1+exp(pred2)),col=colvec[i],lty=2)
}

tc2 <- c(0,1)
Exp <- c(1,1)
RT  <- seq(-0.1,0.2,length.out=10)
fef <- fixef(m1)
fef2 <- fixef(m2)

plot(0,0,type="n",bty="l",xlim=c(-0.1,0.2),ylim=c(0,1),xlab="Centered saccade latency [in s]",ylab="Proportion correct")
for(i in 1:2){
  pred <- fef[[1]] +tc2[i]*fef[[2]] +RT*fef[[3]] +Exp[i]*fef[[4]] +tc2[i]*RT*fef[[5]] +tc2[i]*Exp[i]*fef[[6]] +Exp[i]*RT*fef[[7]] +tc2[i]*Exp[i]*RT*fef[[8]]
  lines(RT,exp(pred)/(1+exp(pred)),col=colvec[i],lty=1)
  pred2 <- fef2[[1]] +tc2[i]*fef2[[2]] +RT*fef2[[3]] +Exp[i]*fef2[[4]] +tc2[i]*RT*fef2[[5]] +tc2[i]*Exp[i]*fef2[[6]] +Exp[i]*RT*fef2[[7]] +tc2[i]*Exp[i]*RT*fef2[[8]]
  lines(RT,exp(pred2)/(1+exp(pred2)),col=colvec[i],lty=2)
}
legend(-0.1,0.5,c("congruent/ 100 ms","incongruent/ 100 ms","congruent/ 400 ms","incongruent/ 400 ms"),col=c(defblue,deforange,defblue,deforange),lty=c(1,1,2,2),bty="n")
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# close figure
dev.off()

#--------------------------------------------------------------------------
