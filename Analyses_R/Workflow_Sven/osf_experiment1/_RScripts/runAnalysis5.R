#--------------------------------------------------------------------------
# The influence of saccade latency on memory performance
#
# use (see genVariables.R):
# testcond2 <- set contrast: baseline is incongruent
# sacRT2    <- z-transformed sacRT for each participant
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# set vp to factor
data$vp <- as.factor(data$vp)

# Define contrast - baseline is cue delay = 3200
data$timecondfac <- as.factor(data$cueTim)
contrasts(data$timecondfac) <- contr.treatment(5,base=1)
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# run glmer
m1 <- glmer(answer ~ testcond2*timecondfac*sacRT2 + (1|vp), data=data, family="binomial")
summary(m1)
#--------------------------------------------------------------------------
