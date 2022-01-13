#--------------------------------------------------------------------------
# Calculate Cosineau-Morey confidence intervals.
# 
# Baguley, T. (2011). Calculating and graphing within- subject confidence intervals for ANOVA. 
#                     Behavior Research Methods, 44(1), 158–175.
# Morey, R. D. (2008). Confidence intervals from normalized data: A correction to Cousineau (2005). 
#                      Tutorial in Quantitative Methods for Psychology, 42(2), 61–64.


# load helper functions
se_up   <-  function(x){mean(x) + 1.96*sd(x)/sqrt(length(x))}
se_down <-  function(x){mean(x) - 1.96*sd(x)/sqrt(length(x))}

# load helper function for cousineau morey
cm.ci <- function(data.frame, conf.level = 0.95, difference = TRUE) {
  #cousineau-morey within-subject CIs
  k = ncol(data.frame)
  if (difference == TRUE) 
    diff.factor = 2^0.5/2
  else diff.factor = 1
  n <- nrow(data.frame)
  df.stack <- stack(data.frame)
  index <- rep(1:n, k)
  p.means <- tapply(df.stack$values, index, mean)
  norm.df <- data.frame - p.means + (sum(data.frame)/(n * k))
  t.mat <- matrix(, k, 1)
  mean.mat <- matrix(, k, 1)
  for (i in 1:k) t.mat[i, ] <- t.test(norm.df[i])$statistic[1]
  for (i in 1:k) mean.mat[i, ] <- mean(as.matrix(norm.df[i]))
  c.factor <- (k/(k - 1))^0.5
  moe.mat <- mean.mat/t.mat * qt(1 - (1 - conf.level)/2, n - 1) * c.factor * 
    diff.factor
  ci.mat <- matrix(, k, 2)
  dimnames(ci.mat) <- list(names(data.frame), c("lower", "upper"))
  for (i in 1:k) {
    ci.mat[i, 1] <- mean.mat[i] - moe.mat[i]
    ci.mat[i, 2] <- mean.mat[i] + moe.mat[i]
  }
  ci.mat
}
#--------------------------------------------------------------------------