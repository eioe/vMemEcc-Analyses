

ci95lower <- function(x, distribution = "t") {
  if (distribution == "t") {
    ci_margin <- qt(0.975,df=length(x)-1)*sd(x)/sqrt(length(x))
  } 
  else if (distribution == "z") {
    ci_margin <- qnorm(0.975) * sd(x)/sqrt(length(x))
  }
  return( mean(x) - ci_margin)
}

ci95upper <- function(x, distribution = "t") {
  if (distribution == "t") {
    ci_margin <- qt(0.975,df=length(x)-1)*sd(x)/sqrt(length(x))
  } 
  else if (distribution == "z") {
    ci_margin <- qnorm(0.975) * sd(x)/sqrt(length(x))
  }
  return( mean(x) + ci_margin)
}


ci95_meandiff <- function(x, y, distribution = "t", dependent = TRUE) {
  
  mean_x = mean(x)
  mean_y = mean(y)
  mean_diff = mean_x - mean_y
  n_x <- length(x)
  n_y <- length(y)
  
  if (dependent) {
    
    ## Dep. samples:
  
    # Formula from here: 
    # https://www.khanacademy.org/math/statistics-probability/significance-tests-confidence-intervals-two-samples/comparing-two-means/v/confidence-interval-of-difference-of-means
    #
    
    if (! (n_x == n_y)) {
      warning("Groups are of different size. You sure that you want to run a dependent test? Using the larger group size for now.")
    }
    
    var_diff <- sd(x)^2/n_x + sd(y)^2/n_y
    sd_diff <- sqrt(var_diff)
    if (distribution == "t") {
      ci_margin <- qt(0.975, df=max(n_x, n_y)) * sd_diff
    }
    else if (distribution == "z") {
      ci_margin <- qnorm(0.975) * sd_diff
    }
  }
    
  
  else {
    
    ## Indep. samples:
  
    # Formula from here: 
    # https://www.r-bloggers.com/2021/11/calculate-confidence-intervals-in-r/
    
    var_pooled <- ((n_x-1) * sd(x)^2 + (n_y-1) *sd(y)^2) / (n_x+n_y-2)  # actually simplifies to (sd(x)^2 + sd(y)^2) / 2 if n_x == n_y  
    
    if (distribution == "t") {
      ci_margin <- qt(0.975,df=n_x+n_y-1)*sqrt(var_pooled/n_x + var_pooled/n_y)
    } 
    else if (distribution == "z") {
      ci_margin <- qnorm(0.975)*sqrt(var_pooled/n_x + var_pooled/n_y)
    }
  }
  
  ci = list()
  ci$lower = mean_diff - ci_margin
  ci$upper = mean_diff + ci_margin
  return (ci)
}


summarize_ttest <- function(x, returnval="p") {
  
  res <- t.test(x)
  if (returnval == "t") {
    returnThis = res$statistic
  } 
  else if (returnval == "p") {
    returnThis = res$p.value
  }
  
  return (returnThis)
}
