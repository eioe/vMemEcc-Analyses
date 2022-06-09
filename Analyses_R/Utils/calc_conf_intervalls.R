

ci95lower <- function(x, distribution = "t") {
  return( mean(x) - qt(0.975,df=length(x)-1)*sd(x)/sqrt(length(x)))
}

ci95upper <- function(x) {
  return( mean(x) + qt(0.975,df=length(x)-1)*sd(x)/sqrt(length(x)))
}


ci95_meandiff <- function(x, y, distribution = "t") {
  mean_x = mean(x)
  mean_y = mean(y)
  mean_diff = mean_x - mean_y
  n_x <- length(x)
  n_y <- length(y)
  
  if (distribution == "t") {
  
    var_pooled <- ((n_x-1) * sd(x)^2 + (n_y-1) *sd(y)^2) / (n_x+n_y-2)  # actually simplifies to (sd(x)^2 + sd(y)^2) / 2 if n_x == n_y  
    ci_margin <- qt(0.975,df=n_x+n_y-1)*sqrt(var_pooled/n_x + var_pooled/n_y)
  }
  
  else if (distribution == "z") {
    var_diff <- sd(x)^2/n_x + sd(y)^2/n_y
    sd_diff <- sqrt(var_diff)
    ci_margin <- qnorm(0.975) * sd_diff
  }
  
  else {
    stop(paste("Invalid distribution argument:", distribution))
  }
  
  ci = list()
  ci$lower = mean_diff - ci_margin
  ci$upper = mean_diff + ci_margin
  return (ci)
}