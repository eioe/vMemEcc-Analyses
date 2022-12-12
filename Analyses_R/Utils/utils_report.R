

library(rjson)



# Define insert_fun - a function to insert source code in a Rmd chunk:
# https://stackoverflow.com/questions/32688936/putting-function-definition-after-call-in-r-knitr
insert_fun <- function(name) {
  read_chunk(lines = capture.output(dump(name, '')), labels = paste(name, 'source', sep = '-'))
}


print_CI_lmer <- function(m) {
  # print the CIs for the single effects of a Linear mixed model fit by REML ['lmerModLmerTest']
  
  # Paramteres:
  #
  # m:  "lmerModLmerTest" (returned by lmer())
  n_coef = length(fixef(m))
  df <- data.frame(lower = numeric(n_coef), upper = numeric(n_coef))  
  colnames(df) <- c("lower", "upper")
  rownames(df) <- names(fixef(m))
  for (i in 1:n_coef){
    L = diag(n_coef)[i, ]
    o <- contest1D(m, L, confint = T)
    ci <- c(o$lower, o$upper)
    df[i, 1:2] <- ci
  }
  return(df)
}


# Export vars to JSON file:
extract_var <- function(var, val, path_ev=path_extracted_vars, overwrite=TRUE, 
                        rm_leading_zero=FALSE, is_pval=FALSE, exp_format="%.2f") {
  exp_vars <- fromJSON(file = path_ev)
  
  if (is_pval | rm_leading_zero) {
    #TODO: handle pvals according to alpha level
    # For now only leading zero is removed.
    val_str <- str_sub(sprintf(exp_format, val), 2)
  } else {
    val_str <- sprintf(exp_format, val)
  }
  
  if (var %in% names(exp_vars)) {
    old_val <- {exp_vars[[var]]}
    if (old_val != val_str) {
      if (overwrite) {
        txt_warn <- str_glue('Overwriting old value of {var} ({old_val}) with: {val_str}')
        warning(txt_warn)
      } else {
        txt_warn <- str_glue('There is already a value for {var}: {old_val}.\n
                              Allow overwriting to extract new value: {val_str}\n
                              Skipping export -- keeping old value.')
        warning(txt_warn)
        return()
      }
    }
  }
  
  exp_vars[[var]] <- val_str
  
  # Save:
  jsonData <- toJSON(exp_vars, indent=4)
  write(jsonData, path_ev)
}