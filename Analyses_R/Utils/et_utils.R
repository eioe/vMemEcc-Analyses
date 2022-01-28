
#--------------------------------------------------------------------------
# ET Utils
#
# Helper functions for the eye tracking analyses
#--------------------------------------------------------------------------

require(tidyverse)
require(usethis)
require(pracma)
require(zoo)

get_blink_frames <- function(df_blink) {
  dd <- df_blink %>% 
    group_by(n = row_number()) %>% 
    do(data.frame(blink_frames = seq(from = .$start_frame_index, to = .$end_frame_index))) %>% 
    ungroup() %>% 
    dplyr::select(blink_frames) %>% 
    as_vector()
  return(dd)
}


normalize_vec <- function(x) {x / sqrt(sum(x^2))}



cart2sph_custom <- function (xyz) {
  # ' Convert cartesian coordinates (xyz) to spherical coordinates (theta-phi-r) with the zenith being at the positve end of z 
  # ' (in a left handed coordinate system with z going into depth!).
  # '
  # ' @param xyz Vector with x-y-z coordinates. Or matrix with shape nx3 cols: x-y-z)
  # ' @return A vector/matrix with the elements/columns: theta (azimuth), phi (inclination), and r (dist to origin).  
  
  stopifnot(is.numeric(xyz))
  if (is.vector(xyz) && length(xyz) == 3) {
    x <- xyz[1]
    y <- xyz[2]
    z <- xyz[3]
    m <- 1
  }
  else if (is.matrix(xyz) && ncol(xyz) == 3) {
    x <- xyz[, 1]
    y <- xyz[, 2]
    z <- xyz[, 3]
    m <- nrow(xyz)
  }
  else stop("Input must be a vector of length 3 or a matrix with 3 columns.")
  hypotxz <- hypot(x, z)
  r <- hypot(y, hypotxz)
  phi <- atan2(y, hypotxz) * -1    #multiply with -1 to correct the sign to fit the left handed coordinate system used by the ET/Unity
  theta <- atan2(x, z) 
  if (m == 1) 
    tpr <- c(theta, phi, r)
  else tpr <- cbind(theta, phi, r)
  return(tpr)
}



translate_xyz2spherical <- function(df, x_col, y_col, z_col, prefix_output) {
  x_col <- enquo(x_col)
  y_col <- enquo(y_col)
  z_col <- enquo(z_col)
  
  res <- df %>% 
    add_column(!!str_c(prefix_output, '_theta') := rad2deg(cart2sph_custom(as.matrix(select(., !! x_col, !! y_col, !! z_col)))[, 1])) %>% 
    add_column(!!str_c(prefix_output, '_phi')   := rad2deg(cart2sph_custom(as.matrix(select(., !! x_col, !! y_col, !! z_col)))[, 2])) %>% 
    add_column(!!str_c(prefix_output, '_r')     :=         cart2sph_custom(as.matrix(select(., !! x_col, !! y_col, !! z_col)))[, 3])
  return(res)
}


spline_interpolate_low_conf_samples <- function(vec, 
                                                conf_vec, 
                                                conf_threshold, 
                                                margin = 20,
                                                maxgap = 100) {
  if (!typeof(vec) == 'double') {
    warning(paste('Expected type "double" but got type: ', 
                  typeof(vec)))
  }
  margin_start <- vec[1:margin]
  margin_end <- vec[(length(vec)-margin+1):length(vec)]
  vec_trimmed <- vec[(margin+1):(length(vec)-margin)]
  conf_vec_trimmed <- conf_vec[(margin+1):(length(vec)-margin)]
  vec_trimmed[conf_vec_trimmed < conf_threshold] <- NA_real_
  vec_concat <- c(margin_start, vec_trimmed, margin_end)
  vec_spline <- na.approx(vec_concat, 
                          maxgap = maxgap, 
                          na.rm = FALSE)
  if (sum(is.na(vec_spline)) > 2*margin) {
    na_idx <- which(is.na(vec_spline))
    vec_spline[na_idx] <- vec[na_idx]
    warning('There were stretches of low confidence longer 
             than the max gap allowed. Leaving original values 
            for these.')
  }
    
  #vec_out <- c(margin_start, vec_spline, margin_end)
  return(vec_spline)                
}




et_resample <- function(df, var_time, srate, tmax = NULL, tmin = NULL) {
  # ' Resample to fixed sampling rate, keeping only sample with highest confidence
  # '
  # ' @param df Dataframe or tibble with eye tracking data as output by `\Analyses_R\EyeTracking\extract_data_to_rds.R`. 
  #             Normally you'll want to hand over a grouped df (by ID, trial, ...). 
  # ' @param var_time (character) Name of the column in the df that contains the original time stamps of the samples.
  # ' @param srate (int) Targeted sampling rate. 
  # ' @param tmin/tmax (double) Start/End time of the interval which shall be extracted and resampled.
  # ' @return A vector/matrix with the elements/columns: theta (azimuth), phi (inclination), and r (dist to origin).  
  
  if (!is_null(tmax)) {
    df <- df %>% 
      filter(get(var_time) <= tmax + (1/srate)/2) 
  } else 
  {
    tmax <- df$'var_time'[length(df$'var_time')]
  }
  
  if (!is_null(tmin)) {
    df <- df %>%
      filter(get(var_time) >= tmin - (1/srate)/2)
  } else 
  {
    tmin <- df$'var_time'[1]
  }
  
  times <- seq(tmin, tmax, 1/srate)
  
  df_resampled <- df %>% 
    mutate(time_resampled = sapply(gaze_timestamp, findindx, times)) %>% 
    group_by(time_resampled, .add = T) %>% 
    mutate(max_conf_ = max(confidence)) %>% 
    filter(confidence == max_conf_) %>% 
    select(!max_conf_) %>% 
    ungroup(time_resampled) %>% 
    distinct(time_resampled, .keep_all = TRUE) 
  
  return(df_resampled)
}


findindx <- function(vec, ttimes) {
  out <- ttimes[which.min(abs(vec - ttimes))]
  return(out)
}




# dir helper:
checkmake_dirs <- function(paths) {
  for (path in paths) {
    if (!dir.exists(path)) {
      dir.create(path)
      ui_info(str_c("Created dir: {path}"))
    }
  }
}

