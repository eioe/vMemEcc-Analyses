
#--------------------------------------------------------------------------
# ET Utils
#
# Helper functions for the eye tracking analyses
#--------------------------------------------------------------------------

require(pracma)

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


