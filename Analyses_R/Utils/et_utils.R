
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

################### old:

# # normalize it:
# mutate(fvec0_norm = Map(function(...) set_names(normalize_vec(c(...)), 
#                                                 c('x', 'y', 'z')),
#                        fvec0_x, fvec0_y, fvec0_z)) %>%
# # unpack to coulmns:
# unnest_legacy() %>%
# mutate(key = rep(c('x','y','z'), nrow(.)/3)) %>%
# pivot_wider(names_from = key,
#             values_from = fvec0_norm,
#             names_prefix = 'fvec0_norm_') %>%
# 
# # same for left eye:
# mutate(fvec1_norm = Map(function(...) set_names(normalize_vec(c(...)), 
#                                                 c('x', 'y', 'z')),
#                         fvec1_x, fvec1_y, fvec1_z)) %>%
# unnest_legacy() %>%
# mutate(key = rep(c('x','y','z'), nrow(.)/3)) %>%
# pivot_wider(names_from = key,
#             values_from = fvec1_norm,
#             names_prefix = 'fvec1_norm_') %>%


# # calculate deviation (in mm and dva):
# mutate(dev0_x = gaze_normal0_x - fvec0_norm_x,
#        dev0_y = gaze_normal0_y - fvec0_norm_y,
#        dev1_x = gaze_normal1_x - fvec1_norm_x,
#        dev1_y = gaze_normal1_y - fvec1_norm_y,
#        dev0_deg_x = asin(dev0_x) * 180/pi,
#        dev0_deg_y = asin(dev0_y) * 180/pi,
#        dev1_deg_x = asin(dev1_x) * 180/pi,
#        dev1_deg_y = asin(dev1_y) * 180/pi, 
#        ang_hor0 =  acos(fvec0_norm_x * gaze_normal0_x + fvec0_norm_z * gaze_normal0_z) * 180/pi, 
#        ang_hor1 =  acos(fvec1_norm_x * gaze_normal1_x + fvec1_norm_z * gaze_normal1_z) * 180/pi,
#        angh_0 = rad2deg(atan2(fvec0_norm_z, fvec0_norm_x) - atan2(gaze_normal0_z, gaze_normal0_x) * -1), 
#        sp_a = sqrt(fvec0_norm_x^2 + fvec0_norm_y^2 + fvec0_norm_z^2), 
#        sp_b = atan(fvec0_norm_z / fvec0_norm_x), 
#        sp_c = acos(fvec0_norm_y / sqrt(fvec0_norm_x^2 + fvec0_norm_y^2 + fvec0_norm_z^2)), 
#        sp_g_a = sqrt(gaze_normal0_x^2 + gaze_normal0_y^2 + gaze_normal0_z^2),
#        sp_g_b = atan(gaze_normal0_z / gaze_normal0_x), 
#        sp_g_z = acos(gaze_normal0_y / sqrt(gaze_normal0_x^2 + gaze_normal0_y^2 + gaze_normal0_z^2)),
#        ang_diff_h = rad2deg(sp_b - sp_g_b)
#        )
# 
# 
# sacc_tibble_names <- c('start', 'end', 'peakvel', 
#                        'sacc_vec_x', 'sacc_vec_y', 
#                        'sacc_amp_x', 'sacc_amp_y')

## Following blocks can be used to produce MSTB plots;
## Will need some tweaking!

# 
# # Plot trajectory
# par(mfrow=c(1,2))
# plot(as_vector(data_fix$dev1_deg_x),as_vector(data_fix$dev1_deg_y),type='l',asp=1,
#      xlab=expression(x[l]),ylab=expression(y[l]),
#      main="Position")
# for ( s in 1:N ) {
#   j <- bin[s,1]:bin[s,2] 
#   lines(as_vector(data_fix$dev1_deg_x[j]),as_vector(data_fix$dev1_deg_y[j]),type='l',col='red',lwd=3)
# }
# points(as_vector(data_fix[bin[,2],42]),as_vector(data_fix[bin[,2],43]),col='red')
# 
# # Plot trajectory in 2D velocity space
# vls <- vecvel(timings_eye1 %>% 
#                 select(dev1_deg_x, dev1_deg_y) %>% 
#                 as.matrix(),200)
# plot(vls[,1],vls[,2],type='l',asp=1,
#      xlab=expression(v[x]),ylab=expression(v[y]),
#      main="Velocity")
# for ( s in 1:N ) {
#   j <- bin[s,1]:bin[s,2] 
#   idxx <- which(timings_eye1$gaze_timestamp %in% data_fix$gaze_timestamp[j])
#   lines(vls[idxx,1],vls[idxx,2],type='l',col='red',lwd=3)
#   #points(vls[idxx,1],vls[idxx,2],col='red',lwd=3)
# }
# phi <- seq(from=0,to=2*pi,length.out=300)
# cx <- msl$radius[1]*cos(phi)
# cy <- msl$radius[2]*sin(phi)
# lines(cx,cy,lty=2)
# 
