
library(here)
library(tidyverse)
library(magrittr)
library(pracma)
library(usethis)
library(ggpubr)
library(zoo)

source("EyeTracking/MS_Toolbox_R/vecvel.R")
source("EyeTracking/MS_Toolbox_R/microsacc.R")
source("EyeTracking/MS_Toolbox_R/binsacc.R")
source("Utils/et_utils.R")

path_data <- here('../../Data/SubjectData/VME_S23/EyeTracking/Block7/000/exports/000')


data_annot  <- read_csv(file.path(path_data, 'annotations.csv'))
data_gaze   <- read_csv(file.path(path_data, 'gaze_positions.csv '))
data_blinks <- read_csv(file.path(path_data, 'blinks.csv'))
data_pupils <- read_csv(file.path(path_data, 'pupil_positions.csv')) 

timings <- list(
  'fix' = 0.8,
  'cue' = 0.8, 
  'stim' = 0.2, 
  'retention' = 2, 
  'buffer_prefix' = 0.4, 
  'buffer_postretention' = 0.4) 

sacc_params <- list(
  #  VFAC			  relative velocity threshold
  vfac = 6, # suggested by Sven [April 28]: 5
  #  MINDUR		  minimal saccade duration, 
  mindur = 4, # suggested by Sven [April 28]: 3
  # Sampling rate: 
  srate = 200, 
  # Use 1 eye for sacc detection (using the one with higher confidence values):
  monoc = TRUE, 
  # Use both eyes:
  binoc = FALSE)



# filter out annotation info re. the trial type:
data_trialtype <- data_annot %>% 
  mutate(ttype = parse_number(label) - 150) %>% 
  filter(ttype < 24 & ttype >= 0) #%>% 

# filter out annotations for stim onsets:
data_stimon <- data_annot %>%  
  mutate(ttype = parse_number(label)) %>% 
  filter(ttype == 2) 

# in case of aborted trials these DFs might have different lengths and need repair:
if (nrow(data_trialtype) > nrow(data_stimon)) {
  # check for subsequent rows in data_trialtype with same ttype:
  n_rows <- nrow(data_trialtype)
  dbl_idx <- which(data_trialtype$ttype[1 : n_rows-1] == data_trialtype$ttype[2 : n_rows])
  
  # now check for all candidates whether there was a stimulus onset between these two trial starts
  # (if there was not, then this was an aborted and repeated trial)
  idx_kickme <- vector(length = 0)
  for (ii in dbl_idx) {
    val <- (any(data_stimon$timestamp > data_trialtype$timestamp[ii] & data_stimon$timestamp < data_trialtype$timestamp[ii+1])) 
    if (!val) {
      idx_kickme <- append(idx_kickme, ii)
    }
  }
  
  # clean data_trialtype:
  data_trialtype <- data_trialtype[-idx_kickme, ] 
}

# Add the ttype info to the stimonset df:
data_stimon <- data_stimon %>% 
  mutate(ttype       = data_trialtype$ttype, 
         CueDir      = ifelse(mod(ttype, 2) < 1, 'Left', 'Right'), 
         ChangeTrial = ifelse(mod(ttype, 4) < 2, TRUE, FALSE), 
         c_StimN     = ifelse(mod(ttype, 8) < 4, 2, 4), 
         c_Ecc       = ifelse(mod(ttype, 24) < 8, 
                              4, 
                              ifelse(mod(ttype, 24) > 15, 
                                     14, 
                                     9))) 

# read blink frames from pupil player export:
blink_frames <- get_blink_frames(data_blinks)


# translate gaze info into dva:
data_gaze <- data_gaze %>% 
  # first calculate a vector per eye from its center to the pos of the 
  # fixation cross (in camera space): (0,0,1000)[mm]
  mutate(fvec0_x = 0 - eye_center0_3d_x, 
         fvec0_y = 0 - eye_center0_3d_y, 
         fvec0_z = 1000 - eye_center0_3d_z, 
         fvec1_x = 0 - eye_center1_3d_x, 
         fvec1_y = 0 - eye_center1_3d_y, 
         fvec1_z = 1000 - eye_center1_3d_z) %>% 
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
  translate_xyz2spherical(fvec0_x, fvec0_y, fvec0_z, 'fvec0') %>% 
  translate_xyz2spherical(fvec1_x, fvec1_y, fvec1_z, 'fvec1') %>% 
  translate_xyz2spherical(gaze_normal0_x, gaze_normal0_y, gaze_normal0_z, 'gaze_normal0') %>% 
  translate_xyz2spherical(gaze_normal1_x, gaze_normal1_y, gaze_normal1_z, 'gaze_normal1') %>% 
  mutate(gaze_dev0_hor  = gaze_normal0_theta - fvec0_theta,
         gaze_dev0_vert = gaze_normal0_phi   - fvec0_phi,
         gaze_dev1_hor  = gaze_normal1_theta - fvec1_theta,  
         gaze_dev1_vert = gaze_normal1_phi   - fvec1_phi,
         dev0_deg_x = gaze_dev0_hor, 
         dev0_deg_y = gaze_dev0_vert, 
         dev1_deg_x = gaze_dev1_hor,
         dev1_deg_y = gaze_dev1_vert) 
  
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

## Loop over trials:
# - epoch data 
# - calc times relative to stim onset
# - extract data per eye
# - calc saccades per eye

plt_list <- NULL

for (trial in 1:nrow(data_stimon)) {

  # epoch data:
  t_stimonset <- data_stimon$timestamp[trial] 
  # get indices for all rows belonging to this trial:
  idx <- data_gaze %>%  
    select(gaze_timestamp) %>% 
    mutate(n = row_number()) %>% 
    filter(
      gaze_timestamp > (t_stimonset - (timings$fix + timings$cue + timings$buffer_prefix)), 
      gaze_timestamp < (t_stimonset + (timings$stim + timings$retention) + timings$buffer_postretention)) %>% 
    pull(n)

  # extract according rows:
  data_fix <-  data_gaze %>% 
    slice(idx) %>% 
    mutate(blink = world_index %in% blink_frames, 
           CueDir = data_stimon$CueDir[trial]) 
  
  # remove samples w/ duplicated timestamps
  n_duplic_timestamps <- sum(duplicated(data_fix$gaze_timestamp))
  if (n_duplic_timestamps > 0) {
    ui_warn(sprintf("Removing %i samples with identical timestamps. Keeping first instance of each.", n_duplic_timestamps))  
  }
  data_fix <- data_fix %>% 
    distinct_at(vars(gaze_timestamp), .keep_all = TRUE)
  
  
  # add saccade info:
  # get separate cols for eye samples: 
  data_fix <- data_fix %>% 
    separate(base_data, c('timestamp_eye_0','eye0' ,'timestamp_eye_1', 'eye1'), '[- ]', convert = TRUE, remove = FALSE) %>% 
    mutate(timestamp_eye_1 = if_else(eye0 == 1, timestamp_eye_0, timestamp_eye_1), 
           timestamp_eye_0 = if_else(eye0 == 1, NA_real_ , timestamp_eye_0)) %>% 
    select(-c(eye0, eye1)) %>% 
    # calc time relative to stimulus onset:
    mutate(gaze_timestamp = gaze_timestamp - t_stimonset, 
           timestamp_eye_0 = timestamp_eye_0 - t_stimonset, 
           timestamp_eye_1 = timestamp_eye_1 - t_stimonset, 
           trial = trial)
  
  # extract one DF per eye:
  timings_eye0 <- data_fix %>% 
    select(timestamp_eye_0, 
           trial, 
           gaze_timestamp, 
           gaze_normal0_x, 
           gaze_normal0_y, 
           dev0_deg_x, 
           dev0_deg_y) %>% 
    distinct_at(vars(timestamp_eye_0, gaze_normal0_x, gaze_normal0_y), 
                .keep_all = TRUE) %>% 
    drop_na(timestamp_eye_0, gaze_normal0_x, gaze_normal0_y) 
  
  timings_eye1 <- data_fix %>% 
    select(timestamp_eye_1, 
           trial, 
           gaze_timestamp, 
           gaze_normal1_x, 
           gaze_normal1_y, 
           dev1_deg_x, 
           dev1_deg_y) %>% 
    distinct_at(vars(timestamp_eye_1, gaze_normal1_x, gaze_normal1_y), 
                .keep_all = TRUE) %>% 
    drop_na(timestamp_eye_1, gaze_normal1_x, gaze_normal1_y)  
    
  
  # Calculate (micro)saccaes per eye:
  # msr: micro-saccades right eye
  # msl: -------------- left ----
  
  msr <- timings_eye0 %>% 
    select(dev0_deg_x, dev0_deg_y) %>% 
    as.matrix() %>% 
    microsacc(sacc_params$vfac,
              sacc_params$mindur,
              sacc_params$srate)
  
  msl <- timings_eye1 %>% 
    select(dev1_deg_x, dev1_deg_y) %>% 
    as.matrix() %>%  
    microsacc(sacc_params$vfac,
              sacc_params$mindur,
              sacc_params$srate)
  
  # translate to global indices used in gaze_data:
  if (!is.null(msl)) {
    for (i in 1:nrow(msl$table)) {
      msl$table[i, 1] <- min(which(timings_eye1$gaze_timestamp[msl$table[i,1]] == data_fix$gaze_timestamp))
      msl$table[i, 2] <- max(which(timings_eye1$gaze_timestamp[msl$table[i,2]] == data_fix$gaze_timestamp))
    }
  } else {
    msl$table <- NULL
  }
  
  if (!is.null(msr)) {
    for (i in 1:nrow(msr$table)) {
      msr$table[i, 1] <- min(which(timings_eye0$gaze_timestamp[msr$table[i,1]] == data_fix$gaze_timestamp))
      msr$table[i, 2] <- max(which(timings_eye0$gaze_timestamp[msr$table[i,2]] == data_fix$gaze_timestamp))
    }
  } else {
    msr$table <- NULL
  }
  
  
  # check which eye has higher average confidence for this trial:
  eye_confidence <- data_pupils %>%  
    group_by(eye_id) %>% 
    slice(which(world_index %in% data_fix$world_index)) %>% 
    summarise(mean_conf = mean(confidence, na.rm = TRUE)) %>% 
    pivot_wider(names_from = eye_id, 
                names_prefix = 'eye_',
                values_from = mean_conf)
  
  chosen_eye <- ifelse(eye_confidence$eye_0 > eye_confidence$eye_1, 
                       0, 
                       1) 
    
  
  if (sacc_params$monoc) {
    
    if (chosen_eye == 0) {
      sacc_table <- msr$table
      } else {
        sacc_table <- msl$table
      }
  
    # initialize columns:
    data_fix <- data_fix %>% 
      mutate(
        sacc               = FALSE, 
        sacc_amp_x    = NA_real_, 
        sacc_amp_y    = NA_real_, 
        sacc_amp      = NA_real_, 
        sacc_peakvel  = NA_real_
      )
    
    if (!is.null(sacc_table)) {
      for (sacc_ in 1:nrow(sacc_table)) {
        idx_sacc <- sacc_table[sacc_,1]:sacc_table[sacc_,2]
        
        data_fix <- data_fix %>% 
          mutate(row_n = row_number(),
                 sacc = if_else((row_n %in% idx_sacc),
                                    TRUE, 
                                    sacc), 
                 sacc_amp_x = if_else((row_n %in% idx_sacc), 
                                          sacc_table[sacc_, 6],
                                          sacc_amp_x),
                 sacc_amp_y = if_else((row_n %in% idx_sacc),
                                          sacc_table[sacc_, 7], 
                                          sacc_amp_y),
                 sacc_amp = if_else((row_n %in% idx_sacc),
                                        sqrt(sacc_amp_x^2 + sacc_amp_y^2),
                                        sacc_amp),
                 sacc_peakvel = if_else((row_n %in% idx_sacc),
                                            sacc_table[sacc_, 3], 
                                            sacc_peakvel))
      }
    }
  } 
  
  
  if(sacc_params$binoc) {

  
    # Calculate binocular saccades:
    sac <- binsacc(msl$table,msr$table)
    bin <- sac$bin
  
    # add info to df:
    data_fix <- data_fix %>% 
      mutate(
        sacc              = FALSE, 
        sacc_amp_x    = NA_real_, 
        sacc_amp_y    = NA_real_, 
        sacc_amp      = NA_real_, 
        sacc_peakvel  = NA_real_
      )
    
  # Loop over bin-saccs:
  if (!is.null(sac$bin)) {
    for (binsacc_ in 1:nrow(sac$bin)) {
      idx_binsacc <- bin[binsacc_,1]:bin[binsacc_,2]
      
      data_fix <- data_fix %>% 
        mutate(row_n = row_number(),
                 sacc = if_else((row_n %in% idx_binsacc),
                                    TRUE, 
                                    sacc), 
                 sacc_amp_x = if_else((row_n %in% idx_binsacc), 
                                         mean(bin[binsacc_, 6], bin[binsacc_, 13]),
                                         sacc_amp_x),
                 sacc_amp_y = if_else((row_n %in% idx_binsacc),
                                         mean(bin[binsacc_, 7], bin[binsacc_, 14]), 
                                         sacc_amp_y),
                 sacc_amp = if_else((row_n %in% idx_binsacc),
                                       sqrt(sacc_amp_x^2 + sacc_amp_y^2),
                                       sacc_amp),
                 sacc_bin_peakvel = if_else((row_n %in% idx_binsacc),
                                           mean(bin[binsacc_, 3], bin[binsacc_, 10]), 
                                           sacc_peakvel))
      }
    }
  }
      
               
               
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


  colors <- c('normal'          = 'black', 
              'saccade'         = 'red', 
              'blink'           = 'blue',
              'blink & saccade' = 'purple')
  
  plt <- data_fix %>% 
    mutate(hor_dev = ifelse(rep(chosen_eye, nrow(data_fix)) == 0, 
                            rollmean(na.spline(dev0_deg_x), 5, na.pad = TRUE), 
                            rollmean(na.spline(dev1_deg_x), 5, na.pad = TRUE)),  
           
           plt_type = case_when(
             (blink & sacc)    ~ 'blink & saccade',
             blink             ~ 'blink',
             sacc              ~ 'saccade',
             TRUE              ~ 'normal'),  
           plt_type = factor(plt_type, levels = names(colors)),
           plt_alpha = if_else(plt_type == '0', 
                               0.3, 
                               0.7)) %>% 
    ggplot(aes(x = gaze_timestamp, #- (timings$fix + timings$cue), 
               y = hor_dev)) + 
    geom_vline(xintercept = c(-timings$cue, 
                              0, 
                              timings$stim, 
                              timings$stim + timings$retention), 
               alpha = 0.3, 
               color = 'brown', 
               linetype = 2) +
    
    #geom_line(aes(y = 3*rollmean(confidence, 5, na.pad = T) +3.5)) +
    geom_line(data = data_pupils %>%  
                filter(eye_id == chosen_eye) %>% 
                slice(which(world_index %in% data_fix$world_index)) %>% 
                mutate(pupil_timestamp = pupil_timestamp - t_stimonset), 
              aes(x = pupil_timestamp, y = rollmean(confidence, 10, na.pad = T) + 3)) +
    
    geom_hline(yintercept = c(3, 3.5,4)) +
    
    annotate(geom = 'text', 
             x = c(-0.8, 0, 0.2, 2.2), 
             y = -Inf-1,
             hjust = 0,
             size = 3,
             label = c('onset cue', 'onset stimulus', 'start retention', 'end retention'), 
             angle = 90) + 
    annotate(geom = 'label',
             x = as.numeric(ifelse(any(data_fix$sacc_amp > 2 & 
                                         (!data_fix$blink) & 
                                     data_fix$gaze_timestamp > (-timings$fix) & 
                                     data_fix$gaze_timestamp < (timings$stim + timings$retention)),
                                   0, NA)),
             y = Inf,
             vjust = 1,
             color = 'red',
             label = "reject") +
    geom_hline(yintercept = 0, 
               alpha = 0.2) +
    geom_line(aes(color = plt_type, 
                  group = trial, 
                  alpha = plt_alpha),
               size = 0.1) +
    geom_segment(aes(x = -0.8, y = 1 * if_else(CueDir == 'Left', -1, 1), 
                     xend = -0.8, yend = 1.5 * if_else(CueDir == 'Left', -1, 1)), 
                 arrow = arrow(length = unit(0.05, 'npc'), 
                               type = 'closed')) +
    ylim(-7,7) + 
    theme_classic() + 
    theme(#legend.position = c(0.7, 0.2), 
          legend.direction = 'vertical',
          legend.position = 'none',
          legend.title = element_blank(),
          #axis.text.x=element_blank(), 
          axis.title = element_blank()) +
    guides(alpha = FALSE) +
    scale_color_manual(values = colors, 
                       breaks = c('blink', 
                                  'saccade', 
                                  'blink & saccade'), 
                       drop = FALSE) 

  plt_list[[trial]] <- plt #append(plt_list, plt)
             
}

li <- plt_list[1:72]
fig <- ggarrange(plotlist = li, ncol = 6, nrow = 12, common.legend = TRUE)
ggsave(file="C:/Users/Felix/Downloads/trials_S23B7_monoo_conf.pdf", 
       plot=fig, 
       device = 'pdf', 
       width = unit(20, 'cm'), 
       height = unit(30, 'cm'))

