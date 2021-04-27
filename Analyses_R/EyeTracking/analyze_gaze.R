
library(here)
library(tidyverse)
library(glue)
library(magrittr)
library(pracma)
library(usethis)
library(ggpubr)
library(zoo)

source("EyeTracking/MS_Toolbox_R/vecvel.R")
source("EyeTracking/MS_Toolbox_R/microsacc.R")
source("EyeTracking/MS_Toolbox_R/binsacc.R")
source("Utils/et_utils.R")

interpolate_low_conf_samples <- TRUE

data_from_hdd <- TRUE
overwrite_existing_dataoutput <- TRUE
suppress_plotting <- FALSE

timings <- list(
  'fix' = 0.8,
  'cue' = 0.8, 
  'stim' = 0.2, 
  'retention' = 2, 
  'buffer_prefix' = 0.4, 
  'buffer_postretention' = 0.4, 
  'buffer_blink' = 0.1, 
  'eeg_bl' = 0.2) 

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
  binoc = FALSE, 
  # Amplitude threshold (dva) to decide which trials to reject:
  rej_threshold = 2, 
  # Confidence threshold: 
  conf_threshold = 0.6)


# Set paths: 
if (data_from_hdd) {
  path_data <- file.path('E:', 'vMemEcc', 'SubjectData_extern', 'SubjectData')
} else {
  path_data <- here('..', '..', 'Data', 'SubjectData')
}


sub_ids <- str_c('VME_S', str_pad(setdiff(c(20), c(11,14,19)), 
                                  2, 'left', 0))

for (sub_id in sub_ids) {
  
  # Initialize data collectors:
  sacc_list <- NULL 
  sacc_list_idx <- 0
  
  # Input dir:
  path_data_sub     <- file.path(path_data, sub_id, 'EyeTracking')
  # Output dir: 
  path_plots        <- file.path(path_data_sub, 'R_plots')
  path_data_sub_out <- file.path(path_data_sub, 'R_data')
  path_data_rejepo_out <- file.path(path_data, 'CSV_rejEpos_ET')
  checkmake_dirs(c(path_plots, path_data_sub_out, path_data_rejepo_out))
  
  blocks <- list.files(path_data_sub)
  # skip trainings blocks & output dirs:
  blocks <- blocks[!blocks %in% c('Block1', 'Block3', 'R_plots', 'R_data')]
  # sort ascending:
  blocks <- blocks[order(parse_number(blocks))]
  
  # start timer:
  tic()
  
  # loop over blocks for this sub:
  for (block_nr in blocks) {
    
    # TODO: The following prevents plotting-only runs - you might want to fix this.
    # Check if output already exists and skip accordingly:
    if ((length(list.files(path_data_sub_out)) > 0) && !overwrite_existing_dataoutput) {
        ui_info(glue("Skipping {block_nr} for {sub_id}. Output exists already \\
                     and overwriting is off."))
        next
    }
  
    # Get dir with data for this block:
    path_data_block <- file.path(path_data_sub, block_nr, '000', 'exports', '000')
    
    # Read in data:
    data_annot  <- read_csv(file.path(path_data_block, 'annotations.csv'))
    data_gaze   <- read_csv(file.path(path_data_block, 'gaze_positions.csv '))
    data_blinks <- read_csv(file.path(path_data_block, 'blinks.csv'))
    data_pupils <- read_csv(file.path(path_data_block, 'pupil_positions.csv')) 
    
    # Set retention interval to 0 for perceptual block:
    if (block_nr == 'Block2') {
      timings$retention_backup <- ifelse(timings$retention > 0, 
                                         timings$retention, 
                                         0)
      timings$retention <- 0
    } else {
      if (timings$retention == 0) {
        timings$retention <- timings$retention_backup
      }
    }
                 
    # filter out annotation info re. the trial type:
    data_trialtype <- data_annot %>% 
      mutate(ttype = parse_number(label) - 150) %>% 
      filter(ttype < 24 & ttype >= 0) #%>% 
    
    # filter out annotations for stim onsets:
    data_stimon <- data_annot %>%  
      mutate(ttype = parse_number(label)) %>% 
      filter(ttype == 2) 
    
    # in case of aborted trials these DFs might have different lengths and need
    # repair:
    if (nrow(data_trialtype) > nrow(data_stimon)) {
      # check for subsequent rows in data_trialtype with same ttype:
      n_rows <- nrow(data_trialtype)
      dbl_idx <- which(data_trialtype$ttype[1 : n_rows-1] == 
                         data_trialtype$ttype[2 : n_rows])
      
      # now check for all candidates whether there was a stimulus onset between 
      # these two trial starts
      # (if there was not, then this was an aborted and repeated trial)
      idx_kickme <- vector(length = 0)
      for (ii in dbl_idx) {
        val <- (any((data_stimon$timestamp > data_trialtype$timestamp[ii]) & 
                      (data_stimon$timestamp < data_trialtype$timestamp[ii+1]))) 
        if (!val) {
          idx_kickme <- append(idx_kickme, ii)
        }
      }
      
      # clean data_trialtype:
      data_trialtype <- data_trialtype[-idx_kickme, ] 
    }
    
    # Add the trial-type info to the stimonset df:
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
      # translate this vec and the gaze normal to spherical coords:
      translate_xyz2spherical(fvec0_x, fvec0_y, fvec0_z, 'fvec0') %>% 
      translate_xyz2spherical(fvec1_x, fvec1_y, fvec1_z, 'fvec1') %>% 
      translate_xyz2spherical(gaze_normal0_x, gaze_normal0_y, gaze_normal0_z, 
                              'gaze_normal0') %>% 
      translate_xyz2spherical(gaze_normal1_x, gaze_normal1_y, gaze_normal1_z, 
                              'gaze_normal1') %>% 
      # Calculate vert. and hor. difference between gaze vector and vector from
      # eye to fix cross (via subtraction of spherical coords theta and phi):
      mutate(gaze_dev0_hor  = gaze_normal0_theta - fvec0_theta,
             gaze_dev0_vert = gaze_normal0_phi   - fvec0_phi,
             gaze_dev1_hor  = gaze_normal1_theta - fvec1_theta,  
             gaze_dev1_vert = gaze_normal1_phi   - fvec1_phi) 
      
      ## Loop over trials:
      # - epoch data 
      # - remove samples with identical timestamps
      # - calc times relative to stim onset
      # - extract data per eye
      # - calc saccades per eye
      # - merge back to common gaze DF
      # - write out saccade info and store in list
      # - create plot for this trial and store in list
    
      # Initialize data holder for plot objects: 
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
      
        # Get the timings of the samples which are already 
        # classified as blinks:
        blink_times <- data_gaze %>% 
          slice(idx) %>% 
          # mutate(row_n = row_number()) %>% 
          slice(which(world_index %in% blink_frames)) %>% 
          # mutate(offset = c(if_else(diff(row_n) > 1, TRUE, FALSE), TRUE), 
          #        onset = c(if_else(diff(c(-Inf, row_n)) > 1, TRUE, FALSE))) %>% 
          # select(gaze_timestamp, onset, offset, row_n) %>% 
          pull(gaze_timestamp)
      
        # extract according rows:
        data_fix <-  data_gaze %>% 
          slice(idx) %>% 
          # add columns with info re. blinks and manipulations: 
          mutate(blink = world_index %in% blink_frames, 
                 CueDir = data_stimon$CueDir[trial], 
                 c_Ecc = data_stimon$c_Ecc[trial], 
                 c_StimN = data_stimon$c_StimN[trial]) %>% 
          # add 100ms buffer before and after blink samples marked by pupil 
          # labs algorithm:
          mutate(blink = sapply(gaze_timestamp,
                                function(x) ifelse(any(abs(x - blink_times) <
                                                         timings$buffer_blink),
                                                   TRUE,
                                                   FALSE)))
        
        # add saccade info:
        # get separate cols for eye samples: 
        data_fix <- data_fix %>% 
          separate(base_data, c('timestamp0', 'timestamp1'), 
                   ' ', fill = 'right') %>% 
          mutate(timestamp_eye_0 = substr(timestamp0, 1, nchar(timestamp0)-2), 
                 eye0 = substr(timestamp0, nchar(timestamp0), nchar(timestamp0)),
                 timestamp_eye_1 = substr(timestamp1, 1, nchar(timestamp1)-2), 
                 eye1 = substr(timestamp1, nchar(timestamp1)-1, nchar(timestamp1))) %>% 
          mutate(timestamp_eye_0 = as.numeric(timestamp_eye_0), 
                 timestamp_eye_1 = as.numeric(timestamp_eye_1), 
                 timestamp_eye_1 = if_else(eye0 == 1, timestamp_eye_0, timestamp_eye_1), 
                 timestamp_eye_0 = if_else(eye0 == 1, NA_real_ , timestamp_eye_0)) %>% 
          select(-c(eye0, eye1, timestamp0, timestamp1)) %>% 
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
                 gaze_dev0_hor, 
                 gaze_dev0_vert, 
                 confidence) %>% 
          # drop duplictaes
          # (these are rows with updates from the other eye)
          distinct_at(vars(timestamp_eye_0, gaze_normal0_x, gaze_normal0_y), 
                      .keep_all = TRUE) %>% 
          drop_na(timestamp_eye_0, gaze_normal0_x, gaze_normal0_y) 
        
        timings_eye1 <- data_fix %>% 
          select(timestamp_eye_1, 
                 trial, 
                 gaze_timestamp, 
                 gaze_normal1_x, 
                 gaze_normal1_y, 
                 gaze_dev1_hor, 
                 gaze_dev1_vert, 
                 confidence) %>% 
          # drop duplictaes
          # (these are rows with updates from the other eye)
          distinct_at(vars(timestamp_eye_1, 
                           gaze_normal1_x, 
                           gaze_normal1_y, 
                           gaze_dev1_hor, 
                           gaze_dev1_vert), 
                      .keep_all = TRUE) %>% 
          drop_na(timestamp_eye_1, gaze_normal1_x, gaze_normal1_y)  
        
        # Interpolate bad samples (confidence below given threshold)
        # CAVE: the naming is unfortunate here, interpolation is actually done in 
        # a linear fashion, not as spline!
        if (interpolate_low_conf_samples) {
          if (!nrow(timings_eye0) == 0) {
            timings_eye0 <- timings_eye0 %>% 
              mutate(
                gaze_dev0_hor = spline_interpolate_low_conf_samples(gaze_dev0_hor, 
                                                                    confidence, 
                                                                    0.6), 
                gaze_dev0_vert = spline_interpolate_low_conf_samples(gaze_dev0_vert, 
                                                                 confidence, 
                                                                  0.6))
          }
          
          if (!nrow(timings_eye1) == 0) {
            timings_eye1 <- timings_eye1 %>% 
              mutate(
                gaze_dev1_hor = spline_interpolate_low_conf_samples(gaze_dev1_hor, 
                                                                    confidence, 
                                                                    0.6), 
                gaze_dev1_vert = spline_interpolate_low_conf_samples(gaze_dev1_vert,
                                                                     confidence,
                                                                     0.6))
          }
        }
        
        # Calculate (micro)saccades per eye:
        # msr: micro-saccades right eye
        # msl: -------------- left ----
        
        if (!nrow(timings_eye0) == 0) {
          msr <- timings_eye0 %>% 
            select(gaze_dev0_hor, gaze_dev0_vert) %>% 
            as.matrix() %>% 
            microsacc(sacc_params$vfac,
                      sacc_params$mindur,
                      sacc_params$srate)
        } else {
          msr <- NULL
        }
        
        if (!nrow(timings_eye1) == 0) {
          msl <- timings_eye1 %>% 
            select(gaze_dev1_hor, gaze_dev1_vert) %>% 
            as.matrix() %>%  
            microsacc(sacc_params$vfac,
                      sacc_params$mindur,
                      sacc_params$srate)
        } else {
          msl <- NULL
        }
        
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
        
        for (eye in c('eye_0', 'eye_1')) {
          if (!eye %in% colnames(eye_confidence)) {
            eye_confidence[eye] <- 0
          }
        }
        
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
              sacc          = FALSE,
              sacc_idx      = NA_integer_,
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
                       sacc_idx = if_else((row_n %in% idx_sacc),
                                          sacc_, 
                                          sacc_idx),
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
          sacc_table <- bin
        
          # add info to df:
          data_fix <- data_fix %>% 
            mutate(
              sacc              = FALSE, 
              sacc_idx      = NA_integer_,
              sacc_amp_x    = NA_real_, 
              sacc_amp_y    = NA_real_, 
              sacc_amp      = NA_real_, 
              sacc_peakvel  = NA_real_
            )
          
        # Loop over bin-saccs:
        if (!is.null(sacc_table)) {
          for (binsacc_ in 1:nrow(sac_table)) {
            idx_binsacc <- bin[binsacc_,1]:bin[binsacc_,2]
            
            data_fix <- data_fix %>% 
              mutate(row_n = row_number(),
                       sacc = if_else((row_n %in% idx_binsacc),
                                          TRUE, 
                                          sacc), 
                       sacc_idx = if_else((row_n %in% idx_binsacc),
                                        binsacc_, 
                                        sacc_idx),
                       sacc_amp_x = if_else((row_n %in% idx_binsacc), 
                                               mean(bin[binsacc_, 6], bin[binsacc_, 13]),
                                               sacc_amp_x),
                       sacc_amp_y = if_else((row_n %in% idx_binsacc),
                                               mean(bin[binsacc_, 7], bin[binsacc_, 14]), 
                                               sacc_amp_y),
                       sacc_amp = if_else((row_n %in% idx_binsacc),
                                             sqrt(sacc_amp_x^2 + sacc_amp_y^2),
                                             sacc_amp),
                       sacc_peakvel = if_else((row_n %in% idx_binsacc),
                                                 mean(bin[binsacc_, 3], bin[binsacc_, 10]), 
                                                 sacc_peakvel))
            }
          }
        }
            
        sacc_summary <- data_fix %>% 
          mutate(sub_id = sub_id, 
                 block = block_nr) %>% 
          select(sub_id, 
                 block,
                 trial,
                 sacc_idx, 
                 sacc_amp_x, 
                 sacc_amp_y, 
                 sacc_amp, 
                 sacc_peakvel, 
                 blink, 
                 c_Ecc,
                 c_StimN,
                 CueDir,
                 gaze_timestamp, 
                 confidence) %>% 
          filter(!blink) %>% 
          mutate(sacc_idx = sacc_idx - min(sacc_idx, na.rm=T) + 1) %>% 
          group_by(sacc_idx) %>% 
          mutate(confidence_mean = mean(confidence, na.rm=T)) %>% 
          drop_na() %>% 
          slice(1)
        
        sacc_summary <- sacc_summary %>%  
          mutate(reject_eeg = 
                   ifelse(sacc_amp > sacc_params$rej_threshold & 
                          gaze_timestamp > (-timings$eeg_bl) & 
                          gaze_timestamp < (timings$stim + timings$retention) &
                          confidence_mean > sacc_params$conf_threshold, 
                          TRUE, 
                          FALSE))    
        
        sacc_list_idx <- sacc_list_idx + 1
        sacc_list[[sacc_list_idx]] <- sacc_summary  
        
        if (!suppress_plotting) {
          colors <- c('normal'          = 'black', 
                      'saccade'         = 'red', 
                      'blink'           = 'blue',
                      'blink & saccade' = 'purple')
          
          # y intercept for lower boundary of confidence subplot:
          ymin_conf_subplt <- 5
          
          plt <- data_fix %>% 
            mutate(hor_dev = ifelse(rep(chosen_eye, nrow(data_fix)) == 0, 
                                    rollmean(na.spline(gaze_dev0_hor), 1, 
                                             na.pad = TRUE), 
                                    rollmean(na.spline(gaze_dev1_hor), 1, 
                                             na.pad = TRUE)),  
                   
                   plt_type = case_when(
                     (blink & sacc)    ~ 'blink & saccade',
                     blink             ~ 'blink',
                     sacc              ~ 'saccade',
                     TRUE              ~ 'normal'),  
                   plt_type = factor(plt_type, levels = names(colors)),
                   plt_alpha = if_else(plt_type == '0', 
                                       0.3, 
                                       0.7)) %>% 
            # Data line colored by classification (blink || saccade, normal)
            ggplot(aes(x = gaze_timestamp, 
                       y = hor_dev), 
                   size = 0.1) + 
            geom_line(aes(color = plt_type, 
                          group = trial, 
                          alpha = plt_alpha)) +
            
            # vert. lines indicating trial events:
            geom_vline(xintercept = c(-timings$cue, 
                                      0, 
                                      timings$stim, 
                                      timings$stim + timings$retention), 
                       alpha = 0.3, 
                       color = 'brown', 
                       linetype = 2) +
            
            # hor. lines for confidence subplot:
            geom_hline(yintercept = c(0, 0.5, 1) + ymin_conf_subplt, 
                       alpha = 0.5) +
            
            # data line for confidence subplot:
            geom_line(data = data_pupils %>%  
                        filter(eye_id == chosen_eye) %>% 
                        slice(which(world_index %in% data_fix$world_index)) %>% 
                        mutate(pupil_timestamp = pupil_timestamp - t_stimonset), 
                      aes(x = pupil_timestamp, 
                          y = rollmean(confidence, 5, na.pad = T) + ymin_conf_subplt)) +
            
            # vertical labels indicating trial events: 
            annotate(geom = 'text', 
                     x = c(-timings$cue, 
                           0, 
                           timings$stim, 
                           timings$stim + timings$retention), 
                     y = -Inf-1,
                     hjust = 0,
                     size = 3,
                     label = c('onset cue', 
                               'onset stimulus', 
                               ifelse(block_nr == 'Block2', 'color change', 'start retention'), 
                               ifelse(block_nr == 'Block2', '', 'end retention')), 
                     angle = 90) + 
            
            # # red trial rejection label:
            # annotate(geom = 'label',
            #          x = as.numeric(ifelse(any(data_fix$sacc_amp > sacc_params$rej_threshold & 
            #                                      (!data_fix$blink) & 
            #                                  data_fix$gaze_timestamp > (-timings$eeg_bl) & 
            #                                  data_fix$gaze_timestamp < (timings$stim + timings$retention)),
            #                                0, NA)),
            #          y = Inf,
            #          vjust = 1,
            #          color = 'red',
            #          label = "reject") +
            # 
            # zero line:
            geom_hline(yintercept = 0, 
                       alpha = 0.2) +
            
            # arrow indicatiing cue direction:
            geom_segment(aes(x = -timings$cue, 
                             y = 1 * if_else(CueDir == 'Left', -1, 1), 
                             xend = -timings$cue, 
                             yend = 1.5 * if_else(CueDir == 'Left', -1, 1)), 
                         arrow = arrow(length = unit(0.05, 'npc'), 
                                       type = 'closed')) +
            ylim(-7,7) + 
            theme_classic() + 
            theme(legend.direction = 'vertical',
                  legend.position = 'none',
                  legend.title = element_blank(),
                  axis.title = element_blank()) +
            guides(alpha = FALSE) +
            scale_color_manual(values = colors, 
                               breaks = c('blink', 
                                          'saccade', 
                                          'blink & saccade'), 
                               drop = FALSE) 
        
          # Decide if rejection label shall be added:
          if (any(sacc_summary$reject_eeg)) {
          # (any(any(
          #   sacc_summary$sacc_amp > sacc_params$rej_threshold & 
          #   sacc_summary$gaze_timestamp > (-timings$eeg_bl) & 
          #   sacc_summary$gaze_timestamp < (timings$stim + timings$retention) &
          #   sacc_summary$confidence_mean > sacc_params$conf_threshold))) {
          
            plt <- plt + 
              # red trial rejection label:
              annotate(geom = 'label',
                       x = 0,
                       y = Inf,
                       vjust = 1,
                       color = 'red',
                       label = "reject") 
          }
              
          plt_list[[trial]] <- plt 
        }
                   
      }
      
      # print timing info: 
      time_elapsed <- pracma::toc(echo = FALSE)
      subs_done <- which(sub_id == sub_ids)
      blocks_done <- (which(block_nr == blocks))
      s_per_block <- time_elapsed / blocks_done
      time_remain <- (length(blocks) - blocks_done) * s_per_block + 
        length(blocks) * s_per_block * (length(sub_ids) - subs_done)
      ui_info(glue('Finishing {block_nr} for {sub_id}. 
                     Running for {format(time_elapsed/60, digits = 3)} minutes.  
                     Approx. {format(time_remain/60, digits = 3)} minutes remaining.'))
      
      if (!suppress_plotting) {
        li <- plt_list[1:72]
        fig <- ggarrange(plotlist = li, ncol = 6, nrow = 12, common.legend = TRUE)
        fname_plot <- file.path(path_plots, str_c(sub_id, '_', block_nr, '.pdf'))
        ggsave(file=fname_plot, 
               plot=fig, 
               device = 'pdf', 
               width = unit(20, 'cm'), 
               height = unit(30, 'cm'))
      }
  }
  
  
  
  # Build sacc info DF for current subject:
  sacc_info <- bind_rows(sacc_list) %>% 
    mutate(c_Ecc = factor(c_Ecc, levels = c("4", "9", "14")), 
           block = factor(block, levels = str_c('Block', c(2,4:13))),
           trial_phase = case_when(
             (gaze_timestamp > -timings$cue) & (gaze_timestamp < 0) ~ 'Cue', 
             (gaze_timestamp < timings$stim) & (gaze_timestamp > 0) ~ 'Encoding', 
             (gaze_timestamp > timings$stim) & 
               (gaze_timestamp < (timings$stim + timings$retention)) ~ 'Retention', 
             TRUE ~ 'other'), 
           trial_phase = factor(trial_phase, levels = c('Cue',
                                                        'Encoding',
                                                        'Retention',
                                                        'other'))) 
  # Save RDS with sacc list for this subject:
  if (overwrite_existing_dataoutput) {
    fname_saccdata <- file.path(path_data_sub_out, 
                                str_c('saccdata_', sub_id, '.rds'))
    saveRDS(sacc_info, 
            file = fname_saccdata) 
  }
   
  # Plot main sequence: 
  if(!suppress_plotting) {
    fig_mseq_x <- sacc_info %>% 
      ggplot(aes(x = abs(sacc_amp_x), 
                 y = sacc_peakvel, 
                 col = c_Ecc)) + 
      facet_grid(block~trial_phase) +
      geom_point(aes(shape = reject_eeg)) +
      theme_bw() +
      theme(aspect.ratio = 1) +
      xlim(0,15) +
      ylim(0, 1000) + 
      labs(x = "Abs. horiz. amplitude (dva)", 
           y = "Peak velocity (dva/s)", 
           color = "Stimulus Eccentricity", 
           shape = "Reject Epoch?")  
    
    ggsave(file = file.path(path_plots, str_c(sub_id, '_MainSequenceHoriz', '.pdf')),
           plot = fig_mseq_x, 
           device = 'pdf', 
           width = unit(15, 'cm'), 
           height = unit(30, 'cm'))
    
    # filter out perc block and plot main sequence for all blocks:
    fig_mseq <- sacc_info %>%
      filter(block != 'Block2') %>% 
      ggplot(aes(x = abs(sacc_amp), y = sacc_peakvel, 
                 col = c_Ecc)) + 
      facet_grid(~trial_phase) +
      geom_point(size = 1) +
      theme_bw() + 
      theme(aspect.ratio = 1) +
      xlim(0,15) +
      ylim(0, 1000) + 
      labs(x = "Amplitude (dva)", 
           y = "Peak velocity (dva/s)", 
           color = "Stimulus Eccentricity")
    
    ggsave(file = file.path(path_plots, str_c(sub_id, '_MainSequence', '.pdf')),
           plot = fig_mseq, 
           device = 'pdf', 
           width = unit(20, 'cm'), 
           height = unit(7.5, 'cm'))
  }
  
  # write out info which trials to reject: 
  rej_trials <- sacc_info %>% 
    ungroup() %>% 
    filter(block != "Block2", 
           reject_eeg == TRUE) %>% 
    mutate(block_int = parse_integer(str_remove(as.character(block),
                                                'Block')) - 4, 
           tot_trial = block_int * 72 + trial) %>% 
    select(tot_trial) 
  
  rej_trials <- unique(rej_trials)
  fname <- file.path(path_data_rejepo_out, 
                     str_c(sub_id, '-rejTrials-ET.csv'))
  write_csv2(rej_trials, fname, col_names = F)
  
}
