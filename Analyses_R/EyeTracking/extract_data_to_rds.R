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
overwrite_existing_dataoutput <- FALSE
suppress_plotting <- TRUE

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
  vfac = 6, 
  #  MINDUR		  minimal saccade duration, 
  mindur = 4, 
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

# load subject list and exclude bad subjects:
sub_ids <- str_c('VME_S', str_pad(setdiff(1:27, c(11,14,19)), 
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
  path_data_rejepo_out <- file.path(path_data, '..', 'DataR', 'CSV_rejEpos_ET')
  # checkmake_dirs(c(path_plots, path_data_sub_out, path_data_rejepo_out))
  
  blocks <- list.files(path_data_sub)
  # skip trainings blocks & output dirs:
  blocks <- blocks[!blocks %in% c('Block1', 'Block3', 'R_plots', 'R_data')]
  # sort ascending:
  blocks <- blocks[order(parse_number(blocks))]
  
  # start timer:
  tic()
  
  # data holder:
  blockdata_list <- list()
  
  # loop over blocks for this sub:
  for (block_nr in blocks) {
    
    ## TODO: The following prevents plotting-only runs - you might want to fix this.
    ## Check if output already exists and skip accordingly:
    # if ((length(list.files(path_data_sub_out)) > 0) && !overwrite_existing_dataoutput) {
    #   ui_info(glue("Skipping {block_nr} for {sub_id}. Output exists already \\
    #                  and overwriting is off."))
    #   next
    # }
    
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
    
    # Initialize data holder: 
    
    trialdata_list <- list()
    
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
      
      timings_eye0$eye <- 0
      timings_eye1$eye <- 1
      timings_eye <- bind_rows(timings_eye0, timings_eye1)
      timings_eye$block_nr <- block_nr
      trialdata_list[[trial]] <-  timings_eye
    }
    
    blockdata <- bind_rows(trialdata_list)
    blockdata_list[[block_nr]] <- blockdata
    
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
    
  }
  
  subdata <- bind_rows(blockdata_list, .id = sub_id)
  fname <- file.path(path_data_sub_out, glue("allgazedata-{sub_id}.rds"))
  saveRDS(subdata, fname)
}