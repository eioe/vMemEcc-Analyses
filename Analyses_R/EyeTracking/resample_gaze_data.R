
### Resample gaze data and write to disk


resample_gaze_data_w2disk <- function(sub_ids, path_data, return_df = FALSE) {
  
  if (return_df) 
    holder <- list()
  
  for (sub_id in sub_ids) {
    path_data_sub <- file.path(path_data, sub_id, 'EyeTracking', 'R_data')
    fname <- file.path(path_data_sub, glue("allgazedata-{sub_id}.rds"))
    data <- readRDS(fname)
    data$sub_id <- sub_id
    df_out <- data %>% 
      mutate(gaze_dev_hor = if_else(eye == 0, gaze_dev0_hor, gaze_dev1_hor), 
             gaze_dev_vert = if_else(eye == 0, gaze_dev0_vert , gaze_dev1_vert)) %>% 
      select(sub_id, gaze_timestamp, trial, block_nr, eye, confidence, gaze_dev_hor, gaze_dev_vert) %>% 
      group_by(sub_id,
               trial,
               block_nr,
               eye) %>% 
      et_resample("gaze_timestamp", 
                  srate = 50, 
                  tmin = -1.1, 
                  tmax = 2.2) 
    fname <- file.path(path_data_sub, glue("dataresampled-{sub_id}.rds"))
    saveRDS(df_out, fname)
    print("##########################")
    print(glue("Done with {sub_id}!"))
    print("##########################")
    if (return_df) 
      holder[[sub_id]] <- ouut
  }
  
  if (return_df) 
    df_resampled = bind_rows(holder, .id = 'sub_id')

  return(df_resampled)
}