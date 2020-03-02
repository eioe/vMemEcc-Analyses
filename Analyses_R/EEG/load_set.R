

# All/most code from www.mattcraddock.com/blog/2017/11/17/loading-eeglab-set-files-in-r-part-2

library(eegUtils)
library(R.matlab)
library(tibble)
library(purrr)

data_path <- 'D:/Felix/Seafile/Experiments/vMemEcc/Data/PilotData/EEG/01_preprocessed'
data_file_name <- 'P07_hp0.01_lp45_epo_blrem_rejepo_rejcomp.set'
data_file_path <- file.path(data_path, data_file_name)

eeg_data <- readMat(data_file_path)
EEG <- eeg_data$EEG
rm(eeg_data)

## Get formated chanlocs:

var_names <- dimnames(EEG)[[1]]
chanlocs_info <- EEG[[which(var_names == "chanlocs")]]
col_names <- dimnames(chanlocs_info)
size_chans <- dim(chanlocs_info)
# write NAs into empty fields
chanlocs_info <- lapply(chanlocs_info, function(x) ifelse(is_empty(x), NA, x))
# reformat:
dim(chanlocs_info) <- size_chans
dimnames(chanlocs_info) <- col_names

chanlocs_info <- as_tibble(t(as.data.frame(chanlocs_info)))
# unlist entries:
chanlocs_info <- as_tibble(data.frame(lapply(chanlocs_info, unlist), stringsAsFactors = FALSE))

chanlocs_df <- as.data.frame(t(rbind(chanlocs[, ,])))




                             