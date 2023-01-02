"""
===========
Config file
===========
Configuration parameters for the study. This should be in a folder called
``library/``.

Code in parts inspired by:
https://github.com/mne-tools/mne-biomag-group-demo/blob/master/scripts/processing/library/config.py

"""

import os
import os.path as op
from pathlib import Path

# Paths:

paths = dict()

path_study = Path(os.path.abspath(__file__)).parents[3]
print(f"Study path is set to: {path_study}")
paths["study"] = path_study
paths["analyses"] = Path(os.path.abspath(__file__)).parents[2]

path_data = os.path.join(path_study, "Data2022")
paths["00_raw"] = os.path.join(path_data, "DataMNE", "EEG", "00_raw")
paths["01_prepared"] = os.path.join(path_data, "DataMNE", "EEG", "01_prepared")
paths["01_prepared-events"] = os.path.join(
    path_data, "DataMNE", "EEG", "01_prepared-events"
)
paths["02_epochs"] = os.path.join(path_data, "DataMNE", "EEG", "02_epochs")
paths["03_preproc"] = op.join(path_data, "DataMNE", "EEG", "03_preproc")
paths["03_preproc-ica"] = os.path.join(path_data, "DataMNE", "EEG",
                                       "03_preproc", "ica")
paths["03_preproc-ica-ar"] = os.path.join(
    path_data, "DataMNE", "EEG", "03_preproc", "ica", "ar"
)
paths["03_preproc-ica-eog"] = os.path.join(
    path_data, "DataMNE", "EEG", "03_preproc", "ica", "eog"
)
paths["03_preproc-ar"] = os.path.join(path_data, "DataMNE", "EEG",
                                      "03_preproc", "ar")
paths["03_preproc-rejectET"] = op.join(paths["03_preproc"], "reject-ET")
paths["03_preproc-rejectET-CSVs"] = op.join(
    paths["03_preproc"], "reject-ET", "CSV_rejEpos_ET"
)
paths["03_preproc-pooled"] = op.join(paths["03_preproc"], "pooled")
paths["04_evokeds"] = op.join(path_data, "DataMNE", "EEG", "04_evokeds")
paths["04_evokeds-pooled"] = op.join(paths["04_evokeds"], "pooled")
paths["04_evokeds-CDA"] = op.join(paths["04_evokeds"], "CDA")
paths["04_evokeds-PNP"] = op.join(paths["04_evokeds"], "PNP")
paths["05_tfrs"] = op.join(path_data, "DataMNE", "EEG", "05_tfrs_clean")
paths["05_tfrs-summaries"] = op.join(paths["05_tfrs"], "summaries")
paths["06_decoding"] = op.join(path_data, "DataMNE", "EEG", "06_decoding")
paths["06_decoding-sensorspace"] = op.join(paths["06_decoding"], "sensorspace")
paths["06_decoding-csp"] = op.join(paths["06_decoding"], "csp")

paths["plots"] = op.join(path_study, "Plots2022")
paths["extracted_vars_dir"] = op.join(paths["analyses"])

for p in paths:
    if not op.exists(paths[p]):
        os.makedirs(paths[p])
        print("creating dir: " + paths[p])

# Add paths to files:
paths["extracted_vars_file"] = op.join(
    paths["extracted_vars_dir"], "VME_extracted_vars.json"
)


# conditions:
factor_levels = [
    load + ecc
    for load in ["LoadLow", "LoadHigh", ""]
    for ecc in ["EccS", "EccM", "EccL", ""]
][:-1]

factor_dict = {name: factor_levels.index(name) for name in factor_levels}

# ROIs:
chans_CDA_dict = {
    "Left": ["P3", "P5", "PO3", "PO7", "O1"],
    "Right": ["P4", "P6", "PO4", "PO8", "O2"],
    "Contra": ["P3", "P5", "PO3", "PO7", "O1"],
    "Ipsi": ["P4", "P6", "PO4", "PO8", "O2"],
}
chans_CDA_all = [ch for v in list(chans_CDA_dict.values())[0:2] for ch in v]

# Freqs:
alpha_freqs = [8, 13]

# times:
times_dict = dict(
    CDA_start=0.450,
    CDA_end=1.450,
    blink_dur=0.8,
    fix_dur=0.8,
    cue_dur=0.8,
    stim_dur=0.2,
    retention_dur=2.0,
    bl_dur_erp=0.2,
    bl_dur_tfr=0.2,
)

# parallelization:
n_jobs = 50  # adapt this to the number of cores in your machine.

# subjects:
n_subjects_total = 27
ids_missing_subjects = [11, 14, 19]
ids_excluded_subjects = [12, 13, 22]

# font sizes:
plt_label_size = 7
plt_fontname = "Helvetica"  # 'Comic Sans Ms'  #

# colors:
# "#66C2A5" "#3288BD"

orange_blue = ("#fb8500", "#023047")  # old: ("#F1942E", "#32628A")
orange_red = ("#fb8500", "#9E0031")

colors = dict()
colors["LoadHigh"], colors["LoadLow"] = orange_red
colors["Load High"] = colors["LoadHigh"]
colors["Load Low"] = colors["LoadLow"]
colors["4"] = colors["LoadHigh"]
colors["2"] = colors["LoadLow"]
colors["Ipsi"] = "purple"
colors["Contra"] = "pink"

black_red_purple = ("#242424", "#8b1e3f", "#ab81cd")
light_greens = ("#70A9A1", "#9EC1A3", "#CFE0C3")
green_blue_red = ("#70C1B3", "#7B8CDE", "#A4778B")
blue_green_red = ("#30C5FF", "#56E39F", "#899E8B")
blue_greens = ("#0373CC", "#009485", "#8CCC4D")
blue_greens2 = ("#1A1A80", "#009485", "#8CCC4D")
purples = ("#5D2A42", "#FB6376", "#FF9999")
purples2 = ("#5B2C6F", "#8E44AD", "#BB8FCE")
greens = ("#0E6655", "#138D75", "#45B39D")
greys = ("#909497", "#A6ACAF", "#D7DBDD")

cols = blue_greens2  # purples  #
colors["4°"], colors["9°"], colors["14°"] = cols
redsandorange = ("#330f0a", "#ffb703", "#a26769")
colors["EccS"] = colors["4°"]
colors["EccM"] = colors["9°"]
colors["EccL"] = colors["14°"]
colors["Chance"] = "#B5B4B3"
colors["Random"] = "#B5B4B3"
colors["Load"] = "black"  # "#72DDED"
colors["Diff"] = "black"
colors["Ipsi"] = "#FAC748"
colors["Contra"] = "#8390FA"
colors["All"] = "black"  # "#3288BD"
colors["CDA"] = "#FAC748"
colors["PNP"] = "#8390FA"

# labels
labels = dict()
labels["EccS"] = "4°"
labels["EccM"] = "9°"
labels["EccL"] = "14°"
labels["LoadLow"] = "2"
labels["LoadHigh"] = "4"
labels["Ecc"] = "Eccentricity"
labels["Load"] = "Size Memory Array"
labels["Chance"] = "Random"
labels["Random"] = "Random"
labels[""] = "All"

event_dict = {
    "CueL": [
        "Stimulus/S150",
        "Stimulus/S152",
        "Stimulus/S154",
        "Stimulus/S156",
        "Stimulus/S158",
        "Stimulus/S160",
        "Stimulus/S162",
        "Stimulus/S164",
        "Stimulus/S166",
        "Stimulus/S168",
        "Stimulus/S170",
        "Stimulus/S172",
    ],
    "CueR": [
        "Stimulus/S151",
        "Stimulus/S153",
        "Stimulus/S155",
        "Stimulus/S157",
        "Stimulus/S159",
        "Stimulus/S161",
        "Stimulus/S163",
        "Stimulus/S165",
        "Stimulus/S167",
        "Stimulus/S169",
        "Stimulus/S171",
        "Stimulus/S173",
    ],
    "LoadLow": [
        "Stimulus/S151",
        "Stimulus/S153",
        "Stimulus/S159",
        "Stimulus/S161",
        "Stimulus/S167",
        "Stimulus/S169",
        "Stimulus/S150",
        "Stimulus/S152",
        "Stimulus/S158",
        "Stimulus/S160",
        "Stimulus/S166",
        "Stimulus/S168",
    ],
    "LoadHigh": [
        "Stimulus/S155",
        "Stimulus/S157",
        "Stimulus/S163",
        "Stimulus/S165",
        "Stimulus/S171",
        "Stimulus/S173",
        "Stimulus/S154",
        "Stimulus/S156",
        "Stimulus/S162",
        "Stimulus/S164",
        "Stimulus/S170",
        "Stimulus/S172",
    ],
    "EccS": [
        "Stimulus/S151",
        "Stimulus/S153",
        "Stimulus/S155",
        "Stimulus/S157",
        "Stimulus/S150",
        "Stimulus/S152",
        "Stimulus/S154",
        "Stimulus/S156",
    ],
    "EccM": [
        "Stimulus/S159",
        "Stimulus/S161",
        "Stimulus/S163",
        "Stimulus/S165",
        "Stimulus/S158",
        "Stimulus/S160",
        "Stimulus/S162",
        "Stimulus/S164",
    ],
    "EccL": [
        "Stimulus/S167",
        "Stimulus/S169",
        "Stimulus/S171",
        "Stimulus/S173",
        "Stimulus/S166",
        "Stimulus/S168",
        "Stimulus/S170",
        "Stimulus/S172",
    ],
}
