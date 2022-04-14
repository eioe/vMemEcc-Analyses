import os
from os import path as op
import shutil

path_main = "C:\\Users\\Felix\\Seafile\\Experiments\\vMemEcc\\Data2022\\DataMNE\\EEG\\06_decoding\\csp\\stimon\\collapsed"
os.listdir(path_main)

conds = ['LoadLow_vs_LoadHigh',
    'LoadLowEccL_vs_LoadHighEccL',
 'LoadLowEccM_vs_LoadHighEccM',
 'LoadLowEccS_vs_LoadHighEccS']

path_gf = 'C:\\Users\\Felix\\Seafile\\Experiments\\vMemEcc\\Data2022\\DataMNE\\EEG\\06_decoding\\csp\\stimon\\collapsed\\LoadLowEccS_vs_LoadHighEccS\\roc_auc\\shrinkage0.4\\VME_S01\\scores'
good_files = {}
for sp in ["patterns", "scores"]:


    reg = "shrinkage0.4"
    for cond in conds:
        path_subs = op.join(path_main, cond, "roc_auc", reg, "labels_shuffled")
        for sub in os.listdir(path_subs):
            if "shuffled" in sub:
                continue
            path_sub = op.join(path_subs, sub)
            for sp in ["scores", "patterns"]:
                good_files["times"] = op.join(path_subs, 'VME_S01', sp, f"{sp}_per_sub__times.npy")
                good_files["freqs"] = op.join(path_subs, 'VME_S01', sp, f"{sp}_per_sub__freqs.npy")
                path_sp = op.join(path_sub, sp)
                for tf in ["times", "freqs"]:
                    good_file = op.join(path_subs, 'VME_S01', sp, f"{sp}_per_sub__{tf}.npy")
                    fname = f"{sp}_per_sub__{tf}.npy"
                    if not op.exists(op.join(path_sp, fname)):
                        shutil.copy(good_file, path_sp)
                        print(f"{sub} --- {sp} --- {tf}")