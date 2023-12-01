import os
import sys
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from obspy import read, read_inventory

sys.path.append(os.path.join("..",'..','..'))
import PNSN_src.core.preprocess as pp

# Define root path to inputs
IROOT = os.path.join("..", "..", '..', "PNSN_data")
IGLOB = os.path.join(IROOT, "EVID*")

#### USER CONTROLS ####
pp_lead = 7.0  # [seconds] delta time to trim off from each trace starttime
pp_lag = 3.0  # [seconds] delta time to trim off from each trace endtime
merge_kwargs = {"method": 0} #, "interpolation_samples": -1}  # kwargs for stream.merge()
proc_str = 'rflexa_default'


save_preprocessed = True
save_csv_fv = True

dir_list = glob(IGLOB)
dir_list.sort()

# Iterate across EVID directories
for _dir in tqdm(dir_list[2542:]):#[685+148:]):#tqdm(dir_list[686+50+385+485+437+43+457+65+10+986+16+37+369+40+46+6+12+7+9+4:]):
    try:
        df_fv, st_in, st_pp = pp.run_event_from_disk(
            _dir,
            decon_method='PAZ',
            return_streams=True
        )
        if save_preprocessed and len(st_pp) > 0:
            st_pp.write(os.path.join(_dir, f'bulk7tp3_pp_{proc_str}.mseed'),
                        fmt='MSEED')
        if save_csv_fv:
            df_fv.to_csv(os.path.join(_dir, f'feature_vectors_{proc_str}.csv'), header=True, index=False)
    except FileNotFoundError:
        pass
    # Run "cleanup" and attach feature vectors to *.csv

