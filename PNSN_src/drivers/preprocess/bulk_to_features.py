"""
:module: PNSN_src/drivers/preprocess/bulk_to_features.py
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: CC-BY-4.0

:purpose:
    This driver runs feature extraction on a series of staged event data
    and writes `event_magnitude_phase_nwf_PAZ_FV.csv` files to each direcotry
    using the `PNSN_src.core.preprocess.run_event_from_disk()` method

    In this current configuration we use the instrument response deconvolution
    implementation from Burky et al. (2021) to produce acceleration traces in
    m s**-2 and then calculate the 140 features for the method from Lara et al.
    (2023) for earthquake source parameter estimation via an ensemble random
    forest approach.

:references:
    
Burky, A. L., J. C. E. Irving, and F. J. Simons (2021). Instrument Response
    Removal and the 2020 MLg 3.1 Marlboro, New Jersey, Earthquake, Seismol.
    Res. Lett. 92, 3865-3872, https://doi.org/10.1785/0220210118

Pablo Lara, Quentin Bletery, Jean-Paul Ampuero, Adolfo Inza, Hernando Tavera.
    Earthquake Early Warning Starting From 3 s of Records on a Single Station
    With Machine Learning. Journal of Geophysical Research: Solid Earth.
    https://doi.org/10.1029/2023JB026575

:attribution:
    The instrument deconvolution subroutine used in this program is licensed 
    under an MIT license (Copyright (c) 2020 Alexander Burky). 
        See: PNSN_src/contrib/rflexa/LICENSE
"""


import os
import sys
from glob import glob
from tqdm import tqdm
sys.path.append(os.path.join("..", "..", ".."))
import PNSN_src.core.preprocess as pp

# Define root path to inputs
IROOT = os.path.join("..", "..", "..", "PNSN_data")
IGLOB = os.path.join(IROOT, "EVID*")

#### USER CONTROLS ####
pp_lead = 7.0  # [seconds] delta time to trim off from each trace starttime
pp_lag = 3.0  # [seconds] delta time to trim off from each trace endtime
proc_str = "rflexa_default"


save_preprocessed = True
save_csv_fv = True

dir_list = glob(IGLOB)
dir_list.sort()

# Iterate across EVID directories
for _dir in tqdm(dir_list):
    try:
        df_fv, st_in, st_pp = pp.run_event_from_disk(
            _dir, decon_method="PAZ", return_streams=True
        )
        if save_preprocessed and len(st_pp) > 0:
            st_pp.write(
                os.path.join(_dir, f"bulk7tp3_pp_{proc_str}.mseed"),
                fmt="MSEED"
            )
        if save_csv_fv:
            df_fv.to_csv(
                os.path.join(_dir, f"feature_vectors_{proc_str}.csv"),
                header=True,
                index=False,
            )
    except FileNotFoundError:
        pass
    # Run "cleanup" and attach feature vectors to *.csv
