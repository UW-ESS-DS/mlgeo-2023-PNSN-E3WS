import os
import sys
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from obspy import read, read_inventory

sys.path.append("..")
import core.preprocess as pp

# Define root path to inputs
IROOT = os.path.join("..", "..", "PNSN_data")
IGLOB = os.path.join(IROOT, "EVID*/bulk*.mseed")

#### USER CONTROLS ####
pp_lead = 7.0  # [seconds] delta time to trim off from each trace starttime
pp_lag = 3.0  # [seconds] delta time to trim off from each trace endtime
merge_kwargs = {"method": 0} #, "interpolation_samples": -1}  # kwargs for stream.merge()
proc_str = 'wlirc'


save_preprocessed = False
save_features = True

dir_list = glob(IGLOB)
dir_list.sort()
# Create super-holder for features
FEATURES = []
# Create headers for output feature CSVs
cols = ['arid', 'evid', 'net', 'sta'] + [f'f{x:03d}' for x in range(len(140))]

# Iterate across EVID directories
for _fdir in tqdm(dir_list):
    # Get path and bulk_file elements split
    _dir, bulk_file = os.path.split(_fdir)

    # Scrape EVID from directory name
    _evid = int(os.path.split(_dir)[-1][4:])
    # Load EVID-specific data labels
    df = pd.read_csv(
        os.path.join(_dir, "event_mag_phase_nwf.csv"),
        parse_dates=["datetime", "arrdatetime"],
        index_col="arid"
    )
    # Ensure stations called have at least one trace
    df = df[df.nchan_wf > 0]
    
    # Load inventory
    _inv = read_inventory(os.path.join(_dir, 'station.xml'))
    # Get list of PAZ files
    paz_list = glob(os.path.join(_dir, 'paz', '*.pz'))
    paz_list.sort()                          

    # Sanity check that EVIDs for the directory and metadata match
    if _evid != int(df.evid.unique()[0]):
        breakpoint()

    # Get waveform file & scrape padding info
    pads = bulk_file[4:].split(".")[0].split("tp")
    pads = [float(x) for x in pads]

    # Load Waveforms
    st_bulk = read(_fdir, fmt='MSEED')

    # Create feature holder for this particular EVID
    features = []

    # Iterate across ARIDs
    for _arid in df.index:
        # Extract series line from dataframe
        _ser = df.loc[_arid, :]
        # Select station from stream
        _st = st_bulk.select(station=_ser.sta)
        # Merge traces
        _st.merge(**merge_kwargs)

        # Do preprocessing
        _st_pp = pp.preprocess_rflexa_pipeline(_st, paz_list,
                                               fill_val=0,
                                               sr=100,
                                               filt=[1,2,44,45],
                                               tplead=pads[0] - pp_lead,
                                               tplag=pads[1] - pp_lag)

        # Placeholder for saving preprocessed waveforms
        if save_preprocessed:
            breakpoint()

        #### Extract features ####
        _features = pp.process_feature_vector(_st_pp, asarray=False)

        #### Capture feature vectors ####
        # Add basic indexing info to feature line
        _fline = [_arid, _evid, _ser.net, _ser.sta] + _features

        # Save feature line to event-holder
        features.append(_fline)

        # Save feature line to super-holder
        FEATURES.append(_fline)

    #### Save event-specific feature vectors ####
    df_fv = pd.DataFrame(features, columns=cols)
    df_fv.to_csv(os.path.join(_dir, f'feature_vectors_{proc_str}.csv'),
                 header=True, index=False)

#### Save Super-Feature Vector ####
DF_FV = pd.DataFrame(FEATURES, columns=cols)
DF_FV.to_csv(os.path.join(IROOT, f'complete_feature_vector_set_{proc_str}.csv'))