"""
:module: drivers.build_waveform_dataset.fetch_raw_waveforms_bulk
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: CC-BY-4.0

:purpose:
    This script generates bulk waveform requests for the IRIS webservice
    by EVID and saves successful waveform queries into EVID labeled
    subdirectories as bulkXtpY.mseed with X and Y indicating integer
    second front and back padding around p-wave arrival time data for
    each labeled arrival (tp). 

    e.g., EVID1234567/bulk20tp40.mseed
        Waveform data for event ID 1234567 with 20 seconds leading and
        40 seconds trailing the P-arrival time for each station with
        input arrival data.

    The script also assesses the number of unique channels and number of
    traces associated with each station associated with an arrival and
    writes subset CSV's with these statistics appended to the affiliated
    EVID directory.

:notes:
    Padding is set to 25tp45 in order to produce 70 second long records
    that can be used in other ML applications besides E3WS, which really only 
    requires 10 seconds of data (best fit in Lara et al. (2023) is 7tp3), such
    as EQTransformer (Mousavi et al., 2019) which has an input vector length of
    3x6000 and was trained on 100 sps data (i.e., 60 sec of 3-C records).


:references:
Pablo Lara, Quentin Bletery, Jean-Paul Ampuero, Adolfo Inza, Hernando Tavera.
    Earthquake Early Warning Starting From 3 s of Records on a Single Station
    With Machine Learning. Journal of Geophysical Research: Solid Earth.
    https://doi.org/10.1029/2023JB026575

Mousavi, S. Mostafa, William L. Ellsworth, Weiqiang Zhu, Lindsay Y. Chuang, and
    Gregory C. Beroza. “Earthquake Transformer—an Attentive Deep-Learning Model
    for Simultaneous Earthquake Detection and Phase Picking.” Nature Communications
    11, no. 1 (August 7, 2020): 3952. https://doi.org/10.1038/s41467-020-17591-w.

"""

import os
import sys
import pandas as pd
import numpy as np
from time import time
from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from tqdm import tqdm

sys.path.append("..", "..")
import core.preprocess as pp
from util.time import unix_to_epoch, UTCDateTime_to_Timestamp

#### USER CONTROLS ####
dtlead = 25.0  # [seconds] inital record pull in advance of P-pick
dtlag = 45.0  # [seconds] initial record pull trailing P-pick
verbose = False

# Define absolute path to
ROOT = os.path.join("..", "..", "..")
DROOT = os.path.join(ROOT, "PNSN_data")
# Define waveform file formatting
st_str = "EVID{EVID}/{NET}.{STA}.{LOC}.{CHANS}.{ARID}.mseed"
bulk_str = "EVID{EVID}/bulk{lead}tp{lag}.mseed"
DATA_FSTR = os.path.join(DROOT, bulk_str)
# Define save location for metadata
md_str = "EVID{EVID}/event_mag_phase_nwf.csv"
META_FSTR = os.path.join(DROOT, md_str)


# Map path to phase file
pick_file = os.path.join(
    "..",
    "..",
    "PNSN_metadata",
    "AQMS_Queries",
    "Event_Mag_Phase",
    "AQMS_event_mag_phase_query_output.csv",
)
# Load pick file from CSV
df = pd.read_csv(pick_file)
# Assign ARID as index
df.index = df.arid
# Reformat epoch datetimes into UTCDateTimes
df.datetime = df.datetime.apply(lambda x: unix_to_epoch(x, output_format=UTCDateTime))
df.arrdatetime = df.arrdatetime.apply(
    lambda x: unix_to_epoch(x, output_format=UTCDateTime)
)

# Limit data to UW network
df = df[df.net == "UW"]

# Initialize waveform / metadata client
client = Client("iris")

print("Starting data pull by EVID")
nevents = len(df.evid.unique())
# Iterate across event ID's
for _i, _evid in tqdm(enumerate(df.evid.unique()), disable=verbose):
    if os.path.isfile(META_FSTR.format(EVID=_evid)):
        if verbose:
            print(f"({_i + 1}/{nevents}) Summary file exists, skipping EVID {_evid}")
    else:
        if verbose:
            print(f"({_i + 1}/{nevents}) EVID {_evid}")
        # Pull subset dataframe
        _df = df[df.evid == _evid]
        # Make holder for arrival ID number of channels with waveforms
        arid_nchans = []
        #### QUERY COMPOSITION ####
        # Compose bluk request
        bulk = []
        _arids = _df.index.values
        # Iterate across arrival IDs
        for _arid in _df.index:
            # Pull series from subset dataframe
            _ser = _df.loc[_arid, :]
            if _ser.location in [" ", "  ", "   ", " ", "  ", "   "]:
                location = ""
            else:
                location = _ser.location
            # Compose bulk line
            bline = (
                _ser.net,
                _ser.sta,
                "*",
                _ser.seedchan[:2] + "?",
                _ser.arrdatetime - dtlead,
                _ser.arrdatetime + dtlag,
            )
            # Append line to bluk
            bulk.append(bline)

        #### DATA FETCH ####
        # Run bluk request
        t0 = time()
        try:
            st = client.get_waveforms_bulk(bulk)
            if len(st) == 0:
                empty_stream = True
            else:
                empty_stream = False
        except:
            empty_stream = True
            st = Stream()
        t1 = time()
        if verbose:
            print(
                f"Bulk request of {len(bulk)} stations returned {len(st)} traces in {t1 - t0 :.3f} sec"
            )
        if empty_stream:
            df_out = pd.concat(
                [_df, pd.DataFrame(columns=["nchan_wf", "ntr_wf"], index=_df.index)],
                axis=1,
                ignore_index=False,
            )
            # Write filtered metadata to disk
            oname = META_FSTR.format(EVID=_evid)
            if not os.path.exists(os.path.split(oname)[0]):
                os.makedirs(os.path.split(oname)[0])
            df_out.to_csv(oname, header=True, index=False)
        else:
            #### SAVE BULK DATA STREAM TO DISK ####
            rname = DATA_FSTR.format(EVID=_evid, lead=int(dtlead), lag=int(dtlag))
            if not os.path.exists(os.path.split(rname)[0]):
                os.makedirs(os.path.split(rname)[0])

            st.write(rname, fmt="mseed")

            #### GET NUMBER OF TRACES PER
            arid_nchans = []
            for line in bulk:
                # Get number of unique stachan
                _st = st.select(station=line[1])
                _ntr = len(_st)
                chlist = []
                for _tr in _st:
                    if _tr.stats.channel not in chlist:
                        chlist.append(_tr.stats.channel)
                _nch = len(chlist)
                # Get matching ARID
                try:
                    _arid = _df[_df.sta == line[1]].index.values[0]
                except:
                    breakpoint()
                arid_nchans.append([_arid, _nch, _ntr])
            #### METADATA FILTER AND SAVE ####
            # Attach number of channels and traces per instrument instance in event
            arid_nchans = np.array(arid_nchans)
            df_arid_log = pd.DataFrame(
                arid_nchans[:, 1:],
                columns=["nchan_wf", "ntr_wf"],
                index=arid_nchans[:, 0],
            )
            df_out = pd.concat([_df, df_arid_log], axis=1, ignore_index=False)
            # Filter down to just arrivals with waveform data downloaded
            df_out = df_out[df_out.nchan_wf.notna()]
            # Put datetimes into pandas-friendly format
            df_out.datetime = df_out.datetime.apply(
                lambda x: UTCDateTime_to_Timestamp(x)
            )
            df_out.arrdatetime = df_out.arrdatetime.apply(
                lambda x: UTCDateTime_to_Timestamp(x)
            )
            # Write filtered metadata to disk
            df_out.to_csv(META_FSTR.format(EVID=_evid), header=True, index=False)


## THIS GETS CUMBERSOME ##
# #### DATA SAVE ####
# # Split out by query element
# for _i, line in enumerate(bulk):
#     t0 = time()
#     qkw = {"station": line[1]}

#     # _st = pp.sort_by_components(st.copy().select(**qkw).merge())
#     _st = st.select(**qkw)
#     nsel = len(_st)
#     t1 = time()
#     print(f"Selection took {t1 - t0:.3f} sec")
#     if len(_st) > 1:
#         _st = _st.merge()

#     nmer = len(_st)
#     # Get component codes
#     t2 = time()
#     print(f"Merge took {t2 - t1:.3f} sec ({nsel:d} --> {nmer:d}) traces")
#     _cc = ""
#     for _tr in _st:
#         _cc += _tr.stats.channel[-1]
#     t3 = time()
#     print(f"Channel code wrangling took {t3 - t2:.3f} sec")
#     # Prepare file structure for writing stream to disk
#     rname = DATA_FSTR.format(
#         EVID=_evid,
#         NET=line[0],
#         STA=line[1],
#         LOC=location,
#         CHANS=line[3][:2] + _cc,
#         ARID=_arids[_i],
#         PROC_STATE="raw",
#     )
#     # If directory structure specified` dosen't exist,
#     # make recursive structure
#     if not os.path.exists(os.path.split(rname)[0]):
#         os.makedirs(os.path.split(rname)[0])
#     t4 = time()
#     print(f"Directory stuff too {t4 - t3:.3f} sec")
#     # Save merged waveforms to disk
#     arid_nchans.append([_arids[_i], len(_st.copy().merge())])
#     if len(_st) > 0:
#         _st.split().write(rname, fmt="MSEED")
#     t5 = time()
#     print(f"Saving waveforms took {t5 - t4:.3f} sec")
