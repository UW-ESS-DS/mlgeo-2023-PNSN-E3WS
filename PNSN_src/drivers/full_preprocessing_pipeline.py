import os
import sys
import pandas as pd
import numpy as np
from obspy import Stream, UTCDateTime
from obspy.clients.fdsn import Client
from tqdm import tqdm

sys.path.append("..")
import core.preprocess as pp
from util.time import unix_to_epoch

# Define absolute path to
ROOT = "../../PNSN_data"
# Define waveform file formatting
st_str = "EVID{EVID}/{PROC_STATE}/{NET}.{STA}.{LOC}.{CHANS}.{ARID}.mseed"
DATA_FSTR = os.path.join(ROOT, "data", st_str)
# Define feature vector file formatting
fv_str = "EVID{EVID}/feature_vectors/{NET}.{STA}.{LOC}.{CHANS}.{ARID}.npy"
FEAT_FSTR = os.path.join(ROOT, "data", fv_str)
# Define save location for metadata
META_OUT = os.path.join(ROOT, "metadata")

#### USER CONTROLS ####
dtlead = 20.0  # [seconds] inital record pull in advance of P-pick
dtlag = 20.0  # [seconds] initial record pull trailing P-pick
pp_lead = 7.0  # [seconds] delta time leading p-pick time for pre-processing
pp_lag = 3.0  # [seconds] delta time lagging p-pick time for pre-processing
merge_kwargs = {"method": 1, "interpolation_samples": -1}  # kwargs for stream.merge()

save_raw = True
save_preprocessed = True
save_features = True

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
# Reformat epoch datetimes into UTCDateTimes
df.datetime = df.datetime.apply(lambda x: unix_to_epoch(x, output_format=UTCDateTime))
df.arrdatetime = df.arrdatetime.apply(lambda x: unix_to_epoch(x, output_format=UTCDateTime))

# Initialize waveform / metadata client
client = Client("IRIS")

# Compose Bulk Request
print("Starting data pull")
wfdisc = []
features = []
for _i in tqdm(range(20)):
    # Pull dataframe file
    _ser = df.iloc[_i, :]
    args = [
        _ser.net,
        _ser.sta,
        "*",
        _ser.seedchan[:2] + "?",
        _ser.arrdatetime - dtlead,
        _ser.arrdatetime + dtlag,
        True,
    ]
    query_kwargs = dict(
        zip(
            [
                "network",
                "station",
                "location",
                "channel",
                "starttime",
                "endtime",
                "attach_response",
            ],
            args,
        )
    )
    st = client.get_waveforms(**query_kwargs)
    st = st.merge(**merge_kwargs)
    # Enforce record length with padding if necessary
    st1 = st.copy().trim(
        starttime=query_kwargs["starttime"],
        endtime=query_kwargs["endtime"],
        pad=True,
        nearest_sample=True
    )
    st2 = pp.sort_by_components(st1.copy())
    # Get component codes
    _cc = ""
    for _tr in st2:
        _cc += _tr.stats.channel[-1]
    if _ser.location in [" ", "  ", "   ", " ", "  ", "   "]:
        location = ""
    else:
        location = _ser.location
    # Format savefile name
    rname = DATA_FSTR.format(
        EVID=_ser.evid,
        NET=_ser.net,
        STA=_ser.sta,
        LOC=location,
        CHANS=_ser.seedchan[:2] + _cc,
        ARID=_ser.arid,
        PROC_STATE="raw",
    )
    if save_raw:
        # If directory structure specified in `_path` dosen't exist,
        # make recursive structure
        if not os.path.exists(os.path.split(rname)[0]):
            os.makedirs(os.path.split(rname)[0])

        # Save merged waveforms to disk
        st2.split().write(rname, fmt="MSEED")


    ## Conduct pre-processing
    # Trim a copy of the stream
    st3 = st2.trim(
        starttime=_ser.arrdatetime - pp_lead, endtime=_ser.arrdatetime + pp_lag
    )
    # Run preprocessing
    st4 = pp.preprocess_NRT_pipeline(st3)

    if save_preprocessed:
        # Save stream to disk
        pname = DATA_FSTR.format(
            EVID=_ser.evid,
            NET=_ser.net,
            STA=_ser.sta,
            LOC=location,
            CHANS=_ser.seedchan[:2] + _cc,
            ARID=_ser.arid,
            PROC_STATE="preprocessed",
        )
        if not os.path.exists(os.path.split(pname)[0]):
            os.makedirs(os.path.split(pname)[0])
        st4.split().write(pname, fmt="MSEED")

    # Extract features
    _features = pp.process_feature_vector(st4)
    features.append(_features)
    if save_features:
        _features = np.array(_features, dtype=np.float32)
        fname = FEAT_FSTR.format(
            EVID=_ser.evid,
            NET=_ser.net,
            STA=_ser.sta,
            LOC=location,
            CHANS=_ser.seedchan[:2] + _cc,
            ARID=_ser.arid,
        )
        if not os.path.exists(os.path.split(fname)[0]):
            os.makedirs(os.path.split(fname)[0])
        np.save(fname, _features, allow_pickle=False)
