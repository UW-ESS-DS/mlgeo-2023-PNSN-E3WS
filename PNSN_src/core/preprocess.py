"""
:module: PNSN_src.preprocess
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: CC-BY-4.0

:purpose: 
    This module provides waveform preprocessing methods for
    training/validation/testing of the E3WS ML model
    (Lara et al., 2023) under the assumption that a user starts
    out with an Obspy Stream that contains trace(s) with attached
    instrument response(s). This module currently contains an adaptation
    of the preprocessing pipeline from Lara et al. (2023), using the 
    trace.remove_response() approach to instrument response correction, 
    rather than the trace.simulate(remove_paz) approach used in their publication.

    This was a decision made out of conveneince at the time of writing this
    module and subsequent data acquisition modules provided insighs on a 
    convenient way to extract SAC Poles and Zeros files (*.pz) from an obspy
    Inventory. (see drivers.get_RESP_inv_from_bulk.py)

:TODO: 
    Provide a method that exactly copies the pre-processing pipeline from 
    Lara et al. (2023).

:references:
Pablo Lara, Quentin Bletery, Jean-Paul Ampuero, Adolfo Inza, Hernando Tavera.
    Earthquake Early Warning Starting From 3 s of Records on a Single Station
    With Machine Learning. Journal of Geophysical Research: Solid Earth.
    https://doi.org/10.1029/2023JB026575

"""
import os
import sys
ROOT = os.path.join("..", "..")
from glob import glob
import numpy as np
import pandas as pd
from obspy import Stream, Trace, read, read_inventory
# Import repository specific modules
sys.path.append(ROOT)
import PNSN_src.core.feature_functions as fvf
import PNSN_src.util.resp as resp


def sort_by_components(stream, order="E1N2Z3"):
    """
    Sort a stream by specified channel component codes
    :: INPUTS ::
    :param stream: [obspy.core.stream.Stream]
    :param order: [string] ordered characters for SEED component codes

    :: OUTPUT ::
    :return st_out: [obspy.core.stream.Stream] sorted stream
    """
    st_out = Stream()
    for _c in order:
        _st = stream.copy().select(channel=f"??{_c}")
        if len(_st) > 0:
            st_out += _st
    return st_out


def preprocess_rr_pipeline(
    stream,
    order="E2N1Z3",
    sr=100,
    fill_value=0.0,
    water_level=30,
    fmin1=0.01,
    fmin2=1.0,
    fmax=45.0,
):
    """
    Conduct the preprocessing routine from Lara et al. (2023)
    replacing their Poles and Zeros instrument deconvolution with
    using Stream.simulate() with Stream.remove_response() and
    a specified waterlevel to stabilize the deconvolution

    This approach applys multiple rounds of detrending and
    tapering in order to suppress progressive processing artifacts
    at the edges of windowed data, which is beneficial in a near-
    real-time processing situation when adding extra data are
    prohibitive.

    :: INPUTS ::
    :param stream: [obspy.core.stream.Stream]
                Input stream with 1 or 3 component traces for a single
                seismometer with arbitrary ordering already trimmed
                to specified window length (10 seconds) with a valid
                instrument response attached to each trace under the
                trace.stats.response AttribDict entry.
    :param order: [string]
                String of component codes to iterate across for sampling
                traces from `stream`
    :param water_level: [float]
                waterlevel value to pass to remove_response()
    :param fmin1: [float]
                highpass frequency in Hz for pre-deconvolution filtering
    :param fmin2: [float]
                highpass frequency in Hz for post-deconvolution filtering
    :param fmax: [float]
                lowpass frequency in Hz for each filtering (if input signal)

    :: OUTPUT ::
    :return pp_stream: [obspy.core.stream.Stream]
                Ordered, pre-processed stream
    """
    # Create holder stream
    pp_stream = Stream()
    # Iterate across copies of input traces
    for _c in order:
        _st = stream.copy().select(channel=f"??{_c}")
        if len(_st) == 1:
            _tr = _st[0]
            # Remove a linear slope and the midpoint
            _tr.detrend("demean").detrend("linear")
            # Taper by 5%
            _tr.taper(0.05, "cosine", side="both")
            # apply specified filter
            if _tr.stats.sampling_rate >= fmax * 2:
                _tr.filter(
                    "bandpass", freqmin=fmin1, freqmax=fmax, zerophase=True, corners=4
                )
            else:
                _tr.filter("highpass", freq=fmin1, zerophase=True, corners=4)
            # resample data
            if _tr.stats.sampling_rate != sr:
                _tr.resample(sr)
            # Handle gappy data (may need to do white noise instead of 0-pad...)
            if np.ma.is_masked(_tr.data):
                _tr.data = _tr.data.filled(fill_value=fill_value)
            # remove the mean again to clean up effects of filtering and resampling
            _tr.detrend("demean")
            # remove instrument response
            _tr.remove_response(water_level=water_level, output="ACC")
            # Second signal conditioning - triaging edge effects from deconvolution
            _tr.detrend("demean").detrend("linear")
            _tr.taper(0.05, "cosine", side="both")
            _tr.filter(
                "bandpass", freqmin=fmin2, freqmax=fmax, zerophase=True, corners=4
            )
            pp_stream += _tr
    return pp_stream


def preprocess_pz_pipeline(
    stream,
    order="E2N1Z3",
    sr=100,
    fill_value=0.0,
    water_level=30,
    fmin1=0.01,
    fmin2=1.0,
    fmax=45.0,
):
    """
    Conduct the preprocessing routine from Lara et al. (2023)
    using the Poles and Zeros instrument deconvolution with
    using Stream.simulate().

    This approach applys multiple rounds of detrending and
    tapering in order to suppress progressive processing artifacts
    at the edges of windowed data, which is beneficial in a near-
    real-time processing situation when adding extra data are
    prohibitive.

    :: INPUTS ::
    :param stream: [obspy.core.stream.Stream]
                Input stream with 1 or 3 component traces for a single
                seismometer with arbitrary ordering already trimmed
                to specified window length (10 seconds) with a valid
                instrument response attached to each trace under the
                trace.stats.paz AttribDict entry.
    :param order: [string]
                String of component codes to iterate across for sampling
                traces from `stream`
    :param fmin1: [float]
                highpass frequency in Hz for pre-deconvolution filtering
    :param fmin2: [float]
                highpass frequency in Hz for post-deconvolution filtering
    :param fmax: [float]
                lowpass frequency in Hz for each filtering (if input signal)

    :: OUTPUT ::
    :return pp_stream: [obspy.core.stream.Stream]
                Ordered, pre-processed stream
    """
    # Create holder stream
    pp_stream = Stream()
    # Iterate across copies of input traces
    for _c in order:
        _st = stream.copy().select(channel=f"??{_c}")
        if len(_st) == 1:
            _tr = _st[0]
            # Remove a linear slope and the midpoint
            _tr.detrend("demean").detrend("linear")
            # Taper by 5%
            _tr.taper(0.05, "cosine", side="both")
            # apply specified filter
            if _tr.stats.sampling_rate >= fmax * 2:
                _tr.filter(
                    "bandpass", freqmin=fmin1, freqmax=fmax, zerophase=True, corners=4
                )
            else:
                _tr.filter("highpass", freq=fmin1, zerophase=True, corners=4)
            if _tr.stats.sampling_rate != sr:
                _tr.resample(sr)
            # Handle gappy data (may need to do white noise instead of 0-pad...)
            if np.ma.is_masked(_tr.data):
                _tr.data = _tr.data.filled(fill_value=fill_value)
            # remove the mean again to clean up effects of filtering and resampling
            _tr.detrend("demean")
            # remove instrument response
            _tr.remove_response(water_level=water_level, output="ACC")
            # Second signal conditioning - triaging edge effects from deconvolution
            _tr.detrend("demean").detrend("linear")
            _tr.taper(0.05, "cosine", side="both")
            _tr.filter(
                "bandpass", freqmin=fmin2, freqmax=fmax, zerophase=True, corners=4
            )
            pp_stream += _tr
    return pp_stream


def process_feature_vector(
    pp_stream,
    tkwargs={"thresh": 0.8, "bins": 200, "alpha": 2},
    fkwargs={"N_fft": 1024, "pct_overlap": 75, "bins": 50, "alpha": 2, "thresh": 0.4},
    ckwargs={"numcep": 13, "nfilt": 26, "freqmin": 1.0},
    asarray=True,
):
    """
    Wrapper for extracting the feature vector for 1-C and 3-C waveform data
    that have been pre-processed

    :: INPUTS ::
    :param pp_stream: [obspy.core.stream.Stream]
                3-component, ordered, pre-processed stream
                    OR
                1-component, pre-processed stream OR trace
    :param tkwargs: [dict]
                key-word-arguments for feature_functions.process_temporal()
    :param skwargs: [dict]
                key-word-arguments for feature_functions.process_spectral()
    :param ckwargs: [dict]
                key-word-arguments for feature_functions.process_cepstral()

    :: OUTPUT ::
    :return features: [(140,) numpy.ndarray(dtype=numpy.float32)]
                Feature vector with the following blocks
                0 - 4 : rectilinearity features (set to 0's for 1-C data)
                5 - 21: E-component temporal features
                22- 36: E-component spectral features
                37- 51: E-component cepstral features
                50- 69: N-component temporal features
                67- 85: N-component spectral features
                82- 99: N-component cepstral features
                95-117: Z-component temporal features
               112-126: Z-component spectral features
               127-139: Z-component cepstral features

    """
    features = []
    if isinstance(pp_stream, Trace):
        pp_stream = Stream(pp_stream)
        oneC = True
    # Extract data from stream/trace
    # Handle 1-C data
    if len(pp_stream) == 1:
        oneC = True
        data_ENZ = np.array([pp_stream[0].data for x in range(3)])
    # If there's one valid horizontal and one dead one, still treat like 1C
    elif len(pp_stream) == 2:
        oneC = True
        data_ENZ = np.array([pp_stream[-1].data for x in range(3)])
    # Otherwise compose 3-C data as normal
    elif len(pp_stream) == 3:
        data_ENZ = np.array([pp_stream[x].data for x in range(3)])
        oneC = False
    else:
        breakpoint()
    # Extract relative times of data and sampling rate regardless
    dtimes = pp_stream[0].times()
    sr = pp_stream[0].stats.sampling_rate

    # If working with 1-C data or a dead channel set 3-C features to 0
    if oneC:
        features = list(
            np.zeros(
                5,
            )
        )
    # otherwise calculate 3C features
    else:
        features = fvf.process_rectilinearity(data_ENZ)

    # Iterate across channels and append features in sequence
    for _data in data_ENZ:
        features += fvf.process_temporal(_data, dtimes, sr, **tkwargs)
        features += fvf.process_spectral(_data, sr, **fkwargs)
        features += fvf.process_cepstral(_data, sr, **ckwargs)

    if asarray:
        features = np.array(features, dtype=np.float32)

    return features


def run_event_from_disk(EVID_dir, out_fstr="{key}_FV.npy", decon_method="PAZ"):
    fv_dict = {}
    # Read in stream
    st = read(os.path.join(EVID_dir, "bulk25tp45.mseed"))

    # If using Poles And Zeros for Instrument Response Correction
    if decon_method == "PAZ":
        # Attach PAZ response information
        pz_st = resp.attach_paz_to_stream(st, os.path.join(EVID_dir, "paz"))
        snc_list = []
        # Use
        for _tr in pz_st:
            _snc = (_tr.stats.station, _tr.stats.network, _tr.stats.channel[:2])
            if _snc not in snc_list:
                snc_list.append(_snc)
                skwargs = dict(zip(["station", "network", "channel"], _snc))
                skwargs["channel"] += "?"
                # Run preprocessing on specific station/instrument
                pp_st = preprocess_pz_pipeline(pz_st.select(**skwargs))
                # Extract features
                fv = process_feature_vector(pp_st, asarray=True)
                # Append to output
                fv_dict.update({f"{_snc[0]:s}_{_snc[2]:s}": fv})

    # If using RESP and water-level stabilized deconvolution
    # for Instrument Response Correction
    elif decon_method == "RESP":
        # Attach RESP response information
        inv = read_inventory(os.path.join(EVID_dir, "station.xml"))
        rr_st = st.attach_response(inv)
        # Use inventory to get station names
        for _n in inv.networks:
            for _s in _n.stations:
                sta = _s.code
                sta_st = rr_st.select(station=sta)
                if len(sta_st) > 0:
                    # Run preprocessing
                    pp_st = preprocess_rr_pipeline(rr_st)

                    # Extract features
                    fv = process_feature_vector(pp_st, asarray=True)
                    # Append to output
                    fv_dict.update({f"{sta:s}_{pp_st[0].stats.channel[:2]:s}": fv})
    # Iterate across feature vectors
    for _k in fv_dict.keys():
        # Compose save file name and path
        out_path = os.path.join(EVID_dir, out_fstr.format(key=_k))
        # Save vector as a Numpy *.npy file
        np.save(out_path, fv_dict[_k])
    # Convert feature vector dictionary into a DataFrame
    df_fv = pd.DataFrame(fv_dict)
    # Return dataframe
    return df_fv


def run_example():
    from time import time

    t0 = time()
    from obspy.clients.fdsn import Client
    from obspy import UTCDateTime

    t1 = time()
    client = Client("IRIS")
    tp = UTCDateTime(2017, 5, 11, 2, 31, 22.48)
    ts = tp - 7.0
    te = tp + 3.0
    st = client.get_waveforms(
        network="UW",
        station="GNW",
        location="*",
        channel="BH?",
        starttime=ts,
        endtime=te,
        attach_response=True,
    )
    t2 = time()
    pp_stream = preprocess_rr_pipeline(st.copy())
    t3 = time()
    fv = process_feature_vector(pp_stream)
    t4 = time()
    print(f"dependency imports took {t1 - t0: .3f} sec")
    print(f"data download took {t2 - t1: .3f} sec")
    print(f"preprocessing took {t3 - t2: .3f} sec")
    print(f"feature extraction took {t4 - t3: .3f} sec")

    return st, pp_stream, fv


# def remove_response_L23(trace, client, inst_type = 'VEL', inplace=False):

#     # Handle decision of inplace changes to trace
#     if inplace:
#         tr_out = trace
#     else:
#         tr_out = trace.copy()
#     # Convert inst_type and output_type into number of zeros to clip
#     # from poles and zeros
#     zc_dict = {'VEL': 2, 'ACC': 0}
#     _zc = zc_dict[inst_type]
#     # Get PAZ file and write as temp
#     _tmp_PAZ = '_tmp.pz'
#     qkw = dict(zip(['network','station','channel','starttime','endtime','level'],
#                    [tr_out.stats.network, tr_out.stats.station, tr_out.stats.channel,
#                     tr_out.stats.starttime, tr_out.stats.endtime, 'response']))
#     _inv = client.get_stations(**qkw)
#     _write_sacpz(_inv, _tmp_PAZ)
#     attach_paz(tr_out, _tmp_PAZ, todisp=False, torad=False)
#     zeros = np.array(tr_out.stats.paz.zeros)
#     zeros = np.delete(zeros, np.argwhere(zeros==0)[0:_zc])
#     poles = np.array(tr_out.stats.paz.poles)
#     constant = tr_out.stats.paz.gain
#     sts2 = {'gain': constant, 'poles': poles, 'sensitivity'}
