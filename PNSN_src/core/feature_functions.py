"""
:creator: Pablo E. Lara
:attribution: This is a copy of the real_time/functions/pb_functions.py
             created by Pablo Lara as part of their publication Lara et al. (2023)
             I have added some annotations and a series of utility methods 
             at the end of the module that merge code segments from pb_utils_v16.py
             from the same submodule in Lara's E3WS repository.

             If this module is used, please reference Pablo Lara's publication and their GitHub
             repository (see our README.md in the root directory of the repository)
:editor: Nathan T. Stevens
:ed. email: ntsteven (at) uw.edu
:ed. org: Pacific Northwest Seismic Network
:license: CC-BY-4.0, inherited from the license for Pablo Lara's GitHub repository for E3WS
"""

# E3WS attributes
# Some of these features are in:
# https://doi.org/10.1109/JSTARS.2020.2982714,
# https://doi.org/10.1109/MSP.2017.2779166

import numpy as np
import scipy as sp
from python_speech_features import mfcc

# from scipy.stats import threshold


# Threshold function from scipy 0.15.1
def threshold(a, threshmin=None, threshmax=None, newval=0):
    a = np.asarray(a).copy()
    mask = np.zeros(a.shape, dtype=bool)
    if threshmin is not None:
        mask |= a < threshmin
    if threshmax is not None:
        mask |= a > threshmax
    a[mask] = newval
    return a


# Feat 1, length of temporal signal
def lenXt(Xt):
    T = len(Xt)
    return T


# Energy of signal
def Pt(Xt):
    Et = Xt**2
    return Et


# Feat 2, Max of temporal energy
def maxEt(Xt):
    Et = Pt(Xt)
    M = np.max(Et)
    return M


# Erms
def Erms(data):
    E = np.sum(data)
    E_rms = np.sqrt((1 / len(data) * E))
    return E_rms


# Feat 3, Mean of temporal energy
# def meanEt(Xt):
# 	Et = Pt(Xt)
# 	MeanE = np.mean(Et)
# 	return MeanE


# Feat 4, Index of max of temporal energy
def argmaxEt(Xt, t):
    Et = Pt(Xt)
    Lt = t[np.argmax(Et)]
    return Lt


# Feat 5, Index of max of PSD
def argmaxEf(PSD, f):
    Lf = f[np.argmax(PSD)]
    return Lf


# Feat 6, Centroid of frequency
def centroid_f(PSD, f):
    Ef = np.sum(PSD)
    centroid = np.sum(f * PSD) / Ef
    return centroid


# Feat 7, Spectrum Spread or Bandwidth (measure of variance around centroid)
def BW_f(PSD, f):
    Ef = np.sum(PSD)
    centroid = centroid_f(PSD, f)
    BWf = np.sqrt(np.sum((f - centroid) ** 2 * PSD) / Ef)
    # BWf = np.sqrt( np.sum((f**2) * (PSD) / Ef) - centroid**2 )
    return BWf


# Feat 8, Frequency skewness, Measure of skewness around bandwidth (This could be negative)
def skewness_f(PSD, f):
    Ef = np.sum(PSD)
    centroid = centroid_f(PSD, f)
    BW = BW_f(PSD, f)
    skewness = np.sum((f - centroid) ** 3 * PSD) / (Ef * BW**3)
    if skewness >= 0:
        return np.sqrt(skewness)
    else:
        return -np.sqrt(-skewness)


# Feat 9, Frequency kurtosis, Measure of kurtosis around BW
def kurtosis_f(PSD, f):
    Ef = np.sum(PSD)
    centroid = centroid_f(PSD, f)
    BW = BW_f(PSD, f)
    kurtosis = np.sqrt(np.sum((f - centroid) ** 4 * PSD) / (Ef * BW**4))
    return kurtosis


# Feat 10, Centroid of time weighted by power
def centroid_t(Xt, t):
    Ene_t = Pt(Xt)
    Et = np.sum(Ene_t)
    centroid = np.sum(t * Ene_t) / Et
    return centroid


# Feat 11, Temporal bandwidth
def BW_t(Xt, t):
    Ene_t = Pt(Xt)
    Et = np.sum(Ene_t)
    centroid = centroid_t(Xt, t)
    BWt = np.sqrt(np.sum((t - centroid) ** 2 * Ene_t) / Et)
    return BWt


# Feat 12, Temporal skewness
def skewness_t(Xt, t):
    Ene_t = Pt(Xt)
    Et = np.sum(Ene_t)
    centroid = centroid_t(Xt, t)
    BW = BW_t(Xt, t)
    skewness = np.sum((t - centroid) ** 3 * Ene_t) / (Et * BW**3)
    if skewness >= 0:
        return np.sqrt(skewness)
    else:
        return -np.sqrt(-skewness)


# Feat 13, Temporal kurtosis
def kurtosis_t(Xt, t):
    Ene_t = Pt(Xt)
    Et = np.sum(Ene_t)
    centroid = centroid_t(Xt, t)
    BW = BW_t(Xt, t)
    kurtosis = np.sqrt(np.sum((t - centroid) ** 4 * Ene_t) / (Et * BW**4))
    return kurtosis


# 16, Rate of decay in time
def ROD_t(Xt):
    Ene_t = Pt(Xt)
    delta = Ene_t[1:] - Ene_t[:-1]
    M = np.max(Ene_t)
    rod_t = np.min(delta / M)
    return rod_t


# 17, Rate of decay in frequency
def ROD_f(PSD):
    delta = PSD[1:] - PSD[:-1]
    M = np.max(PSD)
    rod_f = np.min(delta / M)
    return rod_f


# 18, Ratio of Maximum amplitude envelope to the mean envelope
def RMM(Xn):
    rmm = np.max(Xn) / np.mean(Xn)
    return rmm


# 19, scipy skewness envelope
# 20, scipy kurtosis envelope


# 21, Duration increase respect duration decrease envelope
def IncDec_env(Xenv, env):
    tmax = env[np.argmax(Xenv)]
    ti = env[0]
    tf = env[-1]
    if tf - tmax != 0:
        inc_dec = (tmax - ti) / (tf - tmax)
    else:
        inc_dec = 0
    return inc_dec


# 22, Duration increase respect total duration envelope
def Growth_env(Xenv, env):
    tmax = env[np.argmax(Xenv)]
    ti = env[0]
    tf = env[-1]
    if tf - ti != 0:
        growth = (tmax - ti) / (tf - ti)
    else:
        growth = 0
    return growth


# 23, How many times the signal exceeds 0, Zero Crossing rate (for example 10.5 times/second)
def ZCR_t(Xn, Fs):
    duration = len(Xn) / Fs
    zcr_t = (((Xn[:-1] * Xn[1:]) < 0).sum()) * 1 / duration
    return zcr_t


# 24, Standar deviation envelope


# Ratio of how many times the signal exceeds threshold, for example (0.7 times/second)
def TCR_t(Xn, thres, Fs):
    duration = len(Xn) / Fs
    Xn = Xn / np.max(np.abs(Xn))
    Xn = Xn - thres
    tcr_t = (((Xn[:-1] * Xn[1:]) < 0).sum()) * 1 / duration
    return tcr_t


# Ratio of how many points not exceeds the threshold
def mTCR_t(Xn, thres):
    Xn = Xn / np.max(np.abs(Xn))
    TH = threshold(Xn, threshmin=thres, threshmax=1, newval=-127)
    mtcr_t = np.where(TH != -127)[
        0
    ]  # different of -127 are the elements between thresmin and thresmax
    return len(mtcr_t) / len(Xn)


def shannon_ent(Xn, Bins):
    prob, bins = np.histogram(Xn, bins=Bins)
    prob = prob / len(Xn)
    prob = prob[np.nonzero(prob)]
    shannon_entropy = np.sum(-prob * np.log2(prob))
    return shannon_entropy


def renyi_ent(Xn, alpha, Bins):
    prob, bins = np.histogram(Xn, bins=Bins)
    prob = prob / len(Xn)
    prob = prob[np.nonzero(prob)]
    renyi_entropy = np.log2(np.sum(prob**alpha)) / (1 - alpha)
    return renyi_entropy


# How many times the signal exceeds threshold, for example (5times in 1 to 20Hz)
def TCR_f(PSD, thres):
    PSD = PSD / np.max(np.abs(PSD))
    PSD = PSD - thres
    tcr_f = ((PSD[:-1] * PSD[1:]) < 0).sum()
    return tcr_f


# Turning points
def group_in_threes(slicable):
    for i in range(len(slicable) - 2):
        yield slicable[i : i + 3]


def turns(L):
    for index, three in enumerate(group_in_threes(L)):
        if (three[0] > three[1] < three[2]) or (three[0] < three[1] > three[2]):
            yield index + 1


def diff_turn_points(data):
    turn_points_index = list(turns(data))
    E_turn_points = (data)[turn_points_index]
    diff_E_turn_points = np.diff(E_turn_points)
    if (
        len(diff_E_turn_points) == 0
    ):  # exception when IMF is generate by sine signal -> PSD has not turning points
        diff_E_turn_points = 0
    return diff_E_turn_points


# Wrapper methdos 

def process_rectilinearity(data_ENZ):
    """
    Conduct eigen-value/-vector and rectilinearity calculations
    on 3-C data

    :: INPUT ::
    :param data_ENZ: [(3, m) numpy.ndarray]
            Ordered data array from East, North, (Z)vertical channels
            with uniform sampling rate and sample size "m"
    
    :: OUTPUT ::
    :return features: [list]
            list of rectilinearity features in order specified
            from Lara et al. (2023) 
                0 ] Maximum eigenvalue
                1 ] rectilinearity
                2 ] Unit maximum eigenvector, east basis element
                3 ] Unit maximum eigenvector, north basis element
                4 ] Unit maximum eigenvector, vertical basis element
    """
    # Calculate covariance matrix
    cov = np.cov(data_ENZ)
    # Calculate eigenvalues (w) and normalized eigenvectors (v)
    w, v = np.linalg.eig(cov)
    # Get largest eigenvalue
    ws = np.sort(w)
    features = [ws[-1],
                ws[-1]/(ws[-2] + ws[-3])]
    features += list(v[:, np.argmax(w)])
    return features


def process_temporal(data, dtimes, sr, thresh=0.8, bins=200, alpha=2):
    """
    Wrap temporal feature extraction for a single trace.

    :: INPUTS ::
    :param data: [numpy.ndarray]
            preprocessed trace object data
    :param dtimes: [numpy.ndarray]
            relative time of data in seconds
    :param sr: [int]
            sampling rate in Hz
    :param thresh: [float]
            threshold value for envelope features
    :param bins: [int]
            number of bins for entropy calculations
    :param alpha: [int]
            alpha value for Renyi entropy

    :: OUTPUT ::
    :return features: [list]
            list of temporal features in order specified
            from Lara et al. (2023)
            # Energy features
            0 ] max energy value
            1 ] time of maximum energy value
            2 ] time-weighted-average amplitude
            3 ] amplitude variance
            4 ] amplitude skewness
            5 ] amplitude kurtosis
            6 ] total energy
            # Envelope Features
            7 ] Envelope threshold crossing rate
            8 ] Max/Mean Envelope ratio
            9 ] Envelope mean
            10] Envelope standard deviation
            11] Envelope skewness
            12] Envelope kurtosis (Pearson)
            13] Envelope threshold crossing rate complement
            14] Shannon entropy
            15] Renyi entropy
            # Original timeseries feature
            16] Zero Crossing Rate

    Note: N. Stevens added this method, combining elements of
    E3WS_rt.py and pb_utils_v16.py
    """
    if data.shape != dtimes.shape:
        breakpoint()
    # Get analytic signal
    hilbert = sp.signal.hilbert(data)
    # Get envelope from analytic signal
    envelope = np.abs(hilbert)

    # Calculate/compose feature list
    features = [
        maxEt(data),
        argmaxEt(data, dtimes),
        centroid_t(data, dtimes),
        BW_t(data, dtimes),
        skewness_t(data, dtimes),
        kurtosis_t(data, dtimes),
        np.sum(data**2),
        TCR_t(envelope, thresh, sr),
        RMM(envelope),
        np.mean(envelope),
        np.std(envelope),
        sp.stats.skew(envelope),
        sp.stats.kurtosis(envelope, fisher=False),
        mTCR_t(envelope, thresh),
        shannon_ent(envelope, bins),
        renyi_ent(envelope, alpha, bins),
        ZCR_t(data, sr),
    ]
    return features


def process_spectral(data, sr, N_fft=1024, pct_overlap=75, bins=50, alpha=2, thresh=0.4):
    """
    Wrap spectral feature extraction for a single trace.

    :: INPUTS ::
    :param data: [numpy.ndarray]
            preprocessed trace object data
    :param sr: [int]
            sampling rate in Hz
    :param N_fft: [float]
            maximum number of fourier frequencies per segment
            for Welch's method for PSD's
    :param pct_overlap: [float]
            percent overlap for Welch's method
            (see scipy.signal.welch())
    :param thresh: [float]
            threshold value for envelope features
    :param bins: [int]
            number of bins for entropy calculations
    :param alpha: [int]
            alpha value for Renyi entropy

    :: OUTPUT ::
    :return features: [list]
            list of spectral features in order specified
            from Lara et al. (2023)

            # PSD Features
            0 ] Maximum spectral amplitude
            1 ] Frequency of max spectral ampiltude
            2 ] Spectrum centroid (f-weighted average spectral amplitude)
            3 ] Spectrum bandwidth (variance about spectrum centroid)
            4 ] PSD skewness
            5 ] PSD kurtosis
            6 ] Mean PSD amplitude
            7 ] Std PSD amplitude
            8 ] PSD amplitude skewness
            9 ] PSD amplitdue kurtosis
            10] PSD Shannon entropy (uses bins)
            11] PSD Renyi entropy (uses bins)
            12] PSD Max/Mean ratio
            13] PSD threshold crossing rate (uses thresh)
            14] PSD threshold crossing rate complement (uses thresh)

    Note: N. Stevens added this method, combining elements of
    E3WS_rt.py and pb_utils_v16.py
    """
    # Ensure N_fft does not exceed data length
    if len(data) < N_fft:
        N_fft = len(data)

    N_overlap = N_fft * pct_overlap // 100

    # Calculate Welch's Power Spectral Density (PSD)
    ffreq, spec = sp.signal.welch(
        data, sr, window="hanning", detrend=False, nperseg=N_fft, noverlap=N_overlap
    )

    # Calculate / compose feature list
    features = [
        np.max(spec),
        argmaxEf(spec, ffreq),
        centroid_f(spec, ffreq),
        BW_f(spec, ffreq),
        skewness_f(spec, ffreq),
        kurtosis_f(spec, ffreq),
        np.mean(spec),
        np.std(spec),
        sp.stats.skew(spec),
        sp.stats.kurtosis(spec),
        shannon_ent(spec, bins),
        renyi_ent(spec, alpha, bins),
        RMM(spec),
        TCR_f(spec, thresh),
        mTCR_t(spec, thresh),
    ]

    return features


def process_cepstral(data, sr, numcep=13, nfilt=26, freqmin=1.):
    """
    Wrap cepstral feature extraction for a single trace.

    :: INPUTS ::
    :param trace: [obspy.core.trace.Trace]
            preprocessed trace object
    
    :: OUTPUT ::
    :param features: [list]
            list of 13 cepstrum
    
    Note: N. Stevens added this method, combining elements of
    E3WS_rt.py and pb_utils_v16.py
    """
    npts = len(data)
    # MFCC analysis for a single window
    mfcc_features = mfcc(data, samplerate=sr, winlen=npts/sr, winstep=npts/sr,
                         nfft=npts, numcep=numcep, nfilt=nfilt,
                         lowfreq=freqmin, highfreq=sr/2, appendEnergy=False)
    mfcc_features = mfcc_features.reshape(-1)

    # Pass all 13 cepstrum to list
    features = [_cep for _cep in mfcc_features]
    return features


