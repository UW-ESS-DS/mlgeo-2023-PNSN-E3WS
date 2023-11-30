"""
:module: PNSN_src.drivers.preprocess.characterize_split_data
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: CC-BY-4.0

:purpose:
    This script conducts analysis on event-station record pairs
    (entries in event_mag_phase*.csv files) that have waveforms
    locally stored from the execution of scripts in the drivers/
    build_waveform_dataset/ subdirectory. 

    Analysis steps / questions
    0) what did we get out of the data query?
    1) how are data labels distributed?
    2) what is a reasonable representation of the data labels? 
        Normal distribution? 
        Skew?
        Kurtosis?
        Quantiles?
    3) What metrics can we use to produce split train/test/validate data
    that preserves label variability within each subset?

"""


import os
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT = os.path.join("..", "..", "..")
# glob.glob string for getting each event's
IGLOB = os.path.join(ROOT, "PNSN_data", "EVID*", "*.csv")
# Intermediate output concatenated file
OCAT = os.path.join(ROOT, "PNSN_metadata", "event_mag_phase_with_waveforms.csv")

# This loop takes awhile, it's easier to do a quick work-around in shell
# 1) `cat` all the event_mag_phase_nwf.csv files in each EVID* folder
#   $ cat EVID*/event_mag_phase_nwf.csv > ../PNSN_metadata/event_mag_phase_with_waveforms.csv
# 2) use `vi` to edit the output file
#   $ vi ../PNSN_metadata/event_mag_phase_with_waveforms
# 2.a) erase the 'e' in 'evid' for the first header line
# 2.b) use :%g/evid/d to remove all lines containing 'evid'
# 2.c) put the 'e' back in 'evid on the first (now only!) header line
# 2.d) save with :qw
# 3) that's it - ready in a few seconds!
if not os.path.exists(OCAT):
    df = pd.DataFrame()
    flist = glob(IGLOB)
    flist.sort()
    for _f in tqdm(flist):
        _df = pd.read_csv(_f)
        _df.index = _df.arid
        df = pd.concat([df, _df], axis=1, ignore_index=False)

# If the output file already exists, just load that
else:
    df = pd.read_csv(OCAT)
    df.index = df.arid

pf_nphases = len(df)
pf_nevents = len(df.evid.unique())

# Filter data to remove fixed epicenter/depth event solutions
# and labels without waveform data
df = df[(df.fdepth == "n") & (df.fepi == "n") & (df.nchan_wf > 0)]

# Show how many events and phases are left
print(
    f"Number of phases after filtering:\
      {len(df):6d}/{pf_nphases:6d}\
      ({100*len(df)/pf_nphases:.1f}%)"
)
print(
    f"Number of events after filtering:\
      {len(df.evid.unique()):6d}/{pf_nevents:6d}\
      ({100*len(df.evid.unique())/pf_nevents:.1f}%)"
)

# Create feature lines for cos(seaz) and sin(seaz)
series_cos = df.seaz.copy()
series_cos = series_cos.apply(lambda x:np.cos((np.pi/180.)*x))
series_cos.name = 'cos_seaz'
series_sin = df.seaz.copy()
series_sin = series_sin.apply(lambda x:np.sin((np.pi/180.)*x))
series_sin.name = 'sin_seaz'

df = pd.concat([df, series_cos, series_sin],axis=1, ignore_index=False)

# Get statistics initial distributions and cross-plots of labels
labels = ["delta", "magnitude", "depth", "cos_seaz",'sin_seaz']

# Get mean values
mean = df[labels].mean()
mean.name = "mean"
print(f"\n == Means == \n{mean}")
# Get covariance matrix
cov = df[labels].cov()
std = pd.Series(np.diag(cov.values) ** 0.5, index=labels, name = 'std')
# Get skewness of each population
skew = df[labels].skew()
skew.name = "skew"
# Get kurtosis of each
kurt = df[labels].kurtosis()
kurt.name = "kurtosis"

# Get quantiles (equivalent of mu +/- 1*sigma, 
#                1st, 2nd (median), and 3rd quartiles)
quants = df[labels].quantile([0.16, 0.25, 0.5, 0.75, 0.84])
# pivot
quants = quants.T
for _c in quants.columns:
    quants = quants.rename(columns={_c: f"q{_c:.2f}"})


# Stitch together non Cov statistics
stats = pd.concat([mean, std, skew, kurt, quants], axis=1, ignore_index=False)
print(f"\n == Individual Stats == \n{stats.T}")

print(f"\n == Cov Matrix == \n{cov}")


# for _i, _label in labels:


# # create holder
# z_scores = np.zeros(shape=(len(flds), len(df)))
# means = df[flds].mean()
# cov = df[flds].cov()
# zcols = ["z_" + _fld for _fld in flds]
# # Iterate across fields
# for _i, _fld in enumerate(flds):
#     # Calculate Z-scores for each field
#     _v = df[_fld].values
#     mu_v = np.nanmean(_v)
#     var_v = np.nanstd(_v) ** 2
#     z_v = (_v - mu_v) / var_v
#     z_scores[_i, :] = z_v

# df = pd.concat([df, pd.DataFrame(z_scores.T,columns=zcols, index=df.arid)],axis=1, ignore_index=False)

# fig = plt.figure(figsize=(7,7))
# gs = fig.add_gridspec(ncols=2, nrows=2)
# axs = [fig.add_subplot(gs[x]) for x in range(4)]
# for _i, _fld in enumerate(flds):
#     axs[_i].hist(z_scores[_i,:],100)
#     axs[_i].set_xlabel(zcols[_i])

# axs[3].hist(np.sum(z_scores, axis=0)/3, 100)
# axs[3].set_xlabel('average z-score')


# fig = plt.figure(figsize=(7,7))
# gs = fig.add_gridspec(ncols=3, nrows=3)
# for _i, _ifld in enumerate(flds):
#     for _j, _jfld in enumerate(flds):
#         if _i <= _j:
#             ax = fig.add_subplot(gs[_i, _j])
#         if _i == _j:
#             ax.hist(df[_ifld],100)
#         elif _i < _j:
#             ax.scatter(df[_ifld], df[_jfld], alpha=0.01, s=1)


plt.show()
