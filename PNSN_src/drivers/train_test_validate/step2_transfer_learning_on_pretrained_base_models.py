"""
:module: PNSN_src/drivers/train_test_validate/step2_transfer_learning_on_pretrained_base_models.py
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: CC-BY-4.0

:purpose:
    This script seeks to emulate the training approach described in Lara et al. (2023) to re-train
    an ensemble of random forests (each called "base_model"s) with extracted feature data from 
    the Pacific Northwest Seismic Network (PNSN) back-catalog. In particular, this PNSN dataset
    has significanly smaller earthquakes compared to the training datasets used in Lara et al. (2023).


:references:
    Pablo Lara, Quentin Bletery, Jean-Paul Ampuero, Adolfo Inza, Hernando Tavera.
        Earthquake Early Warning Starting From 3 s of Records on a Single Station
        With Machine Learning. Journal of Geophysical Research: Solid Earth.
        https://doi.org/10.1029/2023JB026575

:attribution:
    This code is based on scripts authored by, and discussion with, Pablo Lara.

    

:TODO: 
    Provide option for re-loading saved split data state
"""

import os
import sys
import pandas as pd
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from time import time
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt

# Define relative path to root directory
ROOT = os.path.join("..", "..", "..")
# Append to path
sys.path.append(ROOT)
# Import model I/O utilities
import PNSN_src.util.model_io as io

# Import the particular module/path definition for the pb_SAM class for model I/O
from PNSN_src.util.SAM import pb_SAM


### PARAMETER CONTROL BLOCK ###
# I/O Controls
LOAD_DATA_SPLIT = False
SAVE_BASE_MODELS = True
# Define input model collection path
IN_ROOT = os.path.join(ROOT, "PNSN_Model", "Lara_2023_Preferred")
# Target Pre-Trained Model Types
TARGET_MODELS = ["MAG"]#, "DIST"]
# Training Label Names
LABEL_NAMES = ["magnitude"]#, "delta"]
# Define output model location
OUT_ROOT = os.path.join(ROOT, "PNSN_model", "PNSN_retrained")


# Define kwargs for train_test_split()
ttsplit = 1/3
tts_kwargs = {"test_size": ttsplit, "random_state": 62323}
# Define kwargs for K-folds cross validation
kfold_kwargs = {"n_splits": 10, "shuffle": False}

### LABELD FEATURE VECTOR LOADING ###
# Define relative path to extracted feature vectors
fv_data = os.path.join(ROOT, "PNSN_data", "bulk_event_magnitude_phase_nwf_FV.csv")
# Load feature vectors and labels
df = pd.read_csv(fv_data, parse_dates=["datetime", "arrdatetime"], index_col=[0])
df.index.name = "arid"

# NOTE: This needs to be maually edited depending on the data labels above
df = df[(df.magnitude.notna()) & (df.magnitude.notna() > -1) & (df.delta.notna())]


if not os.path.exists(OUT_ROOT):
    os.makedirs(OUT_ROOT)

# Run some cleanup on feature data
keep_ind = []
for _i in range(len(df)):
    _series = df.iloc[_i, -140:]
    _vals = _series.values.astype(np.float32)
    if np.sum(np.isfinite(_vals)) == len(_vals):
        keep_ind.append(True)
    else:
        keep_ind.append(False)

# Subset data to desired label and feature vectors
fv_cols = [f"f{x:03d}" for x in range(140)]
# Add arid into label names for indexing purposes
LABEL_NAMES += ["arid"]  # + LABEL_NAMES
df_labeled = df[LABEL_NAMES + fv_cols]

# Apply filter to reject any lines that have inf or nan features
df_labeled = df_labeled[keep_ind]

# Extract arrays for featues (X) and labels (y)
X, y = df_labeled[fv_cols].values, df_labeled[LABEL_NAMES].values

# Create path and file name for data splitting indices
indices_out = os.path.join(OUT_ROOT, "indices")
indices_out_file = os.path.join(indices_out, f'test_train_{ttsplit:.2f}_ARID_assignments.csv')
if not os.path.exists(indices_out):
    print(f"making directory {indices_out}")
    os.makedirs(indices_out)


## DATA SPLITTING SECTION ##
if LOAD_DATA_SPLIT:
    df_indices = pd.read_csv(indices_out_file, index_col=[0])
else:
    # split data for training & testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, **tts_kwargs)

    # Create series to hold ARID - group "split" assignment pair
    _series_train = pd.Series(
        [True] * y_train.shape[0], index=y_train[:, -1].astype(np.int32), name="train"
    )
    _series_test = pd.Series(
        [False] * y_test.shape[0], index=y_test[:, -1].astype(np.int32), name="train"
    )
    df_indices = pd.DataFrame(
        pd.concat([_series_train, _series_test], axis=0, ignore_index=False)
    )
    df_indices.index.name = "arid"

    # Initialize K-Folds data splitting object
    # Rationale: we want to have the same station-event pairs in each labeled
    kf = KFold(**kfold_kwargs)
    kf_generator = kf.split(X_train, y_train)
    # Pre-generate kf splits to ensure stability across models
    kf_indices = [(kinclude, komit) for (kinclude, komit) in kf_generator]

    # Create boolean mask for each k-folds subsampling
    for _I, (_kinclude, _komit) in enumerate(kf_indices):
        _kf_series = pd.Series(
            [True] * len(_kinclude) + [False] * len(_komit) + [False] * len(y_test),
            index=np.r_[y_train[_kinclude, -1], y_train[_komit, -1], y_test[:, -1]],
            name=f"kfi{_I:02d}",
        )
        df_indices = pd.concat([df_indices, _kf_series], axis=1, ignore_index=False)

    print("saving k-folds data splitting to disk")
    df_indices.to_csv(indices_out_file, header=True, index=True)


score_ind = []
# Iterate across folds and run re-training for each individual model
for _I, _TM in enumerate(TARGET_MODELS):
    TICK = time()
    # Load pretrained model
    model_PT = io._load_joblib_persistence(
        io.glob(os.path.join(IN_ROOT, _TM, f"*{_TM}*.joblib"))[0]
    )
    print(f"Starting re-training on {_TM}")
    # Use BOOL training indices to extract test features and labels
    X_test = X[~df_indices.train.values, :]
    y_test = y[~df_indices.train.values, _I]
    # Make a deepcopy of the pretrained model for performance comparison
    # NOTE: Put a pin in this idea - move it to `step3_*.py`
    model_RT = deepcopy(model_PT)

    # Iterate across splits
    for _J in range(10):
        _kinclude = df_indices[f"kfi{_J:02d}"].values
        tick = time()
        print(f"Starting re-training on {_TM} base_model_ {_J:d}")
        # Get subset base model
        _base_model_ = model_RT.base_models_[0][_J]
        # Get Kth fold training data subset and specific labels for model
        _kX_train = X[_kinclude, :]
        _ky_train = y[_kinclude, _I]
        # Run training
        score = _base_model_.fit(_kX_train, _ky_train).score(X_test, y_test)
        score_ind.append([_I, _J, score])
        tock = time()
        print(f"re-training took {tock - tick:.3f} sec")
        print(f"score is {score}")

        ## Progressively write retrained base_model_'s to disk
        # Create path(s)
        if SAVE_BASE_MODELS:
            base_iter_dir = os.path.join(OUT_ROOT, _TM, "retrained_base_models")
            if not os.path.exists(base_iter_dir):
                os.makedirs(base_iter_dir)
            base_iter_out = os.path.join(base_iter_dir, f"retrained_{_TM}_base_model_{_J:01d}_tf{1 - ttsplit:.2f}.joblib")
            print(f"writing {base_iter_out}")
            io.dump_model(_base_model_, base_iter_out)

    ## Dump retrained ensemble model persistence to disk
    ensemble_out = os.path.join(OUT_ROOT, _TM, f"retrained_{_TM}_ensemble_model_tf{1 - ttsplit:.2f}.joblib")
    if not os.path.exists(os.path.split(ensemble_out)[0]):
        os.makedirs(os.path.split(ensemble_out)[0])
    print(f"writing {ensemble_out}")
    io.dump_model(model_RT, ensemble_out)

    breakpoint()


## GRAVEYARD - SHIFT THIS CHUNK OF PLOTTING CODE TO A `util` or `visualize`

# # If not running retraining, just use this script to visualize data splitting
# if not RUN_RETRAINING:
#     # Inspect what the train/test split has done for data distributions
#     fig = plt.figure(figsize=(10, 10))
#     gs = fig.add_gridspec(ncols=2, nrows=2)
#     axs = [fig.add_subplot(gs[x]) for x in range(3)]

#     # Plot full, test, and train label distributions
#     axs[0] = fig.add_subplot(221)
#     axs[0].hist(y, 100, density=True, alpha=0.25, label="Full Set")
#     axs[0].hist(y_test, 100, density=True, alpha=0.25, label="Test Set")
#     axs[0].hist(y_train, 100, density=True, alpha=0.25, label="Train Set")

#     # Plot Kfolds cross validation training label distributions
#     for _i, (ktrain_index, kwithold_index) in enumerate(kf.split(X_train, y_train)):
#         _kX_train = X_train[ktrain_index, :]
#         _kX_witheld = X_train[kwithold_index, :]
#         _ky_train = y_train[ktrain_index]
#         _ky_witheld = y_train[kwithold_index]
#         axs[1].hist(_ky_train, 100, density=False, alpha=0.1, label=f"fold {_i:d}")
#         axs[2].hist(_ky_witheld, 100, density=False, alpha=0.1, label=f"fold {_i:d}")

#     for _i in range(3):
#         axs[_i].legend()
#         axs[_i].set_xlabel(LABEL_NAME)
#         if _i == 0:
#             axs[_i].set_ylabel("Frequency")
#         else:
#             axs[_i].set_ylabel("Counts")

#     axs[0].set_title("Test / Train Split")
#     axs[1].set_title("K-fold cross validation training data labels")
#     axs[2].set_title("K-fold cross validation witheld data labels")

#     plt.show()
