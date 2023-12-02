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
# Target Pre-Trained Model Type
TARGET_MODEL = "MAG"
# Training Label Name
LABEL_NAME = "magnitude"
# Define kwargs for train_test_split()
tts_kwargs = {"test_size": 0.2, "random_state": 62323}
# Define kwargs for K-folds cross validation
kfold_kwargs = {"n_splits": 10, "shuffle": False}
# Development Switch for load/don't load models ()
RUN_RETRAINING = True
# Define input model collection path
IN_ROOT = os.path.join(ROOT, "PNSN_Model", "Lara_2023_Preferred", TARGET_MODEL)
# Define output model location
OUT_ROOT = os.path.join(ROOT, "PNSN_model", "PNSN_retrained", TARGET_MODEL)


if not os.path.exists(OUT_ROOT):
    os.makedirs(OUT_ROOT)


### LABEL, FEATURE, AND MODEL LOADING ###
if RUN_RETRAINING:
    # Use model I/O routine to load the set of models within this directory
    try:
        if models:
            print("models already loaded")
    except NameError:
        models = io.load_models(IN_ROOT, fn_glob_str="*.joblib", verbose=True)

# Define relative path to extracted feature vectors
fv_data = os.path.join(ROOT, "PNSN_data", "bulk_event_magnitude_phase_nwf_FV.csv")
# Load feature vectors and labels
df = pd.read_csv(fv_data, parse_dates=["datetime", "arrdatetime"], index_col=[0])
df.index.name = "arid"

# Subset data to desired label and feature vectors
cols_to_use = [LABEL_NAME] + [f"f{x:03d}" for x in range(140)]
df_labeled = df[cols_to_use]

# Run some cleanup on data
keep_ind = []
for _i in range(len(df)):
    _series = df_labeled.iloc[_i, :]
    _vals = _series.values.astype(np.float32)
    if np.sum(np.isfinite(_vals)) == len(_vals):
        keep_ind.append(True)
    else:
        keep_ind.append(False)
# Apply filter to reject any lines that have inf or nan features
df_labeled = df_labeled[keep_ind]

# order labeled feature vectors by magnitude
df_labeled = df_labeled.sort_values(by=LABEL_NAME)
# Extract arrays for featues (X) and labels (y)
X, y = df_labeled[cols_to_use[1:]].values, df_labeled[LABEL_NAME].values

# split data for training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, **tts_kwargs)

# Initialize K-Folds data splitting object
kf = KFold(**kfold_kwargs)
kf.get_n_splits(X=X_train, y=y_train)

# Iterate across folds and run re-training for each individual model
if RUN_RETRAINING:
    TICK = time()
    print(f'Starting re-training on {TARGET_MODEL}')
    model = deepcopy(models[TARGET_MODEL])
    for _I, (ktrain_index, kwithold_index) in tqdm(enumerate(kf.split(X_train, y_train))):
        tick = time()
        print(f'Starting re-training on base_model_ {_I:d}')
        # Get subset base model
        _base_model_ = model.base_models_[0][_I]
        _base_model_[1].verbosity = 1
        # breakpoint()
        # Get Kth fold training data subset
        _kX_train = X_train[ktrain_index, :]
        _ky_train = y_train[ktrain_index]
        # Run training
        _base_model_.fit(_kX_train, _ky_train)#.score(X_test, y_test)
        tock = time()
        print(f're-training took {tock - tick:.3f} sec')



    breakpoint()


# If not running retraining, just use this script to visualize data splitting
if not RUN_RETRAINING:
    # Inspect what the train/test split has done for data distributions
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(ncols=2, nrows=2)
    axs = [fig.add_subplot(gs[x]) for x in range(3)]

    # Plot full, test, and train label distributions
    axs[0] = fig.add_subplot(221)
    axs[0].hist(y, 100, density=True, alpha=0.25, label="Full Set")
    axs[0].hist(y_test, 100, density=True, alpha=0.25, label="Test Set")
    axs[0].hist(y_train, 100, density=True, alpha=0.25, label="Train Set")

    # Plot Kfolds cross validation training label distributions
    for _i, (ktrain_index, kwithold_index) in enumerate(kf.split(X_train, y_train)):
        _kX_train = X_train[ktrain_index, :]
        _kX_witheld = X_train[kwithold_index, :]
        _ky_train = y_train[ktrain_index]
        _ky_witheld = y_train[kwithold_index]
        axs[1].hist(_ky_train, 100, density=False, alpha=0.1, label=f"fold {_i:d}")
        axs[2].hist(_ky_witheld, 100, density=False, alpha=0.1, label=f"fold {_i:d}")

    for _i in range(3):
        axs[_i].legend()
        axs[_i].set_xlabel(LABEL_NAME)
        if _i == 0:
            axs[_i].set_ylabel("Frequency")
        else:
            axs[_i].set_ylabel("Counts")

    axs[0].set_title("Test / Train Split")
    axs[1].set_title("K-fold cross validation training data labels")
    axs[2].set_title("K-fold cross validation witheld data labels")


    plt.show()
