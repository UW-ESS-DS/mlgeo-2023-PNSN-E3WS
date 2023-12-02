import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

# Define relative path to root directory
ROOT = os.path.join("..", "..", "..")
# Append to path
sys.path.append(ROOT)
# Import model I/O utilities
import PNSN_src.util.model_io as io

# Import the particular module/path definition for the pb_SAM class that is used to read/write models
from PNSN_src.util.SAM import pb_SAM


OUT_ROOT = os.path.join(ROOT, "PNSN_processed_data")

if not os.path.exists(OUT_ROOT):
    os.makedirs(OUT_ROOT)

### LABEL, FEATURE, AND MODEL LOADING ###

# Define relative path to model directory
model_dir = os.path.join(ROOT, "PNSN_Model", "Lara_2023_Preferred")
# Use model I/O routine to load the set of models within this directory

try:
    if models:
        print("models already loaded")
except NameError:
    models = io.load_models(model_dir, fn_glob_str="*/*.joblib", verbose=True)

# Define relative path to extracted feature vectors
fv_data = os.path.join(ROOT, "PNSN_data", "bulk_event_magnitude_phase_nwf_FV.csv")
# Load feature vectors and labels
df = pd.read_csv(fv_data, parse_dates=["datetime", "arrdatetime"], index_col=[0])
df.index.name = "arid"


### PROCESSING SECTION ###

# Create holders for LASSO and individual model predictions
LASSO_pred_dict = {}
segment_pred_dict = {}
# Iterate across keys for model type
for _k in models.keys():  # for Benz / Jake change to: "for _k in ['MAG']:"
    # Get specific ensemble model from model dictionary
    _model = models[_k]
    # Create capture line for LASSO prediction outputs for each arrival
    LASSO_pred_line = []
    # Create capture "array" for lines of individual estimates for each arrival
    segment_pred_array = []
    # Report to command line what's running
    print(f"Running {_k}")
    # Iterate across each arrival in the dataframe
    for _i in tqdm(range(len(df))):  # len(df))):
        # Create a holder for individual estimates for each arrival
        segment_pred_line = []
        # pull arrival labels and features as a series from dataframe
        _series = df.iloc[_i, :]
        # Extract feature vector from series
        FV = _series.values[-140:].reshape(1, 140).astype(np.float32)
        # Sanity check that all features are finite
        if np.sum(np.isfinite(FV)) == 140:
            # RUN ENSEMBLE PREDICTION
            pred = _model.predict(FV)[0]
            # Iterate across inidividual base models
            for _i in range(len(_model.base_models_[0])):
                # Extract ensemble member
                _imod = _model.base_models_[0][_i]
                # Run prediction for ensemble member
                ipred = _imod.predict(FV)[0]
                # Append individual prediction to line
                segment_pred_line.append(ipred)
        # If the feature vector contains nan or inf, return nan for ensemble and member predictions
        else:
            pred = np.nan
            segment_pred_line = [np.nan] * len(_model.base_models_[0])
        # Capture predictions at the arrival level
        LASSO_pred_line.append(pred)
        segment_pred_array.append(segment_pred_line)
    # Write out individual model outputs following each model completion
    try:
        array = np.array(segment_pred_array)
    except:
        breakpoint()
    idf = pd.DataFrame(
        array,
        columns=[f"{_k}{x:02d}" for x in range(array.shape[1])],
        index=df.index.values[: array.shape[0]],
    )
    idf.index.name = "arid"
    idf.to_csv(
        os.path.join(
            OUT_ROOT, f"event_mag_phase_FV_pretrained_individual_{_k}_predictions.csv"
        ),
        header=True,
        index=True,
    )
    # Capture predictions at the model level
    LASSO_pred_dict.update({_k: LASSO_pred_line})
    segment_pred_dict.update({_k: np.array(segment_pred_array)})


### MODELED VALUE SAVING SECTION ###

# Package predictions as a CSV-friendly format
df_pred = pd.DataFrame(LASSO_pred_dict, index=df.index)
# Concatenate data labels, features, and predictions into one dataframe
df_out = pd.concat([df, df_pred], axis=1, ignore_index=True)
df_out.columns = list(df.columns.values) + list(df_pred.columns.values)
# Write complete set to disk
df_out.to_csv(
    os.path.join(OUT_ROOT, "event_mag_phase_FV_pretrained_ENSEMBLE_predictions.csv"),
    header=True,
    index=False,
)

for _k in segment_pred_dict.keys():
    array = segment_pred_dict[_k]
    idf = pd.DataFrame(
        array,
        columns=[f"{_k}{x:02d}" for x in range(array.shape[1])],
        index=df.index.values[: array.shape[0]],
    )
    idf.index.name = "arid"
    idf.to_csv(
        os.path.join(
            OUT_ROOT, f"event_mag_phase_FV_pretrained_individual_{_k}_predictions.csv"
        ),
        header=True,
        index=True,
    )
