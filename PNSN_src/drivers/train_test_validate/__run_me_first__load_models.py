"""
:module: PNSN_src.drivers.train_test_validate.__run_me_first__load_models
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: CC-BY-4.0

:purpose:
    This script provides a minimal example of loading E3WS models from disk
    into an python instance. In the case of an interactive job (i.e., ipython)
    this script should be run from the PNSN_src/drivers/train_test_validate
    directory to load models.

    For scripts run via the command-line, this code block can be used as a
    template for a "model loading" section of the script.
"""
# Run minimal imports for path definitions & updates
import os
import sys
# Define relative path to root directory
ROOT = os.path.join("..", "..", "..")
# Append to path
sys.path.append(ROOT)
# Import model I/O utilities
import PNSN_src.util.model_io as io
# Import the particular module/path definition for the pb_SAM class that is used to read/write models
from PNSN_src.util.SAM import pb_SAM

# Define relative path to model directory
model_dir = os.path.join(ROOT, 'PNSN_Model', 'Lara_2023_Preferred')
# Use model I/O routine to load the set of models within this directory
models = io.load_models(model_dir, fn_glob_str='*/*.joblib', verbose=True)
