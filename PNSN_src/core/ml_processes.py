"""
:module: core.ml_processes
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: CC-BY-4.0

:purpose:
    Using already-loaded `pb_SAM` object(s) (see PNSN_src/drivers/train_test_validate/__run_me_first__load_models.py),
    and pre-extracted features from waveform data, provide a set of methods for training, testing, and validating
    within the E3WS / XGBoost / scikit-learn combined API 

    This set of methods seeks to create an abstraction of the example scripts
    `E3WS/real_time/E3WS_rt.py` (model prediction)
    `E3WS/models/DET/pb_save_DET_model.py` (model training) 
    provided in the E3WS publication repository
"""

import os
import sys
# from sklearn import StandardScaler
# from sklearn import 


# Set path to root directory and append to path
ROOT = os.path.join('..', '..')
sys.path.append(ROOT)

# # Load repository specific codes
# import PNSN_src.util.model_io as io


def run_prediction(pb_SAM_model, feature_vector):
    """
    Run a prediction on a pb_SAM model object for 
    a given feature_vector.

    :: INPUTS ::
    :param pb_SAM_model: [PNSN_src.util.SAM.pb_SAM]
        Ensemble model object trained to estimate a particular
        source property given a feature_vector
    :param feature_vector: [(1, 140) numpy.ndarray]
        140 element feature vector proposed in Lara et al. (2023)
    
    :: OUTPUT ::
    :return predict: [float]
        Predicted value (model specific)
    """
    if feature_vector.shape == (1, 140):
        fv = feature_vector
        predict = pb_SAM_model.predict(fv)[0]

    elif len(feature_vector.shape) == 1:
        if feature_vector.shape[0] == 140:
            fv = feature_vector.reshape(1,140)
            predict = pb_SAM_model.predict(fv)[0]
        elif len(feature_vector.shape) == 2:
            if feature_vector.shape[1] == 1:
                fc = feature_vector.T
                predict = pb_SAM_model.predict(fv)[0]
            elif feature_vector.shape[1] > 1:
                preds = []
                for _i in range(feature_vector.shape[1]):

    
    return predict


def L23_base_model_options():
    out = { 'colsample_bytree': 0.4603,
            'gamma': 0.0468,
            'learning_rate': 0.05,
            'max_depth': 4,
            'min_child_weight': 1.7817,
            'n_estimators': 6000,
            'reg_alpha': 0.4640,
            'reg_lambda': 0.8571,
            'subsample': 0.8,
            'verbosity': 0,
            'nthread ':  -1
          }
    return out

# def compose_xgb_base_model(scaler, xgbtype='regressor', **options):
#     if xgb_type == 'regressor':
#         model_xgb = make_pipeline(scaler, xgb.XGBRegressor(**options))
#     elif xgb_type == 'classifier':
#         model_xgb = make_pipeline(scaler, xgb.XGBClassifier(**options))

#     return model_xgb


# def train_XGB_base_model(xgb_base_model, train_x, train_y):

