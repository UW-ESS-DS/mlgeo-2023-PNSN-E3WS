import os
import joblib
from glob import glob
from time import time


def _load_joblib_persistence(file_path, readtype="rb"):
    """
    Load a specified *.joblib file from disk and return
    contents as 'model'
    """
    with open(file_path, readtype) as _f:
        model = joblib.load(_f)
    return model


def load_model(file_path, readtype='rb'):
    with open(file_path, readtype) as _f:
        model = joblib.load(_f)
    return model


def dump_model(model, file_path, writetype='wb', compress=3):
    with open(file_path, writetype) as _f:
        model = joblib.dump(model, _f, compress=compress)


def load_models(model_dir, fn_glob_str="*/*.joblib", verbose=True):
    """
    Load a series of *.joblib preserved models contained
    within the following directory/naming structure:
    {model_dir}/{kind}/{kind}_*.joblib
    or for kind = BAZ
    {model_dir}/{kind}/{kind}_{sub}_*.joblib

    into a dictionary

    :: INPUTS ::
    :param model_dir: [str] relative file path to the directory
        holding sub-directories of a given model {kind}
    :param fn_glob_str: [str] glob-compliant string to append
        to `model_dir` to get a list of model files.

    :: OUTPUTS ::
    :return mod_dict: [dictionary]
        Dictionary containing pb_SAM E3WS objects with keys denoting
        {kind} or in the case of BAZ {kind}{sub} where 
        {sub} is "Sin" or "Cos"
    """
    if verbose:
        tick = time()
    flist = glob(os.path.join(model_dir, fn_glob_str))
    mod_dict = {}
    for _f in flist:
        # Strip metadata from file-path string
        _path, _tail = os.path.split(_f)
        _kind = os.path.split(_path)[1]
        if _kind == "BAZ":
            _sub = _tail.split("_")[1]
        else:
            _sub = ""
        # Load model
        _model = _load_joblib_persistence(_f)
        mtype = _kind + _sub
        # If there is a new model type
        if mtype not in mod_dict.keys():
            # Create new dictionary entry
            mod_dict.update({mtype: _model})
        # Otherwise
        else:
            if isinstance(mod_dict[mtype], list):
                # Append new model to existing entry list
                mod_dict[mtype].append(_model)
            else:
                mod_dict.update({mtype: [mod_dict[mtype], _model]})
    if verbose:
        tock = time()
        print(f'runtime: {tock - tick: .3f} sec')
    return mod_dict


def save_models(mod_dict, save_dir='PNSN_model/temp_models',fstring='{KIND}/{KIND}{SUB}_{name}.joblib'):
    return None

def _display_model_loss(pb_SAM):
    loss_fn = pb_SAM.base_models[0][1].objective
    print(f'loss function "{loss_fn}"')

def _display_model_cost(pb_SAM):
    cost_fn = pb_SAM.meta_model[1]
    print(f'cost function "{cost_fn}"')