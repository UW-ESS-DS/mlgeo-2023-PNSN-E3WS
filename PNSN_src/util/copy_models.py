import os
import sys
import joblib
from glob import glob
import pandas

# Map relative path to root directory
ROOT = os.path.join("..", "..")
sys.path.append(ROOT)
# Explicitly import pb_SAM Class for file reading
from PNSN_src.util.SAM import pb_SAM

# Get source path
flist = glob(os.path.join(ROOT, 'models', '*', '*7tp3*.joblib'))
# Define destiation format
ofname = os.path.join(ROOT, 'PNSN_model', 'Lara_2023_Preferred',
                      '{KIND}', '{KIND}{MODIFIER}_7tp3_Lara2023_preferred.joblib')

# Iterate across models
for ifile in flist:
    print(f'attempting to load {ifile}')
    # Try to load model
    try:
        with open(ifile, 'rb') as _i:
            model = joblib.load(_i)
    except:
        print('loading failed')
        break
    # If successful, strip kind and reformat name
    _path, _tail = os.path.split(ifile)
    _kind = os.path.split(_path)[1]
    if _kind == 'BAZ':
        _modifier = "_" + _tail.split("_")[2]
    else:
        _modifier = ''
    ofile = ofname.format(KIND=_kind, MODIFIER=_modifier)
    opath = os.path.split(ofile)[0]
    if not os.path.exists(opath):
        os.makedirs(opath)
    with open(ofile, 'wb') as _o:
        joblib.dump(model, _o, compress=3)
    