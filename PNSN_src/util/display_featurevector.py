import os
import matplotlib.pyplot as plt
import numpy as np


def annotate_feature_vector(fv):
    three_c_features = ["max eigenvalue", "rectilinearity"] + [f"max eigenvector {_c}" for _c in "ENZ"]
    temporal_features = [
        "max energy",
        "max energy time (sec)",
        "time weighted mean amp",
        "amp var",
        "amp skew",
        "amp kurt",
        "total energy",
        "envelope threshold X rate",
        "max/mean envelope ratio",
        "envelope mean",
        "envelope std",
        "envelope skew",
        "envelope kurt",
        "envelope threshold X rate complement",
        "Shannon entropy",
        "Renyi entropy",
        "Zero-crossing rate",
    ]
    spectral_features = [
        "PSD max amp",
        "PSD max amp freq",
        "PSD centroid",
        "PSD bandwidth",
        "PSD skew",
        "PSD kurt",
        "PSD mean amp",
        "PSD std amp",
        "PSD skew amp",
        "PSD kurt amp",
        "PSD Shannon entropy",
        "PSD Renyi entropy",
        "PSD max/mean ratio",
        "PSD threshold X rate",
        "PSD threshold X rate complement",
    ]
    cepstral_features = [f"Mel-freq coef {x+1:d}" for x in range(13)]

    label_vector = three_c_features
    if len(np.nonzero(fv[:5])[0]) == 0:
        for _N in range(3):
            label_vector += [f"Z{_N}-{_f}" for _f in temporal_features]
            label_vector += [f"Z{_N}-{_f}" for _f in spectral_features]
            label_vector += [f"Z{_N}-{_f}" for _f in cepstral_features]

    else:
        for _C in "ENZ":
            label_vector += [f"{_C}-{_f}" for _f in temporal_features]
            label_vector += [f"{_C}-{_f}" for _f in spectral_features]
            label_vector += [f"{_C}-{_f}" for _f in cepstral_features]

    out = dict(zip(label_vector, fv))

    return out


def plot_feature_vectors(ax, f_array, label_list):
    f_index = np.arange(1, 141)
    if len(f_array.shape) > 1:
        if f_array.shape[0] == 140:
            f_array = f_array.T
    else:
        f_array = f_array.reshape(1, 140)

    for _i in range(f_array.shape[0]):
        if len(np.nonzero(f_array[_i, :5])[0]) == 0:
            ax.plot(f_index[5:52], f_array[_i, 5:52], ".-", label=label_list[_i])

        else:
            ax.plot(f_index, f_array[_i, :], ".-", label=label_list[_i])

    ylims = ax.get_ylim()
    xlims = ax.get_xlim()

    if xlims[0] < 4:
        ax.fill_between(
            [0.5, 5.5],
            [ylims[0]] * 2,
            [ylims[1]] * 2,
            color="k",
            alpha=0.1,
            label="Rectilinearity",
        )
    for _j in range(3):
        if _j == 0:
            labels = ["temporal", "spectral", "cepstral"]
        else:
            labels = [""] * 3
        ax.fill_between(
            [5.5 + _j * 45, 22.5 + _j * 45],
            [ylims[0]] * 2,
            [ylims[1]] * 2,
            color="b",
            alpha=0.1,
            label=labels[0],
        )
        ax.fill_between(
            [22.5 + _j * 45, 37.5 + _j * 45],
            [ylims[0]] * 2,
            [ylims[1]] * 2,
            color="r",
            alpha=0.1,
            label=labels[1],
        )
        ax.fill_between(
            [37.5 + _j * 45, 50.5 + _j * 45],
            [ylims[0]] * 2,
            [ylims[1]] * 2,
            color="y",
            alpha=0.1,
            label=labels[2],
        )
        ax.text(
            28 + _j * 45,
            (ylims[1] - ylims[0]) * 0.8 + ylims[0],
            ["East", "North", "Vertical"][_j],
            ha="center",
            va="center",
        )

    ax.legend(loc="lower left")
    ax.set_xlim(0, 141)
    ax.set_ylim(ylims)
