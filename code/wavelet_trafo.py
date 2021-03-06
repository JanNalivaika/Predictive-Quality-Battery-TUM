import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import random
from visualize_test import importData
import os
import pywt

def cwt(signal, db_name):

    rg_scales = 112  # arbitrary range, to be adjusted
    scales = range(1, rg_scales)
    wavelet = 'morl' # generic approach

    tr_shape = list(np.shape(signal))
    tr_shape.append(rg_scales-1)
    tr_shape = tuple(tr_shape)

    trafo = np.empty(tr_shape)

    for x in range(len(signal)):

        coefs, freqs = pywt.cwt(signal[x], scales, wavelet)
        trafo[x] = np.transpose(coefs)
    print(trafo.shape)
    np.save(db_name, trafo)


if __name__ == "__main__":
    db_name = "Signal1_DN_cwt_112.npy"
    datafile = "../Data/All/All_data.xlsx"
    data = importData(datafile)
    signal = data[:, 118:229]
    cwt(signal, db_name)
