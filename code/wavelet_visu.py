import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from visualize_test import importData
import os
import pywt

def plot_cwt(data):

    scales = range(1, 3000) # arbitrary range, to be adjusted
    wavelet = 'morl' # generic approach

    randomlist = []
    for i in range(0, 3):
        n = random.randint(1, 1300)
        #while data[n, 0] == 0: n = random.randint(1, 1300) # filter for nok
        randomlist.append(n)

    for x in randomlist:
        # sensor 1 = idx 6 + x
        # sensor 1 DN = idx 118 + x
        # Sensor 2 =  230 + x
        
        signal1 = data[x, 6:117]
        signal1_DN = data[x, 118:229]
        signal2 = data[x, 230:341]

        x = np.linspace(0.0, 1, 111)
        y = np.sin(50.0 * 2.0 * np.pi * x) + np.sin(80.0 * 2.0 * np.pi * x) + np.sin(180.0 * 2.0 * np.pi * x)

        coefs1, freqs1 = pywt.cwt(signal1, scales, wavelet)
        coefs1_dn, freqs1_dn = pywt.cwt(signal1_DN, scales, wavelet)
        coefs2, freqs2 = pywt.cwt(signal2, scales, wavelet)

        #print(np.shape(coefs1))

        plt.figure()
        plt.title("Sample " + str(x))

        plt.pcolormesh(range(0,len(signal1)), freqs1, coefs1)
        plt.colorbar()
        plt.xlabel("'time'")
        plt.ylabel("'frequency'")
        #plt.yscale('log')
    
    plt.show()



if __name__ == "__main__":
    datafile = "../Data/Datensatz_Batteriekontaktierung.xlsx"
    data = importData(datafile)
    plot_cwt(data)
