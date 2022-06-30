import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.signal import savgol_filter
from visualize_test import importRawSignals
import os


def percentilation(signal1, signal2, pct_val, num_pct_window, num_of_samples):

    """
    signal1, signal2: array mit signal
    pct_val: welches percentil soll berechnet werden, wert zw. 0 und 100
    num_pct_window: anzahl der bereiche in denen percentil berechnet wird
    num_of_samples: anzahl der random reihen die betrachtet werden
    """

    randomlist = []
    for i in range(0, num_of_samples):
        n = random.randint(1, 1300)
        randomlist.append(n)

    for x in randomlist:

        size_pct_window = int(signal1.shape[1]/num_pct_window)
        signal1_pct, signal2_pct = np.zeros(num_pct_window), np.zeros(num_pct_window)
        signal1_l = signal1[x, :]
        signal2_l = signal2[x, :]

        for i in range(num_pct_window-1):

            signal1_pct[i] = np.percentile( signal1[x ,i*size_pct_window : (i + 1) * size_pct_window], pct_val)
            signal2_pct[i] = np.percentile(signal2[x , i * size_pct_window:(i + 1) * size_pct_window], pct_val)


        r_pct = np.corrcoef((signal1_pct, signal2_pct))
        print('cov:', r_pct[0, 1])

        plt.figure()
        plt.title("Sample " + str(x) + " Correlation = " + str(r_pct[0, 1]))
        plt.plot(signal1_pct, label="Signal 1 percentile")
        plt.plot(signal2_pct, label="Signal 2 percentile")
        plt.legend()
        plt.show()

        ###plt.savefig("../DataCorrelation/" + str(x) + ".png", dpi=500)
        plt.close()


    pass


if __name__ == "__main__":
    print("Lets GOOOOO")
    datafile = "../Data/Datensatz_Batteriekontaktierung.xlsx"
    signal1, signal2 = importRawSignals(datafile)
    percentilation(signal1, signal2, pct_val=50, num_pct_window=30, num_of_samples=5)
    print("DONE")