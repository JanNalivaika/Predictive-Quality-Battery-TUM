import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from visualize_test import importRawSignals

def read_pandas_to_numpy(datafile):
    df = pd.read_excel(io=datafile)
    signal = df.loc[:, 'Signal1_  1':'Signal1_112']
    ok_label = df.loc[:, 'not OK']
    signal = np.asarray(signal)
    ok_label = np.asarray(ok_label)

    return signal, ok_label


def percentilation(signal1, ok_label, pct_val, num_pct_window, num_of_samples):

    """
    signal1: array mit signal
    nok_label: y vektor mit nok label (0=ok, 1=nok), wird nur im plot verwendet
    pct_val: welches percentil soll berechnet werden, wert zw. 0 und 100
    num_pct_window: anzahl der bereiche in denen percentil berechnet wird
    num_of_samples: anzahl der random reihen die betrachtet werden
    """

    size_pct_window = int(signal1.shape[1] / num_pct_window)

    randomlist = []
    for i in range(0, num_of_samples):
        n = random.randint(1, 1200)
        randomlist.append(n)

    for x in randomlist:
        signal1_pct = np.zeros(num_pct_window)

        for i in range(num_pct_window-1):
            signal1_pct[i] = np.percentile( signal1[x ,i*size_pct_window : (i + 1) * size_pct_window], pct_val)

        plt.figure()
        plt.title("Sample " + str(x) + '  not OK:' + str(ok_label[x]) )
        plt.plot(signal1_pct, label="Signal percentiles")
        plt.plot( np.linspace(0, num_pct_window, 112) ,signal1[x], label="Signal")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    print("Lets GOOOOO")

    datafile = "../Data/S1.xlsx"
    signal1, nok_label = read_pandas_to_numpy(datafile)
    percentilation(signal1, nok_label, pct_val=91, num_pct_window=10, num_of_samples=20)
    print("DONE")