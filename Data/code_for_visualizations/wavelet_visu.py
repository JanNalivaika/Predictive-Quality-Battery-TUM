from turtle import forward
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from visualize_test import importData
import os
import pywt

def plot_cwt(data):

    scales = range(1, 112)
    wavelet = 'morl'

    randomlist = []
    for i in range(0, 4):
        n = random.randint(1, 1349)
        while data[n, 0] == 1: n = random.randint(1, 1349) # filter ok/nok
        label = " (OK)"
        randomlist.append(n)

    for x in randomlist:
        
        # signal = data[x, 6:117]
        signal = data[x, 118:229]
        # signal2 = data[x, 230:341]
        
        coefs1, freqs1 = pywt.cwt(signal, scales, wavelet)
        # coefs1_dn, freqs1_dn = pywt.cwt(signal1_DN, scales, wavelet)
        # coefs2, freqs2 = pywt.cwt(signal2, scales, wavelet)
        
        plt.figure()
        plt.title("Sample " + str(x) + " DN" + label)
        plt.imshow(coefs1)
        # plt.pcolormesh(range(0, len(signal1)), freqs1, coefs1)
        plt.colorbar()
        plt.xlabel("Time")
        plt.ylabel("Scale")

        plt.savefig("Sample " + str(x) + " DN" + label + ".png")
    
    plt.show()



if __name__ == "__main__":
    datafile = "../Data/All/All_Data_relabeled.xlsx"
    data = importData(datafile)
    plot_cwt(data)
    #print(pywt.central_frequency('morl'))

