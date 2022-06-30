
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.signal import savgol_filter
import os


def importData(datafile):
    try:
        new_name = datafile.replace(".xlsx", ".npy")
        with open(new_name, 'rb') as f:
            df = np.load(f)
    except:
        df = pd.read_excel(io=datafile)
        df = np.asarray(df)
        new_name = datafile.replace(".xlsx", ".npy")
        with open(new_name, 'wb') as f:
            np.save(f, df)

    return df

def importRawSignals(datafile):
    try:
        new_name = datafile.replace(".xlsx", ".npy")
        with open(new_name, 'rb') as f:
            df = np.load(f)
            data = df
    except:
        df = pd.read_excel(io=datafile)
        df = np.asarray(df)
        new_name = datafile.replace(".xlsx", ".npy")
        with open(new_name, 'wb') as f:
            np.save(f, df)
            data = df

    signal1 = data[:, 6:117]
    signal2 = data[:, 230:341]

    return signal1, signal2



def testCorrelation(data):

    #dir = '../DataCorrelation'
    #for f in os.listdir(dir):
    #    os.remove(os.path.join(dir, f))


    randomlist = []
    for i in range(0, 20):
        n =  random.randint(1, 1300)
        randomlist.append(n)

    for x in randomlist:
        # sensor 1 = idx 6 + x
        # sensor 1 DN = idx 118 + x
        # Sensor 2 =  230 + x



        signal1 = data[x, 6:117]
        signal1_DN = data[x, 118:229]
        signal2 = data[x, 230:341]

        signal2_DN = savgol_filter(signal2, 111, 9) # window size , polynomial order
        signal1_DN_s  = savgol_filter(signal1, 111, 9)

        percentile_val = 91
        num_of_percentile_windows = 30
        #print(signal1.size)
        window_size = int(signal1.size/num_of_percentile_windows)
        #print(window_size)

        signal1_pct, signal2_pct = np.zeros(num_of_percentile_windows), np.zeros(num_of_percentile_windows)


        for i in range(num_of_percentile_windows-1):

            signal1_pct[i] = np.percentile(signal1[i*window_size:(i+1)*window_size], percentile_val)
            signal2_pct[i] = np.percentile(signal2[i*window_size:(i+1)*window_size], percentile_val)

        ave1 = np.average(signal1_DN_s)
        ave2 = np.average(signal2_DN)


        #signals = np.vstack((signal1_DN_s, signal2_DN))



        #r = np.corrcoef(signal1_DN_s, signal2_DN)
        r_pct = np.corrcoef((signal1_pct, signal2_pct))
        print('cov:', r_pct[0, 1])

        plt.figure()
        plt.title("Sample " + str(x) + " Correlation = " + str(r_pct[0, 1]))

       # plt.plot(signal1, label="Signal 1")
        #plt.plot(signal2, label="Signal 2")
        #plt.plot(signal1_DN, label="Signal 1 DN original")
        #plt.plot(signal2_DN, label="Signal 2 DN")
        #plt.plot(signal1_DN_s, label="Signal 1 DN self made")
        plt.plot(signal1_pct, label="Signal 1 percentile")
        plt.plot(signal2_pct, label="Signal 2 percentile")
        plt.legend()
        plt.show()
        #plt.savefig("../DataCorrelation/" + str(x) + ".png", dpi=500)
        plt.close()
##
    pass


if __name__ == "__main__":
    print("Lets GOOOOO")
    datafile = "../Data/Datensatz_Batteriekontaktierung.xlsx"
    data = importData(datafile)
    testCorrelation(data)
    print("DONE")
