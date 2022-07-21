
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


def testCorrelation(data): # visualization plot for random samples

    # dir = '../Data/correlation_visu'
    # for f in os.listdir(dir):
    #     os.remove(os.path.join(dir, f))


    randomlist = []
    for i in range(0, 3):
        n = random.randint(1, 1300)
        randomlist.append(n)

    for x in randomlist:
        # sensor 1 = idx 6 + x
        # sensor 1 DN = idx 118 + x
        # Sensor 2 =  230 + x

        if data[x, 0] == 1: label = " (not OK, "
        else: label = " (OK, "
        
        if data[x, 2] == 1: label += "WD40) "
        elif data[x, 3] == 1: label += "Gleitmo) "
        else: label += "no lube) "

        signal1 = data[x, 6:117].tolist()
        signal1_DN = data[x, 118:229]
        signal2 = data[x, 230:341]

        signal2_DN = savgol_filter(signal2, 111, 9) # window size , polynomial order
        signal1_DN_s  = savgol_filter(signal1, 111, 9)

        ave1 = np.average(signal1_DN_s)
        ave2 = np.average(signal2_DN)

        #signal1_DN_s = signal1_DN_s + (ave2 - ave1)


        r = np.corrcoef(signal1_DN_s, signal2_DN)

        plt.figure()
        plt.title("Sample " + str(x) + label + " Correlation = " + str(r[0, 1]))

        plt.plot(signal1, label="Signal 1")
        plt.plot(signal2, label="Signal 2")
        plt.plot(signal1_DN, label="Signal 1 DN original")
        plt.plot(signal2_DN, label="Signal 2 DN")
        plt.plot(signal1_DN_s, label="Signal 1 DN self made")
        plt.xlabel('Measuring point')
        plt.ylabel("Signal value")
        plt.legend()
        # plt.savefig("../Data/correlation_visu/" + str(x) + label + ".png", dpi=500)
        # plt.close()
        plt.show()
    
    pass


if __name__ == "__main__":
    print("Lets GOOOOO")
    datafile = "../Data/All/All_data.xlsx"
    data = importData(datafile)
    testCorrelation(data)
    print("DONE")
