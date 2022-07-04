import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import random
import pandas as pd
from scipy.signal import savgol_filter



def importData(datafile):
    try:
        new_name = datafile.replace(".xlsx", ".npy")
        with open(new_name, 'rb') as f:
            df = np.load(f)
            #print(df)
    except:
        df = pd.read_excel(io=datafile)
        print(df)
        df = np.asarray(df)
        print(df)
        new_name = datafile.replace(".xlsx", ".npy")
        with open(new_name, 'wb') as f:
            np.save(f, df)
        #print("we are here")
    return df

def selector(data):
    x = random.randint(1, 1300) # 0
    signal1 = data[x, 6:117]
    signal1_DN = data[x, 118:229]
    signal2 = data[x, 230:341]

    return signal1, signal1_DN, signal2, x


def fourier(s1,s1DN, s2):
    # Number of samplepoints
    N = 200
    # sample spacing
    T = 1.0 / 1000.0
    x = np.linspace(0.0, N * T, N)
    y = np.sin(50.0 * 2.0*np.pi*x) + np.sin(80.0 * 2.0*np.pi*x) + np.cos(180.0 * 2.0*np.pi*x)

    signal1_DN_self = savgol_filter(s1, 111, 9)

    yf_1 = scipy.fftpack.fft(s1)
    #print(yf_1)
    yf_1DN = scipy.fftpack.fft(s1DN)
    yf_2 = scipy.fftpack.fft(s2)

    test = scipy.fftpack.fft(y)

    yf_1DN_self = scipy.fftpack.fft(signal1_DN_self)


    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    #print(xf)

    fig, ax = plt.subplots()
    ax.plot(xf, 2.0 / N * np.abs(yf_1[:N // 2]), label="S1")
    ax.plot(xf, 2.0 / N * np.abs(yf_1DN[:N // 2]), label="S1DN")
    ax.plot(xf, 2.0 / N * np.abs(yf_2[:N // 2]), label="S2")
    delme1 = 2.0 / N * np.abs(yf_1[:N // 2])
    delme2 = 2.0 / N * np.abs(yf_2[:N // 2])
    r = np.corrcoef(delme1, delme2)
    ax.plot(xf, 2.0 / N * np.abs(yf_1DN_self[:N // 2]), label="S1DN_self")
    #ax.plot(xf, 2.0 / N * np.abs(test[:N // 2]), label="S1DN_self")
    plt.legend()
    n_sample = selector(data)[3]
    plt.title("Sample " + str(n_sample))
    plt.savefig("../DataCorrelation/Fourier.png", dpi=500)
    plt.show()
    plt.close()




if __name__ == "__main__":
    print("Lets GOOOOO")
    datafile = "../Data/All/All_Data.xlsx"
    data = importData(datafile)
    s1 , s1DN, s2 = selector(data)[0:3]
    fourier(s1,s1DN, s2)
    pass



