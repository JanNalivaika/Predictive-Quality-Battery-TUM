import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import random
import pandas as pd

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

def selector(data):
    x = random.randint(1, 1300)
    signal1 = data[x, 6:117]
    signal1_DN = data[x, 118:229]
    signal2 = data[x, 230:341]

    return signal1, signal1_DN, signal2


def fourier(s1,s1DN, s2):
    # Number of samplepoints
    N = 111
    # sample spacing
    T = 1.0 / 800.0
    x = np.linspace(0.0, N * T, N)
    y = np.sin(50.0 * 2.0*np.pi*x) + np.sin(80.0 * 2.0*np.pi*x) + np.sin(180.0 * 2.0*np.pi*x)
    yf_1 = scipy.fftpack.fft(s1)
    yf_1DN = scipy.fftpack.fft(s1DN)
    yf_2 = scipy.fftpack.fft(s2)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

    fig, ax = plt.subplots()
    ax.plot(xf, 2.0 / N * np.abs(yf_1[:N // 2]), label="S1")
    ax.plot(xf, 2.0 / N * np.abs(yf_1DN[:N // 2]), label="S1DN")
    ax.plot(xf, 2.0 / N * np.abs(yf_2[:N // 2]), label="S2")
    plt.legend()
    plt.savefig("../DataCorrelation/Fourier.png", dpi=500)
    plt.close()
    plt.show()



if __name__ == "__main__":
    print("Lets GOOOOO")
    datafile = "../Data/Datensatz_Batteriekontaktierung.xlsx"
    data = importData(datafile)
    s1 , s1DN, s2 = selector(data)
    fourier(s1,s1DN, s2)
    pass



