import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import random
import pandas as pd
from scipy.signal import savgol_filter



def importData(datafile):
    try:
        new_name = datafile.replace(".xlsx", ".npy")            #if file not created create first
        with open(new_name, 'rb') as f:
            df = np.load(f)
            #print(df)
    except:
        df = pd.read_excel(io=datafile)                         #if created do this
        print(df)
        df = np.asarray(df)
        print(df)
        new_name = datafile.replace(".xlsx", ".npy")
        with open(new_name, 'wb') as f:
            np.save(f, df)
        #print("we are here")
    return df

### slector of data signals ###
def selector(data, x):
    #x = random.randint(2, 200) # 0 - 200 WD 40 / 200 - 400 GLeitmo
    signal1 = data[x, 6:117]
    signal1_DN = data[x, 118:229]
    signal2 = data[x, 230:341]
    signal1_sample2 = data[x+200, 6:117]
    signal1_DN_sample2 = data[x+200, 118:229]
    signal2_sample2 = data[x+200, 230:341]

    return signal1, signal1_DN, signal2, signal1_sample2


def fourier(s1,s1DN, s2, s1_sample2):
    # Number of samplepoints
    N = 200
    # sample spacing
    T = 1.0 / 1000.0
    x = np.linspace(0.0, N * T, N)
    y = np.sin(50.0 * 2.0*np.pi*x) + np.sin(80.0 * 2.0*np.pi*x) + np.cos(180.0 * 2.0*np.pi*x)

    #signal1_DN_self = savgol_filter(s1, 111, 9)

    yf_1 = scipy.fftpack.fft(s1)
    #print(yf_1)
    yf_1DN = scipy.fftpack.fft(s1DN)
    #yf_2 = scipy.fftpack.fft(s2)
    #yf_1_sample2 = scipy.fftpack.fft(s1_sample2)
    #test = scipy.fftpack.fft(y)

    #yf_1DN_self = scipy.fftpack.fft(signal1_DN_self)


    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    #print(xf)

    ### plotting the results ###
    fig, ax = plt.subplots()
    ax.plot(xf, 2.0 / N * np.abs(yf_1[:N // 2]), label="Sample WD 40")

    ax.plot(xf, 2.0 / N * np.abs(yf_1DN[:N // 2]), label="Sample Gleitmo")

    #ax.plot(xf, 2.0 / N * np.abs(yf_1[:N // 2]), label="Signal 1")

    #ax.plot(xf, 2.0 / N * np.abs(yf_1DN[:N // 2]), label="Signal 1 DN")


    plt.legend()

    n_sample = i + 1
    sample2_NOK = data[(i + 200), 0]                                                                          #Variante 1 (WD 40 Gleitmo OK NOT OK Vergleich)
    sample1_NOK = data[i, 0]
    print(data[i + 200, 4], data[i +200 , 5])
    plt.title("Sample " + str(n_sample) + " NOK = " + str(int(sample1_NOK)) + " and " + str(n_sample + 200) + " NOK = " + str(int(sample2_NOK)))
    # fig_name = str("../Data/code_for_visualizations/Data_Visualization_plots/Fourier" + str(i + 1) + ".png")
    # plt.savefig(fig_name, dpi=500)
    plt.show()
    plt.close()


    # n_sample = i + 1
    # sample2_NOK = data[(i + 200), 0]
    # sample1_NOK = data[i, 0]                                                                                        #Variante 2 (Vergleich 1 and 1DN)
    # print(data[i + 200, 4], data[i + 200, 5])
    # plt.title(
    #     "Sample " + str(n_sample))
    # fig_name = str("../Data/code_for_visualizations/Data_Visualization_plots/Fourier" + str(i + 1) + ".png")
    # plt.savefig(fig_name, dpi=500)
    # plt.show()
    # plt.close()



if __name__ == "__main__":
    print("Lets GOOOOO")
    datafile = "../Data/All/All_Data.xlsx"
    data = importData(datafile)
    for i in range(0, 10):
        s1 , s1DN, s2, s1_sample2 = selector(data, i)[0:4]
        print(s1)
        fourier(s1,s1DN, s2, s1_sample2)
    pass



