import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#df = pd.read_excel(io = S1)

#S1 = np.asarray(pd.read_excel(io=S1))

#S1_DN = np.asarray(pd.read_excel(io=S1_DN))
#S2 = np.asarray(pd.read_excel(io=S2))
#TH1 = np.asarray(pd.read_excel(io=TH1))
#TH1_DN = np.asarray(pd.read_excel(io=TH1_DN))
#TH2 = np.asarray(pd.read_excel(io=TH2))


#def scatter_plot:



def read_pandas(datafile):
    df = pd.read_excel(io=datafile)
    return df

def mean_and_std (pandas_df):
    mean = pandas_df.mean(axis = 1)
    std = pandas_df.std(axis = 1)
    return mean, std

def RMS(series):
    rms = np.sqrt(np.sum(series**2)/len(series))
    return rms


if __name__ == "__main__":
    print("Lets GOOOOO")
    S1 = "../S1.xlsx"
    S1_DN = "../S1_DN.xlsx"
    S2 = "../S2.xlsx"

    TH1 = "../OOT/S1_OOT_ONLY.xlsx"
    TH1_DN = "../OOT/S1_DN_OOT_ONLY.xlsx"
    TH2 = "../OOT/S2_OOT_ONLY.xlsx"
    data = read_pandas(S1)
    mean, std = mean_and_std(data)
    rmsvalues = pd.Series([RMS(e[1]) for e in data.iterrows()])
    x = np.arange(len(mean))
    plt.title("mean")                       #MEAN
    plt.scatter(x, mean)
    plt.savefig("Data_Visualization_plots/Statistical_approach/Mean_S1_w.o_OTT.png", dpi=500)

    plt.figure()
    plt.title("std")                        #STD
    plt.scatter(x, std)
    plt.savefig("Data_Visualization_plots/Statistical_approach/Std_S1_w.o_OTT.png", dpi=500)


    plt.figure()
    plt.title("rms")                        #RMS
    plt.scatter(x, rmsvalues)
    plt.savefig("Data_Visualization_plots/Statistical_approach/RMS_S1_w.o_OTT.png", dpi=500)






    plt.show()



    print("DONE")

