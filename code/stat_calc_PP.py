import numpy as np
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import matplotlib.pyplot as plt
import random as rd

def RMS(series: pd.DataFrame):
    rms = np.sqrt(np.sum(series**2)/len(series))
    return rms

def filter_OOT(df: pd.DataFrame):
    df = df[df['signal'] == 0]
    return df

def calc_stats():
    df1 = pd.read_excel("../Data/Statistical_features/All_Data_stats.xlsx", sheet_name="S1")
    labels = df1.loc[:, "not OK":"LWMID_2"]
    labels.insert(4, "Lube", ((labels["WD40"] == 1) | (labels["Gleitmo"] == 1)).astype(int))
    df1 = df1.loc[:, "Signal1_  1":"Signal1_112"]
    #df1.columns = range(112)

    df1_dn = pd.read_excel("../Data/Statistical_features/All_Data_stats.xlsx", sheet_name="S1_DN")
    df1_dn = df1_dn.loc[:, "Signal1_dn_  1":"Signal1_dn_112"]
    #df1_dn.columns = range(112)

    df2 = pd.read_excel("../Data/Statistical_features/All_Data_stats.xlsx", sheet_name="S2")
    df2 = df2.loc[:, "Singal2_  1":"Singal2_112"]
    #df2.columns = range(112)

    df_corr = labels.copy(deep=True)
    df_corr["Pearson"] = df1.corrwith(df2, axis=1, method='pearson')
    df_corr["Spearman"] = df1.corrwith(df2, axis=1, method='spearman')
    df_corr.to_excel("../Data/Statistical_features/S1_S2_corr.xlsx", index=False)

    dfs = [df1, df1_dn, df2]
    df_names = ["S1", "S1_DN", "S2"]

    for i in range(len(dfs)):
        df = dfs[i]
        df_stat = labels.copy(deep=True)
        df_stat['Mean'] = df.mean(axis=1)
        df_stat['Median'] = df.median(axis=1)
        df_stat['STD'] = df.std(axis=1)
        df_stat['skew'] = df.skew(axis=1)
        #print(df_stat)
        df_stat.to_excel("../Data/Statistical_features/" + df_names[i] + "_stats.xlsx", index=False)
        pass

if __name__ == "__main__":
    #calc_stats()
    #df_corr = pd.read_excel("../Data/Statistical_features/S1_S2_corr.xlsx")
    #df1 = pd.read_excel("../Data/Statistical_features/S1_stats.xlsx")
    #plt.figure()
    #plt.scatter(df1["STD"], df1["Lube"])
    #plt.scatter(df_corr["Pearson"], df_corr["signal"])
    #plt.show()
    pass
