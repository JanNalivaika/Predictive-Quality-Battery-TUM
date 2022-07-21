import numpy as np
from scipy.stats import pearsonr, spearmanr, entropy
import pandas as pd
import matplotlib.pyplot as plt
import random as rd
import ntpath

def RMS(series: pd.DataFrame):
    rms = np.sqrt(np.sum(series**2)/len(series))
    return rms

def filter_OOT(df: pd.DataFrame):
    df = df[df['signal'] == 0]
    return df

def calc_stats():
    df_all = pd.read_excel("../All/All_Data_relabeled.xlsx")
    df_all = filter_OOT(df_all)
    labels = df_all.loc[:, "not OK":"LWMID_2"]
    labels.insert(4, "Lube", ((labels["WD40"] == 1) | (labels["Gleitmo"] == 1)).astype(int))

    df1 = df_all.loc[:, "Signal1_  1":"Signal1_112"]
    df1.columns = range(112)

    # df1_dn = pd.read_excel("../Data/Statistical_features/All_Data_stats.xlsx", sheet_name="S1_DN")
    df1_dn = df_all.loc[:, "Signal1_dn_  1":"Signal1_dn_112"]
    df1_dn.columns = range(112)

    # df2 = pd.read_excel("../Data/Statistical_features/All_Data_stats.xlsx", sheet_name="S2")
    df2 = df_all.loc[:, "Singal2_  1":"Singal2_112"]
    df2.columns = range(112)

    # df_corr = labels.copy(deep=True)
    # df_corr["Pearson"] = df1.corrwith(df2, axis=1, method='pearson')
    # df_corr["Spearman"] = df1.corrwith(df2, axis=1, method='spearman')
    #print(df_corr)
    #df_corr.to_excel("../Data/Statistical_features/S1_S2_corr.xlsx", index=False)

    dfs = [df1, df1_dn, df2]
    df_names = ["S1", "S1_DN", "S2"]

    for i in range(len(dfs)):
        df = dfs[i]
        df_stat = labels.copy(deep=True)
        df_stat['MEAN'] = df.mean(axis=1)
        df_stat['MEDIAN'] = df.median(axis=1)
        df_stat['STD'] = df.std(axis=1)
        df_stat['VARIANCE'] = df.var(axis=1)
        df_stat['SKEWNESS'] = df.skew(axis=1)
        df_stat['Q25'] = df.quantile(q=0.25, axis=1)
        df_stat['Q75'] = df.quantile(q=0.75, axis=1)
        df_stat['MAX'] = df.max(axis=1)
        df_stat['MIN'] = df.min(axis=1)
        df_stat['RMS'] = (df.pow(2, axis=1).sum(axis=1) / df.shape[1]).apply(np.sqrt, axis=1)
        df_stat['ENTROPY'] = df.apply(entropy, axis=1)
        #print(df_stat)
        df_stat.to_excel("../Statistical_features/" + df_names[i] + "_relabeled_stats.xlsx", index=False)
        pass

def plot_stats(paths, signal_names, labels, features): # label vs feature
    for i in range(len(paths)):
        path = paths[i]
        signal_name = signal_names[i]
        signal = pd.read_excel(path)
        for label in labels:
            for feature in features:
                plt.figure()
                plt.scatter(signal[feature], signal[label])
                # plt.savefig(signal_name + "_" + label + "_" + feature + ".png")

def plot_stats2(paths, labels, features): # feature value across samples
    for i in range(len(paths)):
        path = paths[i]
        title = ntpath.basename(path).split(".")[0].split("_")
        if title[1] == "DN": signal_name = title[0] + " " + title[1]
        elif title[1] == "S2": signal_name = title[0] + "-" + title[1]
        else: signal_name = title[0]
        signal = pd.read_excel(path)
        for label in labels:
            signal_0 = signal[signal[label] == 0]
            signal_1 = signal[signal[label] == 1]
            for feature in features:
                plt.figure()
                plt.scatter(signal_0.index, signal_0[feature], label=label + " == 0")
                plt.scatter(signal_1.index, signal_1[feature], label=label + " == 1", c='r')
                plt.xlabel("Sample")
                plt.ylabel(feature)
                plt.title(signal_name + " " + feature)
                plt.legend()
                # plt.savefig(signal_name + "_" + label + "_" + feature + "_relabeled.png")
                plt.show()

def plot_stats3(paths, labels, feature1, feature2): # feature vs feature
    for i in range(len(paths)):
        path = paths[i]
        signal_name = ntpath.basename(path).split(".")[0].split("_")[0]
        signal = pd.read_excel(path)
        for label in labels:
            signal_0 = signal[signal[label] == 0]
            signal_1 = signal[signal[label] == 1]
            plt.figure()
            plt.scatter(signal_0[feature1], signal_0[feature2], label=label + " == 0")
            plt.scatter(signal_1[feature1], signal_1[feature2], label=label + " == 1", c='r')

            # plt.axvline(x=200, color='black')
            # plt.axvline(x=401, color='black')
            # plt.axvline(x=716, color='black')
            # plt.axvline(x=826, color='black')
            # plt.axvline(x=949, color='black')
            # plt.axvline(x=1125, color='black')

            plt.xlabel(feature1)
            plt.ylabel(feature2)
            plt.legend()
            # plt.savefig(signal_name + "_" + feature1 + "_" + feature2 + ".png")

if __name__ == "__main__":
    # calc_stats()
    #signal_names = ["S1_S2"]
    paths = [
        # "..\Statistical_features\S1_relabeled_stats.xlsx",
        # "..\Statistical_features\S1_DN_relabeled_stats.xlsx"
        "../Data/Statistical_features/S2_relabeled_stats.xlsx"
        # "../Statistical_features/S1_S2_corr.xlsx"
        ]
    labels = [
        "not OK"
        # "Lube"
        # "Gleitmo"
        ]
    # features = [
    #     "Pearson",
    #     "Spearman"
    #     ]
    features = [
        # 'MEAN',
        # 'STD',
        # 'MAX',
        # 'RMS',
        # 'SKEWNESS',
        'ENTROPY'
        ]
    #plot_stats(paths, signal_names, labels, features)
    plot_stats2(paths, labels, features)
    #plot_stats3(paths, labels, 'MEAN', 'STD')

    #df_corr = pd.read_excel("../Data/Statistical_features/S1_S2_corr.xlsx")
    #plt.figure()
    #plt.scatter(df_corr["Pearson"], df_corr["signal"])
    #plt.show()
    
