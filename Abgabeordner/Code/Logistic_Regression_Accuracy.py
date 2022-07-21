import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def read_pandas(datafile):
    df = pd.read_excel(io=datafile)
    df_just_signal = df.drop(['not OK', 'WD40', 'Gleitmo'], axis = 1)
    return df, df_just_signal

def mean_and_std (pandas_df):
    mean = pandas_df.mean(axis = 1)
    std = pandas_df.std(axis = 1)
    return mean, std

def RMS(series):
    rms = np.sqrt(np.sum(series**2)/len(series))
    return rms

def importStatFeatures(datafile):
    df = pd.read_excel(io=datafile)
    OOT_index = df.index[df['signal'] == 1].tolist()
    df = df.drop(OOT_index, axis=0)

    nok = df.loc[:, 'not OK']
    signal_std = df.loc[:, 'STD']

    WD40 = df.loc[:, 'WD40']
    Gleitmo = df.loc[:, 'Gleitmo']

    nok = np.asarray(nok)
    signal_std = np.asarray(signal_std)

    WD40 = np.asarray(WD40)
    Gleitmo = np.asarray(Gleitmo)
    Lubricant = WD40 + Gleitmo

    return nok, signal_std, WD40, Gleitmo, Lubricant

def importSignal(datafile):
    df = pd.read_excel(io=datafile)


    ok_label = df.loc[:, 'not OK']
    ok_label = np.asarray(ok_label)

    df_just_signal = df.drop(['not OK', 'WD40', 'Gleitmo'], axis=1)
    signal1 = np.asarray(df_just_signal)

    return signal1, ok_label

if __name__ == "__main__":
    print("Lets GOOOOO")

    check_Gleitmo = True

    if check_Gleitmo == False:

        S1_relabeled = "../Data/Signals_relabeled/S1_relabeled.xlsx"
        label, signal = read_pandas(S1_relabeled)
        X = signal                                                            #RAW DATA as feature -> Accurace = 0.910
        y = label["not OK"].to_numpy()


    elif check_Gleitmo == True:
        datafile = "../Data/Statistical_features/S1_relabeled_stats.xlsx"
        nok, signal_std, WD40, Gleitmo, Lubricant = importStatFeatures(datafile)
        datafile = "../Data/All_Data/S1.xlsx"
        signal1, _ = importSignal(datafile)

    # histogramm of the labels
    # plt.hist(y, 2)
    # plt.show()

    X = np.vstack((nok, signal_std)).T

    y = Gleitmo


    num_iterations = 30
    temp_acc = np.zeros((num_iterations))

    for i in range(num_iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, shuffle=True)


        ### Pipeline Solution ###

        pipeline_classification = Pipeline(steps=[('model', LogisticRegression(max_iter=1e5))])

        pipeline_classification.fit(X_train, y_train)

        score = pipeline_classification.score(X_test, y_test)
        score = score*100
        temp_acc[i] = score
        print('The accuracy of the classifier is: %.3f ' % score)

    acc_mean = np.mean(temp_acc)
    acc_err = np.max(temp_acc) - np.mean(temp_acc)
    print('Accuracy mean value:', acc_mean, ' Accuracy error:', acc_err)

