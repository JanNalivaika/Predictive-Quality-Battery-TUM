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


if __name__ == "__main__":
    print("Lets GOOOOO")
    S1 = "../Data/S1.xlsx"
    S1_DN = "../Data/S1_DN.xlsx"
    S2 = "../Data/S2.xlsx"

    #TH1 = "../OOT/S1_OOT_ONLY.xlsx"
    #TH1_DN = "../OOT/S1_DN_OOT_ONLY.xlsx"
    #TH2 = "../OOT/S2_OOT_ONLY.xlsx"

    data, data_just_signal = read_pandas(S1_DN)
    mean, std = mean_and_std(data_just_signal)

    rmsvalues = pd.Series([RMS(e[1]) for e in data_just_signal.iterrows()])



    X = data_just_signal                                                            #RAW DATA as feature -> Accurace = 0.910
    #X = mean.to_numpy().reshape(-1, 1)                                              #mean as feature -> Accurace = 0.81
    X = std.to_numpy().reshape(-1, 1)                                               #std as feature -> Accuracy = 0.785
    #X = rmsvalues.to_numpy().reshape(-1, 1)                                         #rms as feature -> Accuracy = 0.822
    #X = np.stack((std.to_numpy() ,mean.to_numpy()), axis = 1).reshape(-1, 2)        #mean and std as feature ->  Accuracy = 0.822


    #y = data["not OK"].to_numpy()



    #y = data["WD40"].to_numpy()                                                        # ->  Accuracy = 0.668 with RAW
    #y  = data["Gleitmo"].to_numpy()                                                   # ->  Accuracy = 0.696 with RAW
    y = data["WD40"].to_numpy() + data["Gleitmo"].to_numpy()
    y =  np.minimum(y, 1)                                                               #   Accuracy = 0.67
    print(y)



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, shuffle=True) #, random_state = 43)  # ;)

    #print(X_train, X_test)

    ### Pipeline Solution ###

    pipeline_classification = Pipeline(steps=[('model', LogisticRegression(max_iter=1e5))])

    pipeline_classification.fit(X_train, y_train)

    score = pipeline_classification.score(X_test, y_test)
    print('The accuracy of the classifier is: %.3f' % score)


    #regressor = LogisticRegression()
    #regressor.fit(X_train, y_train)

    #y_hat = regressor.predict(X_test)

    #plt.scatter(X_test, y_test)
    #plt.show()
    #print(type(y_train))
    #result = np.stack((y_hat, y_test), axis = 1)
    #print(result)

    #score = regressor.score(X_test, y_test)

    #print(score)

    print("DONE")