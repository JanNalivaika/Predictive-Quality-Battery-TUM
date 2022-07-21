import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#### data import ####
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

###  Selection of Input based on ModeX variable ###
def selectorInput(ModeX):
    rmsvalues = pd.Series([RMS(e[1]) for e in data_just_signal.iterrows()])
    if ModeX == "RAW":
        X = np.asarray(data_just_signal)
    elif ModeX == "mean":
        X = mean.to_numpy().reshape(-1, 1)
    elif ModeX =="std":
        X = std.to_numpy().reshape(-1, 1)
    elif ModeX == "rms":
        rmsvalues = pd.Series([RMS(e[1]) for e in data_just_signal.iterrows()])
        X = rmsvalues.to_numpy().reshape(-1, 1)
    elif ModeX == "mean_and_std":
        X = np.stack((std.to_numpy(), mean.to_numpy()), axis = 1).reshape(-1, 2)
    return X

###  Selection of Output based on ModeY variable ###
def selectorOutput(modeY):
    if modeY == "NOK":
        y = data["not OK"].to_numpy()
    elif modeY == "LUBE":
        y = data["WD40"].to_numpy() + data["Gleitmo"].to_numpy()
        y = np.minimum(y, 1)
    elif modeY == "WD40":
        y = data["WD40"].to_numpy()
    elif modeY == "Gleitmo":
        y = data["Gleitmo"].to_numpy()
    return y
#### training of log reg model and calculation of score and fp fn
def LogisticEval(X_train, X_test, y_train, y_test):
    regressor = LogisticRegression()
    regressor.fit(X_train, y_train)

    y_hat = regressor.predict(X_test)

    ###calc of false negatives and false positives###

    np_false_negative = np.zeros(len(y_hat))
    np_false_positive = np.zeros(len(y_hat))
    np_false_positive[y_hat > y_test] = 1
    np_false_negative[y_hat < y_test] = 1
    false_negative = np_false_negative.sum()                                                            #sum of all flags set to 1
    false_positive = np_false_positive.sum()
    #print("false positives: ", false_positive)
    #print("false negatives: ", false_negative)

    # x_scatter_1 = np.arange(len(y_test))
    # plt.scatter(x_scatter_1, y_test)
    # # plt.scatter(x_scatter_1, y_hat, c="red")
    # plt.show()
    # result = np.stack((y_hat, y_test), axis=1)
    # print(result)

    ### Pipeline Solution ###

    # pipeline_classification = Pipeline(steps=[('model', LogisticRegression(max_iter=1e5))])
    #
    # pipeline_classification.fit(X_train, y_train)
    #
    # score = pipeline_classification.score(X_test, y_test)
    # print('The accuracy of the classifier is: %.6f' % score)

    ### calc of score  ####
    score = regressor.score(X_test, y_test)

    return false_positive, false_negative, score

def main(input_name, output_name):

    ### RAW, mean, std, rms, mean_and_std as input ###
    X = selectorInput(input_name)
    print(input_name)
    ### NOK, WD40, Gleitmo, LUBE as Output ###
    y = selectorOutput(output_name)

    ###cross validation###
    number_splits = 10
    score_mean = 0
    np_false_positive_total = 0
    np_false_negative_total = 0
    score_array = []
    skf = StratifiedKFold(n_splits=number_splits, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        np_false_positive, np_false_negative, score = LogisticEval(X_train,X_test, y_train, y_test)   # fp, fn, score for current fold (1 of number_splits)

        score_array.append(score)                                                                     # append current score to the score array
        np_false_negative_total += np_false_negative                                                  # sum up all the fns over each iter
        np_false_positive_total += np_false_positive                                                  # """""""""""""  fps """"""""""""""

    score_mean = np.mean(score_array)
    standard_dev = np.std(score_array)
    np_false_positive_total = np_false_positive_total / number_splits
    np_false_negative_total = np_false_negative_total / number_splits
    print("mean score: ", score_mean)
    print("mean false negative: ", np_false_negative_total)
    print("mean false positive: ", np_false_positive_total)
    print("Standard deviation: ", standard_dev)
    return score_mean, standard_dev,score

    ###normal train-test-split###

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, shuffle=True)  # ;)

    #print(X_train, X_test)

if __name__ == "__main__":
    print("Lets GOOOOO")
    S1 = "../Data/S1.xlsx"
    S1_DN = "../Data/S1_DN.xlsx"
    S2 = "../Data/S2.xlsx"

    data, data_just_signal = read_pandas(S1_DN)                                                 #loading data
    mean, std = mean_and_std(data_just_signal)                                                  #calc mean and std
    rmsvalues = pd.Series([RMS(e[1]) for e in data_just_signal.iterrows()])                     #calc rms

    #for i in ["RAW", "mean", "std", "rms", "mean_and_std"]#
    score_mean, standard_dev, score = main("RAW", "NOK")
    ### plot the resulting score with its spread ###
    fig = plt.figure()
    plt.axis([0,0, 0, 100])
    plt.title("Accuracy of Log Reg: ")
    plt.ylabel("Accuracy in %")
    plt.scatter(0, score_mean * 100, s=20, marker='x')
    plt.errorbar(0, score_mean * 100, standard_dev * 100, capsize=10,
                capthick=1, ls='none')
    plt.show()
    # fig.savefig('../Data/code_for_visualizations/Data_Visualization_plots/Log Reg Accuracies/Accuracy_Log_Reg_on_RAW_NOK_0-100.jpg', bbox_inches='tight', dpi=250)
