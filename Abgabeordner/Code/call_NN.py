import numpy as np
import pandas as pd
from NN_Kvalidation import main as mainKval
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

"""
min_num_hidden_layers: minimale anzahl der betrachteten hidden layer
max_num_hidden_layers: maximale anzahl der betrachteten hidden layer
min_num_neurons: maximale anzahl der betrachteten knoten
max_num_neurons: maximale anzahl der betrachteten knoten
step_neurons: schrittweite der betrachteten kotenzahl
#
Für alle kombinationen wird der Mittelwert der Test Accuracy, die Standardabweichung und die false positives/negatives
gespeichert und ausgegeben
"""

def importSignal(datafile):
    df = pd.read_excel(io=datafile)

    ok_label = df.loc[:, 'not OK']
    ok_label = np.asarray(ok_label)

    df_just_signal = df.drop(['not OK', 'WD40', 'Gleitmo'], axis=1)
    signal1 = np.asarray(df_just_signal)

    return signal1, ok_label

def importStatFeatures(datafile):
    df = pd.read_excel(io=datafile)
    OOT_index = df.index[df['signal'] == 1].tolist()
    df = df.drop(OOT_index, axis=0)

    nok = df.loc[:, 'not OK']
    signal_std = df.loc[:, 'STD']
    signal_mean = df.loc[:, 'MEAN']
    signal_rms = df.loc[:, 'RMS']

    WD40 = df.loc[:, 'WD40']
    Gleitmo = df.loc[:, 'Gleitmo']

    nok = np.asarray(nok)
    signal_std = np.asarray(signal_std)

    WD40 = np.asarray(WD40)
    Gleitmo = np.asarray(Gleitmo)
    Lubricant = WD40 + Gleitmo

    return nok, signal_std, signal_mean, signal_rms, WD40, Gleitmo, Lubricant

min_num_hidden_layers = 1
max_num_hidden_layers = 1

min_num_neurons = 75
max_num_neurons = 75

step_neurons = 10

hid_layers = np.arange(start=min_num_hidden_layers, stop=max_num_hidden_layers + 1, step = 1)
neurons = np.arange(start=75, stop=max_num_neurons + 1, step=step_neurons)

x1 = hid_layers.size
x2 = neurons.size
x3 = 2

# arrays in denen die ergebnisse gespeichert werden
mean_err = np.zeros((x1, x2, x3))
fn_fp = np.zeros((x1, x2, x3))

# import data
datafile = "../Data/Statistical_features/S1_DN_relabeled.xlsx"
RAW, NOK = importSignal(datafile)

datafile = "../Data/Statistical_features/S1_DN_relabeled_stats.xlsx"
_, signal_std, signal_mean, signal_rms, WD40, Gleitmo, Lubricant = importStatFeatures(datafile)

# Input Daten durch auskommentieren auswählen
X = RAW
#X = np.vstack((signal_mean))
#X = np.vstack((signal_std))
#X = np.vstack((signal_mean, signal_std)).T
#X = np.vstack((signal_rms))

# Label durch auskommentieren wählen
y = NOK
#y = Lubricant
#y = WD40
#y = Gleitmo

dim_input = X.shape[1]

# 10-fold stratified, random seat um vergleichbarkeit zwischen modellen zu gewährleisten
num_splits = 10
skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)

#hilfsarries
temp_acc = np.zeros((num_splits))
temp_fn = np.zeros((num_splits))
temp_fp = np.zeros((num_splits))


for l in range(0, hid_layers.size):

    for n in range(0, neurons.size):
        print('l:', hid_layers[l], ' n:', neurons[n])

        split_iteration = 0
        for train_index, test_index in skf.split(X, y):
            print('fold nr:', split_iteration+1)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            temp_acc[split_iteration], temp_fn[split_iteration], temp_fp[split_iteration] = mainKval(dim_input, hid_layers[l], neurons[n], X_train, y_train, X_test, y_test)
            split_iteration += 1

        mean_err[l, n, 0] = np.mean(temp_acc)
        mean_err[l, n, 1] = np.std(temp_acc)
        fn_fp[l, n, 0] = np.mean(temp_fn)
        fn_fp[l, n, 1] = np.mean(temp_fp)

print('\nmean acc & std:',mean_err, '\n\nfalse negatives & false positves:', fn_fp)


for h in range(0, hid_layers.size):
    fig = plt.figure()
    plt.title("No. of hidden layers: " + str(hid_layers[h]))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of neurons in hidden layers")
    plt.ylim([0, 100])
    plt.xlim([0, 100])
    plt.yticks(np.arange(0, 100 + 1, 10))
    plt.scatter(neurons, mean_err[h, :, 0], s=20, marker='x')
    plt.errorbar(neurons, mean_err[h, :, 0], mean_err[h, :, 1], capsize=10,
             capthick=1, ls='none')
    #plt.show()




