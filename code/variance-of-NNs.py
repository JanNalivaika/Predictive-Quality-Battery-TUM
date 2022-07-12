import numpy as np
import pandas as pd
from NN_v2 import main
from NN_Kvalidation import main as mainKval
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

"""
num_hidden_layers: maximale anzahl der betrachteten hidden layer
num_neurons: maximale anzahl der betrachteten knoten
step_neurons: schrittweite der betrachteten kotenzahl
#
Für alle kombinationen wird der Mittelwert der Test Accuracy und die Varianz gespeichert
"""

def importSignal(datafile):
    df = pd.read_excel(io=datafile)
    ok_label = df.loc[:, 'not OK']
    ok_label = np.asarray(ok_label)

    df_just_signal = df.drop(['not OK', 'WD40', 'Gleitmo'], axis=1)
    signal1 = np.asarray(df_just_signal)

    return signal1, ok_label



num_hidden_layers = 6
num_neurons = 100
step_neurons = 5
num_iterations = 50

hid_layers = np.arange(start=0, stop=num_hidden_layers+1, step = 1)
neurons = np.arange(start=5, stop=num_neurons+1, step=step_neurons)

x1 = hid_layers.size
x2 = neurons.size
x3 = 2
mean_err = np.zeros((x1, x2, x3))

#import data
datafile = "../Data/S1_DN_relabeled.xlsx"
signal, nok = importSignal(datafile)

X = signal
y = nok

#split data
#num_splits = int(np.sqrt(len(y)))
num_splits = 10
skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)


temp_acc = np.zeros((num_splits))
for l in range(0, hid_layers.size):


    for n in range(0, neurons.size):
        print('l:', hid_layers[l], ' n:', neurons[n])

        split_iteration = 0
        for train_index, test_index in skf.split(X, y):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            temp_acc[split_iteration] = mainKval(hid_layers[l], neurons[n], X_train, y_train, X_test, y_test)
            print(temp_acc[split_iteration])
            split_iteration += 1

        mean_err[l, n, 0] = np.mean(temp_acc)
        mean_err[l, n, 1] = np.max(temp_acc) - np.mean(temp_acc)

print(mean_err)

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
    plt.show()
    #fig.savefig('No. of hidden layers' + str(hid_layers[h]) + '.jpg', bbox_inches='tight', dpi=250)



