import numpy as np
from NN_v2 import main
import matplotlib.pyplot as plt

"""
num_hidden_layers: maximale anzahl der betrachteten hidden layer
num_neurons: maximale anzahl der betrachteten knoten
step_neurons: schrittweite der betrachteten kotenzahl

Für alle kombinationen wird der Mittelwert der Test Accuracy und die Varianz gespeichert
"""

num_hidden_layers = 1
num_neurons = 10
step_neurons = 3
num_iterations = 2

hid_layers = np.arange(start=0, stop=num_hidden_layers+1, step = 1)
neurons = np.arange(start=1, stop=num_neurons+1, step=step_neurons)

x1 = hid_layers.size
x2 = neurons.size
x3 = 2
mean_err = np.zeros((x1, x2, x3))
print(neurons.size)

temp_acc = np.zeros((num_iterations))

for l in range(0, hid_layers.size):
    print('l:', l)

    for n in range(0, neurons.size):
        print('n:', n)

        for i in range(num_iterations):
            temp_acc[i] = main(hid_layers[l], neurons[n])
            print(temp_acc[i])

        mean_err[l, n, 0] = np.mean(temp_acc)
        mean_err[l, n, 1] = np.max(temp_acc) - np.mean(temp_acc)

print(mean_err)

for h in range(0, hid_layers.size):
    plt.title("No. of hidden layers: " + str(h))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of neurons in hidden layers")
    plt.ylim([0, 100])
    plt.scatter(neurons, mean_err[h, :, 0], s=20, marker='x')
    plt.errorbar(neurons, mean_err[h, :, 0], mean_err[h, :, 1], capsize=10,
             capthick=1, ls='none')
    plt.show()



