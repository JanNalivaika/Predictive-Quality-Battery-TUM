import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, MSELoss, L1Loss, CrossEntropyLoss, NLLLoss, Sigmoid
from torch.optim import Adam, RMSprop, SGD, NAdam, LBFGS
from torch import from_numpy, no_grad, argmax
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn import preprocessing  #
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def importSignal(datafile):
    df = pd.read_excel(io=datafile)
    signal1 = df.loc[:, 'Signal1_  1':'Signal1_112']
    ok_label = df.loc[:, 'not OK']
    signal1 = np.asarray(signal1)
    ok_label = np.asarray(ok_label)

    return signal1, ok_label

class NNClassifier(nn.Module):
    def __init__(self, num_hidden_layers, num_neurons):
        super().__init__()
        ##############
        ##############

        # define three linear layers##

        num_of_features_input = 112
        num_of_classes = 2
        self.num_hidden_layers = num_hidden_layers

        #self.num_of_neurons = num_neurons
        #num_of_neurons

        if num_hidden_layers == 0:
            self.fcOut = nn.Linear(num_of_features_input, num_of_classes)


        if num_hidden_layers >= 1:
            self.fc1 = nn.Linear(num_of_features_input, num_neurons)
            if num_hidden_layers >= 2:
                self.fc2 = nn.Linear(num_neurons, num_neurons)
                if num_hidden_layers >= 3:
                    self.fc3 = nn.Linear(num_neurons, num_neurons)
                    if num_hidden_layers >= 4:
                        self.fc4 = nn.Linear(num_neurons, num_neurons)
                        if num_hidden_layers == 5:
                            self.fc5 = nn.Linear(num_neurons, num_neurons)
                            if num_hidden_layers > 5:
                                print('number of hidden layers must be < 6')

            self.fcOut = nn.Linear(num_neurons, num_of_classes)

        ##############
        ##############

    def forward(self, x):
        ##############
        ##############

        # define the activation functions for the three previously define linear layers
        if self.num_hidden_layers >= 0:
            if self.num_hidden_layers >= 1:
                x = torch.tanh(self.fc1(x))
                if self.num_hidden_layers >= 2:
                    x = torch.tanh(self.fc2(x))
                    if self.num_hidden_layers >= 3:
                        x = torch.tanh(self.fc3(x))
                        if self.num_hidden_layers >= 4:
                            x = torch.tanh(self.fc4(x))
                            if self.num_hidden_layers == 5:
                                x = torch.tanh(self.fc5(x))
                                if self.num_hidden_layers > 5:
                                    print('number of hidden layers must be < 6')

            x = torch.sigmoid(self.fcOut(x))
        else:
            print('number of hidden layers must be >0')

        ##############
        ##############
        return x


def main(num_hidden_layers, num_neurons):
    torch.manual_seed(123)
    BATCH_SIZE = 128
    EPOCHS_ADAM = 12
    EPOCH_LBFGS = 0
    LEARNING_RATE = 0.001

    datafile = "../Data/S1.xlsx"
    signal, nok = importSignal(datafile)

    # histogramm of the labels
    # plt.hist(y, 2)
    # plt.show()

    X = signal
    y = nok


    ## split dataset into train, validation, test
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5)

    # normalize and transform the data
    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    ##############
    ##############

    # convert data to Tensor and initialize dataloader
    X_train_scaled, X_val_scaled, X_test_scaled = torch.from_numpy(X_train_scaled.astype(float)).float(), \
                                                  torch.from_numpy(X_val_scaled.astype(float)).float(), \
                                                  torch.from_numpy(X_test_scaled.astype(float)).float()

    y_train, y_val, y_test = torch.from_numpy(y_train.astype(float)).float(), \
                             torch.from_numpy(y_val.astype(float)).float(), \
                             torch.from_numpy(y_test.astype(float)).float()

    dl_train = DataLoader(TensorDataset(X_train_scaled, y_train), batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(TensorDataset(X_val_scaled, y_val), batch_size=BATCH_SIZE, shuffle=True)
    dl_test = DataLoader(TensorDataset(X_test_scaled, y_test), batch_size=BATCH_SIZE, shuffle=True)

    ##############
    ##############

    standard_model = NNClassifier(num_hidden_layers, num_neurons)  # create model from calss NNClassifier

    optimizer = Adam(standard_model.parameters(), lr=LEARNING_RATE)  # optimizer for finding optimal weights
    loss_fun = CrossEntropyLoss()  # define common loss function for classification

    for epoch in range(EPOCHS_ADAM):
        epoch_loss = 0.0
        for inputs, true_output in dl_train:
            predictions = None
            loss = None
            ##############
            ##############

            logits = standard_model(inputs)
            to = true_output.type(torch.LongTensor)
            loss = loss_fun(logits, to)

            epoch_loss += float(loss.detach())

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()



        with no_grad():
            standard_model.eval()
            train_acc = 0.0
            for inputs, true_output in dl_train:
                predictions = standard_model(inputs)

                train_acc += (argmax(predictions, -1) == true_output).sum()

        with no_grad():
            standard_model.eval()
            val_acc = 0.0
            for inputs, true_output in dl_val:
                predictions = standard_model(inputs)
                val_acc += (argmax(predictions, -1) == true_output).sum()
        standard_model.train()

        print(["Epoch: %d, Training accuracy: %3.4f, Validation accuracy: %3.4f" % (
            epoch + 1, train_acc * 100 / len(dl_train.dataset), val_acc * 100 / len(dl_val.dataset))])


    torch.save(standard_model.state_dict(), "my_classification_model.pt")


    with no_grad():
        standard_model.eval()
        epoch_loss = 0.0
        test_acc = 0
        for inputs, labels in dl_test:
            logits = standard_model(inputs)
            labels_to = labels.type(torch.LongTensor)
            loss = loss_fun(logits, labels_to)

            epoch_loss += float(loss.detach())
            test_acc += (argmax(logits, -1) == labels).sum()

        print(["Test accuracy: %3.4f" % (test_acc * 100 / len(dl_test.dataset))])
        test_acc_return = test_acc * 100 / len(dl_test.dataset)

    return test_acc_return


if __name__ == "__main__":
    main()
