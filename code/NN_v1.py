import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, MSELoss, L1Loss, CrossEntropyLoss, NLLLoss, Sigmoid
from torch.optim import Adam, RMSprop, SGD, NAdam
from torch import from_numpy, no_grad, argmax
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn import preprocessing  #
from sklearn.model_selection import train_test_split
from visualize_test import importData
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class NNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        ##############
        ##############

        # define three linear layers##

        num_of_features_input = 112
        num_of_classes = 2

        self.fc1 = nn.Linear(num_of_features_input, 100)
        self.fc2 = nn.Linear(100, 80)
        self.fc3 = nn.Linear(80, 60)
        self.fc4 = nn.Linear(60, 50)
        self.fc5 = nn.Linear(50, 30)
        self.fcOut = nn.Linear(30, num_of_classes)

        ##############
        ##############

    def forward(self, x):
        ##############
        ##############

        # define the activation functions for the three previously define linear layers

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.sigmoid(self.fcOut(x))

        ##############
        ##############
        return x


def main():
    torch.manual_seed(123)
    BATCH_SIZE = 64
    EPOCHS = 64
    LEARNING_RATE = 0.0005

    datafile = "../Data/S1.xlsx"
    data = importData(datafile)

    signal = data[:, 3:]  # rows = samples, col. = features

    nok = data[:, 0]  # 1 = not ok, 0 = ok
    WD_40 = data[:, 1]  # 1 = WD-40, 0 = no WD-40
    Gleitmo = data[:, 2]  # 1 = Gleitmo, 0 = no gleitmo
    lubericant = WD_40 + Gleitmo
    ### determine input ###
    X = signal
    # combined = np.concatenate((signal1, signal2, signal1_DN), axis=1)
    y = nok
    # y = WD_40
    # y = Gleitmo
    # y = lubericant

    # BATCH_SIZE = int(data.shape[0]/10) #size of the batches the data is split up into
    # print('Data Size:', data.shape[0], 'Batch Size:', BATCH_SIZE)

    # histogramm of the labels
    # plt.hist(y, 2)
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, )  # random_state=42)

    scaler = preprocessing.StandardScaler()  # normalize the data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ##############
    ##############

    # convert data to Tensor and initialize dataloader
    X_train_scaled, X_test_scaled = torch.from_numpy(X_train_scaled.astype(float)).float(), torch.from_numpy(
        X_test_scaled.astype(float)).float()
    y_train, y_test = torch.from_numpy(y_train.astype(float)).float(), torch.from_numpy(y_test.astype(float)).float()

    dl_train = DataLoader(TensorDataset(X_train_scaled, y_train), batch_size=BATCH_SIZE, shuffle=True)
    dl_test = DataLoader(TensorDataset(X_test_scaled, y_test), batch_size=BATCH_SIZE, shuffle=True)

    ##############
    ##############

    standard_model = NNClassifier()  # create model from calss NNClassifier

    optimizer = Adam(standard_model.parameters(), lr=LEARNING_RATE)  # optimizer for finding optimal weights
    loss_fun = CrossEntropyLoss()  # define common loss function for classification

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for inputs, true_output in dl_train:
            predictions = None
            loss = None
            ##############
            ##############

            logits = standard_model(inputs)
            to = true_output.type(torch.LongTensor)
            loss = loss_fun(logits, to)

            ##############
            ##############
            epoch_loss += float(loss.detach())

            ##############
            ##############

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            ##############
            ##############
            pass

        with no_grad():
            standard_model.eval()
            acc = 0.0
            for inputs, true_output in dl_train:
                predictions = standard_model(inputs)

                acc += (argmax(predictions, -1) == true_output).sum()

        with no_grad():
            standard_model.eval()
            val_acc = 0.0
            for inputs, true_output in dl_test:
                predictions = standard_model(inputs)
                val_acc += (argmax(predictions, -1) == true_output).sum()
        standard_model.train()

        print(["Epoch: %d, Training accuracy: %3.4f, Validation accuracy: %3.4f" % (
            epoch + 1, acc * 100 / len(dl_train.dataset), val_acc * 100 / len(dl_test.dataset))])

    torch.save(standard_model.state_dict(), "my_classification_model.pt")


if __name__ == "__main__":
    main()
