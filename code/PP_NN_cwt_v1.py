import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, MSELoss, L1Loss, CrossEntropyLoss, NLLLoss, Sigmoid
from torch.optim import Adam, RMSprop, SGD
from torch import from_numpy, no_grad, argmax, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn import preprocessing#
from sklearn.model_selection import train_test_split
from visualize_test import importData
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(input_size, 256), # hidden layer, nn.Linear(input_size, output_size)
                                        nn.BatchNorm1d(256),        # normalization layer
                                        nn.ReLU(),                  # activation function ReLU
                                        nn.Linear(256, 2),          # output layer
                                        nn.BatchNorm1d(2),          # normalization layer
                                        nn.LogSoftmax(dim=1))       # squeeze outputs to add up to 1, classification
        
    def forward(self, x):
        # make sure the input tensor is flattened in order to fit input_size
        x = torch.flatten(x)
        x = self.classifier(x) # sequentially push input x throught the neural net
        return x


def main():
    #torch.manual_seed(123)
    SCALE_DIM = 127
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.001

    model = NeuralNet(111 * SCALE_DIM)
    criterion = nn.NLLLoss() # negative log likelihood loss for classification
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE) # standard stochastic gradient descnet algorithm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # check whether GPU is available

    model.to(device)

    Y_datafile = "../Data/All/All_Data.xlsx"
    Y_data = importData(Y_datafile)
    Y = Y_data[:, 0]

    X_datafile = "../Data/Preprocessed/Signal1_cwt.npy"
    X_data = importData(X_datafile)
    X = X_data

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    dl_train = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(Y_train)), shuffle=True)
    dl_test = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(Y_test)), shuffle=True)


    for epoch in range(EPOCHS):
        running_loss = 0
        test_loss = 0
        for inputs, true_output in dl_train:

            inputs = inputs.to(device)
            true_output = true_output.to(device)
            log_probs = model.forward(inputs)
            loss = criterion(log_probs, true_output)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()


        with no_grad():
            for inputs, true_output in dl_test:
                inputs, true_output = inputs.to(device), true_output.to(device) # transfer test data to GPU if available
                logps = model.forward(inputs) # make predictions
                batch_loss = criterion(logps, true_output) # calculate current batch's loss
                test_loss += batch_loss.item() # add to cumulative batch loss
            model.train()

    print('Epoch %i / %i ... Training loss: %.5f ... Test loss: %.5f'%(epoch+1, \
                                                                       EPOCHS, \
                                                                       running_loss/len(dl_train),\
                                                                       test_loss/len(dl_test)))


if __name__ == "__main__":
    main()
