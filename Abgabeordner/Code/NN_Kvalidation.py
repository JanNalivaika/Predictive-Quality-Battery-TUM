import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch import no_grad, argmax
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing

"""
Das Neuronale Netz in dieser Datei wird der Übersichtlichkeit halber mit Hilfe der Datei "call_NN.py" aufgerufen.
Es ist ohne diese Datei nur mit Modifikationen lauffähig

"""


class NNClassifier(nn.Module):
    def __init__(self, dim_input, num_hidden_layers, num_neurons):
        super().__init__()
        ##############
        ##############

        #number of layers is selected in call_NN##

        num_of_features_input = dim_input
        num_of_classes = 2
        self.num_hidden_layers = num_hidden_layers


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
                        if num_hidden_layers >= 5:
                            self.fc5 = nn.Linear(num_neurons, num_neurons)
                            if num_hidden_layers >= 6:
                                self.fc6 = nn.Linear(num_neurons, num_neurons)
                                if num_hidden_layers >= 7:
                                    self.fc7 = nn.Linear(num_neurons, num_neurons)
                                    if num_hidden_layers > 8:
                                        print('number of hidden layers must be < 8')

            self.fcOut = nn.Linear(num_neurons, num_of_classes)

        ##############
        ##############

    def forward(self, x):
        ##############
        ##############

        if self.num_hidden_layers >= 0:
            if self.num_hidden_layers >= 1:
                x = torch.relu(self.fc1(x))
                if self.num_hidden_layers >= 2:
                    x = torch.relu(self.fc2(x))
                    if self.num_hidden_layers >= 3:
                        x = torch.relu(self.fc3(x))
                        if self.num_hidden_layers >= 4:
                            x = torch.relu(self.fc4(x))
                            if self.num_hidden_layers >= 5:
                                x = torch.relu(self.fc5(x))
                                if self.num_hidden_layers >= 6:
                                    x = torch.relu(self.fc6(x))
                                    if self.num_hidden_layers >= 7:
                                        x = torch.relu(self.fc7(x))
                                        if self.num_hidden_layers > 8:
                                            print('number of hidden layers must be < 8')

            x = torch.sigmoid(self.fcOut(x))
        else:
            print('number of hidden layers must be > 0')

        ##############
        ##############
        return x


def main(dim_input, num_hidden_layers, num_neurons, X_train, y_train, X_val, y_val):
    torch.manual_seed(123)
    BATCH_SIZE = 128
    EPOCHS_ADAM = 22

    learning_rate = 0.001

    # normalize and transform the data
    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    ##############
    ##############

    # convert data to Tensor and initialize dataloader
    X_train_scaled, X_val_scaled = torch.from_numpy(X_train_scaled.astype(float)).float(), \
                                                  torch.from_numpy(X_val_scaled.astype(float)).float()

    y_train, y_val = torch.from_numpy(y_train.astype(float)).float(), \
                             torch.from_numpy(y_val.astype(float)).float()

    dl_train = DataLoader(TensorDataset(X_train_scaled, y_train), batch_size=BATCH_SIZE, shuffle=False)
    dl_val = DataLoader(TensorDataset(X_val_scaled, y_val), batch_size=BATCH_SIZE, shuffle=False)

    ##############
    ##############
    # create model
    model = NNClassifier(dim_input, num_hidden_layers, num_neurons)
    # set optimizer and common loss function for classification
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fun = CrossEntropyLoss()

    #train model and evaluate performance on train dataset
    for epoch in range(EPOCHS_ADAM):
        epoch_loss = 0.0
        for inputs, true_output in dl_train:
            predictions = None
            loss = None
            ##############
            ##############

            logits = model(inputs)

            to = true_output.type(torch.LongTensor)
            loss = loss_fun(logits, to)

            #optimizer parameters/perform backward step
            epoch_loss += float(loss.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with no_grad():
            model.eval()
            train_acc = 0.0
            for inputs, true_output in dl_train:

                predictions = model(inputs)
                train_acc += (argmax(predictions, -1) == true_output).sum()

        model.train()

        #optionally display training accuracy
        #print(["Epoch: %d, Training accuracy: %3.4f" % (
            #epoch + 1, train_acc * 100 / len(dl_train.dataset))])

    torch.save(model.state_dict(), "my_classification_model.pt")

    # evalute performance on test/validation data
    with no_grad():
        model.eval()
        epoch_loss = 0.0
        val_acc = 0
        false_positive = 0
        false_negative = 0

        for inputs, labels in dl_val:
            logits = model(inputs)

            epoch_loss += float(loss.detach())
            val_acc += (argmax(logits, -1) == labels).sum()

            # false positives und negatives werden hier "sinnvoll" berechnet, d.h. false positve = fälschlicherweise als
            # ok klassifiziert. Beim label 'lubricant'  müssen diese werte umgedreht betrachtet werden, d.h.
            # die false positives entsprechen dann den false negatives
            false_negative += (argmax(logits, -1) > labels).sum()
            false_positive += (argmax(logits, -1) < labels).sum()


        print(["Test accuracy: %3.4f" % (val_acc * 100 / len(dl_val.dataset))])
        print(["Test false negatives: %3.4f" % (false_negative * 100 / len(dl_val.dataset))])
        print(["Test false positives: %3.4f" % (false_positive * 100 / len(dl_val.dataset))])
        val_acc_return = val_acc * 100 / len(dl_val.dataset)
        false_negative_return = false_negative * 100 / len(dl_val.dataset)
        false_positive_return = false_positive * 100 / len(dl_val.dataset)

    return val_acc_return, false_negative_return, false_positive_return


if __name__ == "__main__":
    main()
