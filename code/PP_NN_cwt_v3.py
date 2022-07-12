import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import random
from sklearn.model_selection import train_test_split
from visualize_test import importData
import os
import pywt
import keras
from keras.utils import to_categorical
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import History

def cwt(signal):

    rg_scales = 112
    scales = range(1, rg_scales)
    wavelet = 'morl'

    x_shape = tuple([len(signal), rg_scales-1, 111])

    x_train = np.empty(x_shape)

    for x in range(len(signal)):
        coefs, freqs = pywt.cwt(signal[x], scales, wavelet)
        # x_train[x] = np.transpose(coefs)
        x_train[x] = coefs

    #print(x_train.shape)
    return x_train

if __name__ == "__main__":
    datafile = "../Data/All/All_Data.xlsx"
    data = importData(datafile)
    # signal = data[:, 6:117]
    signal = data[:, 118:229]
    # signal = data[:, 230:341]

    x_train = cwt(signal)
    x_train = x_train.astype('float32')

    y_train = to_categorical(data[:, 0], 2)
    
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25)

    model = Sequential([
        Flatten(input_shape=(111,111)),
        Dense(100, activation='relu'),
        Dense(2, activation='softmax')
    ])

    
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])
    
    
    history = model.fit(x_train, y_train, epochs=10, validation_split=0.25, shuffle=True, verbose=1)
    
    # print(history.history.keys())
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label = 'test')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('accuracy across epochs')
    plt.show()

    test_score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))

