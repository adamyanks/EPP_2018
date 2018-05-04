import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau


def parser(csv):
    rawData = pd.read_csv(csv, sep=',')
    return rawData

def shifter(rawData, shiftMonths):
    dfTargets = rawData.shift(-shiftMonths)
    xData = rawData.values[0:-shiftMonths]
    yData = dfTargets.values[:-shiftMonths, 0]
    print(type(xData))
    print('Shape:', xData.shape)
    print(type(yData))
    print('Shape:', yData.shape)
    return xData, yData

def testTrainSplit(xData, yData, shiftMonths):
    xTrain = xData[:(len(yData)-shiftMonths)]
    xTest = xData[-shiftMonths:]
    yTrain = yData[:(len(yData)-shiftMonths)]
    yTest = yData[-shiftMonths:]
    print ('Number of samples in training data:',len(xTrain))
    print ('Number of samples in test data:',len(xTest))
    return xTrain, xTest, yTrain, yTest

def scaler(xTrain, xTest, yTrain, yTest):
    xScaler = MinMaxScaler()
    xTrainScaled = xScaler.fit_transform(xTrain)
    xTestScaled = xScaler.transform(xTest)
    yTrain = yTrain.reshape(-1, 1)
    yTest = yTest.reshape(-1, 1)
    yScaler = MinMaxScaler(feature_range=(2, 4))
    yTrainScaled = yScaler.fit_transform(yTrain)
    yTestScaled = yScaler.transform(yTest)
    print("Min:", np.min(xTrainScaled))
    print("Max:", np.max(xTrainScaled))
    print('Input shape:', xTrainScaled.shape)
    print('Output shape:', yTrainScaled.shape)
    return xTrainScaled, xTestScaled, yTrainScaled, yTestScaled, xScaler, yScaler

def difference(dataset):
    shape = dataset.shape
    if len(shape) == 1:
        datasetDiff = np.zeros([shape[0], 1])
        for i in range(shape[0] - 1):
            diffVal = dataset[i+1] - dataset[i]
            datasetDiff[i] = diffVal
    elif len(shape) == 2:
        datasetDiff = np.zeros([shape[0], shape[1]])
        for c in range(shape[1]):
            for i in range(shape[0] - 1):
                diffVal = dataset[i+1, c] - dataset[i, c]
                datasetDiff[i, c] = diffVal
    return datasetDiff

def reshape(xTrainScaled, xTestScaled, yTrainScaled, yTestScaled):
    # reshape input to be 3D [samples, timesteps, features]
    xTrainScaled = xTrainScaled.reshape((xTrainScaled.shape[0], 1, xTrainScaled.shape[1]))
    xTestScaled = xTestScaled.reshape((xTestScaled.shape[0], 1, xTestScaled.shape[1]))
    print(xTrainScaled.shape, yTrainScaled.shape, xTestScaled.shape, yTestScaled.shape)
    return xTrainScaled, xTestScaled, yTrainScaled, yTestScaled

def modelFit(units, activationLSTM, xTrainScaled, yTrainScaled, dropout, activationDense, learningRate, epochs, batchSize, xTestScaled, yTestScaled):
    # design network
    model = Sequential()
    model.add(LSTM(units=units, activation=activationLSTM, input_shape=(xTrainScaled.shape[1], xTrainScaled.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation=activationDense))
    adam = optimizers.Adam(lr=learningRate)
    model.compile(loss='mae', optimizer=adam)
    model.summary()
    # pathCheckpoint = 'checkpoint.keras'
    # callbackCheckpoint = ModelCheckpoint(filepath=pathCheckpoint,
                                        # monitor='val_loss',
                                        # verbose=1,
                                        # save_weights_only=True,
                                        # save_best_only=True)
    # callbackEarlyStopping = EarlyStopping(monitor='val_loss',
                                        # patience=5, verbose=1)
    # callbackReduceLR = ReduceLROnPlateau(monitor='val_loss',
                                        # factor=0.1,
                                        # min_lr=1e-4,
                                        # patience=0,
                                        # verbose=1)
    # callbacks = [callbackEarlyStopping,
                # callbackCheckpoint,
                # callbackReduceLR]

    history = model.fit(xTrainScaled, yTrainScaled, epochs=epochs, batch_size=batchSize, validation_data=(xTestScaled, yTestScaled), verbose=1, shuffle=False)
    return model, history

# def plotHistory(history):
    # plot history
    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend()
    # plt.show()

def plotHistory(history, one=True):
    plt.plot(history.history['loss'], color='blue')
    plt.plot(history.history['val_loss'], color='orange')
    if one == True:
        plt.show()
    else:
        plt.pause(0.05)

def modelPredict(model, xTestScaled, xTrainScaled, yScaler):
    # make a prediction
    yPredTest = model.predict(xTestScaled)
    yPredTrain = model.predict(xTrainScaled)
    # invert scaling for test
    xTestScaled = xTestScaled.reshape((xTestScaled.shape[0], xTestScaled.shape[2]))
    yPredTestRescaled = concatenate((yPredTest, xTestScaled[:, 1:]), axis=1)
    yPredTestRescaled = yScaler.inverse_transform(yPredTestRescaled)
    yPredTestRescaled = yPredTestRescaled[:,0]
    # invert scaling for training
    xTrainScaled = xTrainScaled.reshape((xTrainScaled.shape[0], xTrainScaled.shape[2]))
    yPredTrainRescaled = concatenate((yPredTrain, xTrainScaled[:, 1:]), axis=1)
    yPredTrainRescaled = yScaler.inverse_transform(yPredTrainRescaled)
    yPredTrainRescaled = yPredTrainRescaled[:,0]
    return yPredTestRescaled, yPredTrainRescaled

def plotResults(rawData, shiftMonths, yTrain, yPredTrainRescaled, yTest, yPredTestRescaled):
    t = np.arange(len(rawData['Price']) - shiftMonths)
    # Make the plotting-canvas bigger.
    plt.figure(figsize=(25,8))

    # Plot and compare the two signals.
    plt.plot(t[:len(yTrain)], yTrain, label='true train')
    plt.plot(t[:len(yTrain)], yPredTrainRescaled, label='prediction train')
    plt.plot(t[len(yTrain):], yTest, label='true test')
    plt.plot(t[len(yTrain):], yPredTestRescaled, label='prediction test')

    # Plot labels etc.
    plt.xlabel('Time Steps') # in number of months
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def accuracy(yPredTestRescaled, yTest):
    accuracy = np.sqrt(mean_squared_error(yPredTestRescaled, yTest))
    return accuracy
