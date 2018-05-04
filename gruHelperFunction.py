import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.python.keras import metrics
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import LeakyReLU

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

def batchGenerator(batchSize, sequenceLength, xData, xTrainScaled, yTrainScaled, yTrain):
    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        numXSignals = xData.shape[1]
        xShape = (batchSize, sequenceLength, numXSignals)
        xBatch = np.zeros(shape=xShape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        numYSignals = 1
        yShape = (batchSize, sequenceLength, numYSignals)
        yBatch = np.zeros(shape=yShape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batchSize):
            # Get a random start-index.
            # This points somewhere into the training-data.
            numTrain = len(yTrain)
            idx = np.random.randint(numTrain - sequenceLength)

            # Copy the sequences of data starting at this index.
            xBatch[i] = xTrainScaled[idx:idx+sequenceLength]
            yBatch[i] = yTrainScaled[idx:idx+sequenceLength]

        yield (xBatch, yBatch)

def modelFit(units, activationGRU, dropout, activationDense, learningRate, epochs, stepsPerEpoch, numXSignals, numYSignals, generator, validationData):
    model = Sequential()
    model.add(GRU(units=units,
              activation=activationGRU,
              return_sequences=True,
              input_shape=(None, numXSignals,)))
    model.add(Dropout(dropout))
    model.add(Dense(numYSignals, activation=activationDense))
    optimizer = RMSprop(lr=learningRate)
    model.compile(loss='mae',
              optimizer=optimizer,
              metrics = ['mse'])
    model.summary()
    pathCheckpoint = 'checkpoint.keras'
    callbackCheckpoint = ModelCheckpoint(filepath=pathCheckpoint,
                                        monitor='val_loss',
                                        verbose=1,
                                        save_weights_only=True,
                                        save_best_only=True)
    callbackEarlyStopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)
    callbackReduceLR = ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.1,
                                        min_lr=1e-4,
                                        patience=0,
                                        verbose=1)
    callbacks = [callbackEarlyStopping,
                callbackCheckpoint,
                callbackReduceLR]
    history = model.fit_generator(generator=generator,
                                epochs=epochs,
                                steps_per_epoch=stepsPerEpoch,
                                validation_data=validationData,
                                callbacks=callbacks)
    return model, history

def modelPredict(model, xTrainScaled, xTestScaled, xScaler, yScaler):
    x1 = np.expand_dims(xTrainScaled, axis=0)
    yPred1 = model.predict(x1)
    yPredRescaled1 = yScaler.inverse_transform(yPred1[0])
    x2 = np.expand_dims(xTestScaled, axis=0)
    yPred2 = model.predict(x2)
    yPredRescaled2 = yScaler.inverse_transform(yPred2[0])
    return yPredRescaled1, yPredRescaled2

def errorCalculation(trainMseArr, testMseArr, history):
    trainMseArr = np.append(trainMseArr, history.history['mean_squared_error'])
    testMseArr = np.append(testMseArr, history.history['val_mean_squared_error'])

    results = pd.DataFrame({'train MSE':trainMseArr, 'test MSE':testMseArr})
    return results

def plotHistory(history, error=True, one=True):
    if error == True:
        plt.plot(history.history['mean_squared_error'], color='blue')
        plt.plot(history.history['val_mean_squared_error'], color='orange')
        if one == True:
            plt.show()
        else:
            plt.pause(0.05)
    else:
        plt.plot(history.history['loss'], color='blue')
        plt.plot(history.history['val_loss'], color='orange')
        if one == True:
            plt.show()
        else:
            plt.pause(0.05)

def plotResults(rawData, shiftMonths, yTrain, yTest, yPredRescaled1, yPredRescaled2):
    t = np.arange(len(rawData['Price']) - shiftMonths)
    plt.figure(figsize=(25,8))
    plt.plot(t[:len(yTrain)], yTrain, label='true train')
    plt.plot(t[:len(yTrain)], yPredRescaled1, label='prediction train')
    plt.plot(t[len(yTrain):], yTest, label='true test')
    plt.plot(t[len(yTrain):], yPredRescaled2, label='prediction test')
    plt.ylim([0,20])
    plt.xlabel('Time Steps') # in number of months
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def accuracy(yPredRescaled2, yTest):
    accuracy = np.sqrt(mean_squared_error(yPredRescaled2, yTest))
    return accuracy
