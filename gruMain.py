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
from gruHelperFunctions import parser, shifter, testTrainSplit, scaler, batchGenerator, modelFit, modelPredict, errorCalculation, plotHistory, plotResults, accuracy

"""
Model Parameters
"""
shiftMonths = 36

"""
Tuning Parameters
"""
batchSize = 17
sequenceLength = 10

units = 1000
activationGRU = 'tanh'
dropout = .4
activationDense = 'relu'
learningRate = np.logspace(0, -4, 5)
epochs = 50
stepsPerEpoch = 36

"""
Script
"""

rawData = parser(csv='toyData.csv')

xData, yData = shifter(rawData=rawData, shiftMonths=shiftMonths)
xTrain, xTest, yTrain, yTest = testTrainSplit(xData=xData, yData=yData, shiftMonths=shiftMonths)
xTrainScaled, xTestScaled, yTrainScaled, yTestScaled, xScaler, yScaler = scaler(xTrain=xTrain, xTest=xTest, yTrain=yTrain, yTest=yTest)

numXSignals = xData.shape[1]
numYSignals = 1

trainMseArr = []
testMseArr = []

generator = batchGenerator(batchSize=batchSize, sequenceLength=sequenceLength, xData=xData, xTrainScaled=xTrainScaled, yTrainScaled=yTrainScaled, yTrain=yTrain)
xBatch, yBatch = next(generator)
validationData = (np.expand_dims(xTestScaled, axis=0), np.expand_dims(yTestScaled, axis=0))

# testVar = learningRate
# for i in range(len(testVar)):
    # print('Test variable:', testVar[i])
model, history = modelFit(units=units, activationGRU=activationGRU, dropout=dropout, activationDense=activationDense, learningRate=learningRate[i], epochs=epochs, stepsPerEpoch=stepsPerEpoch, numXSignals=numXSignals, numYSignals=numYSignals, generator=generator, validationData=validationData)

results = errorCalculation(trainMseArr=trainMseArr, testMseArr=testMseArr, history=history)
print(results)

plotHistory(history=history, error=True, one=True)

# plt.show()

yPredRescaled1, yPredRescaled2 = modelPredict(model=model, xTrainScaled=xTrainScaled, xTestScaled=xTestScaled, xScaler=xScaler, yScaler=yScaler)

plotResults(rawData=rawData, shiftMonths=shiftMonths, yTrain=yTrain, yTest=yTest, yPredRescaled1=yPredRescaled1, yPredRescaled2=yPredRescaled2)

accuracy = accuracy(yPredRescaled2, yTest)
print('RMSE:', accuracy)
