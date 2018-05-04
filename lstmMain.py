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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau


from lstmHelperFunctions import *

"""
Model Parameters
"""
shiftMonths = 36

"""
Tuning Parameters
"""
units = 512
activationLSTM = 'tanh'
dropout = 0.2
activationDense = 'relu'
learningRate = 1e-4
epochs = 100
batchSize = 12

"""
Script
"""
rawData = parser(csv='toyData.csv')

xData, yData = shifter(rawData=rawData, shiftMonths=shiftMonths)
xTrain, xTest, yTrain, yTest = testTrainSplit(xData=xData, yData=yData, shiftMonths=shiftMonths)
xTrainScaled, xTestScaled, yTrainScaled, yTestScaled, xScaler, yScaler = scaler(xTrain=xTrain, xTest=xTest, yTrain=yTrain, yTest=yTest)

xTrainScaled, xTestScaled, yTrainScaled, yTestScaled = reshape(xTrainScaled=xTrainScaled, xTestScaled=xTestScaled, yTrainScaled=yTrainScaled, yTestScaled=yTestScaled)

# testVar = batchSize
# for i in range(len(testVar)):
    # print('Test variable:', testVar[i])
model, history = modelFit(units=units, activationLSTM=activationLSTM, xTrainScaled=xTrainScaled, yTrainScaled=yTrainScaled, dropout=dropout, activationDense=activationDense, learningRate=learningRate, epochs=epochs, batchSize=batchSize, xTestScaled=xTestScaled, yTestScaled=yTestScaled)

plotHistory(history=history, one=True)

yPredTestRescaled, yPredTrainRescaled = modelPredict(model=model, xTestScaled=xTestScaled, xTrainScaled=xTrainScaled, yScaler=yScaler)

plotResults(rawData=rawData, shiftMonths=shiftMonths, yTrain=yTrain, yPredTrainRescaled=yPredTrainRescaled, yTest=yTest, yPredTestRescaled=yPredTestRescaled)

accuracy = accuracy(yPredTestRescaled=yPredTestRescaled, yTest=yTest)
print(accuracy)
