from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import math

np.random.seed(42)
df = pd.read_csv("YHOO.csv", usecols=[2], engine='python', header=1)
ds = df.values.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
ds = scaler.fit_transform(ds)

len_ds = len(ds)
train_size = int(len_ds * 2 / 3)
test_size = len_ds - train_size
train_data, test_data = ds[:train_size, :], ds[train_size:, :]


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

lb = 1
trainX, trainY = create_dataset(train_data, lb)
testX, testY = create_dataset(test_data, lb)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_dim=lb))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
trainScore = math.sqrt(trainScore)
trainScore = scaler.inverse_transform(np.array([[trainScore]]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = model.evaluate(testX, testY, verbose=0)
testScore = math.sqrt(testScore)
testScore = scaler.inverse_transform(np.array([[testScore]]))
print('Test Score: %.2f RMSE' % (testScore))

# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# shift train predictions for plotting
trainPredictPlot = np.empty_like(ds)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[lb:len(trainPredict)+lb, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(ds)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(lb*2)+1:len(ds)-1, :] = testPredict

fp = open("test_data.txt", "w")
for ii in range(len(testX)):
	fp.write("%f\t%f\n" % (testY[ii], testPredict[ii]))
fp.close()

fp = open("train_data.txt", "w")
for ii in range(len(trainX)):
	fp.write("%f\t%f\n" % (trainY[ii], trainPredict[ii]))
fp.close()

show_plot = False
if show_plot:
    import matplotlib as mp
    import matplotlib.pyplot as plt

    mp.style.use('ggplot')
    plt.plot(ds)
    plt.show()

    # plot baseline and predictions
    plt.plot(ds)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()