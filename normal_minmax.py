from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt


## loading the NREL wind feature dataset
dataset= pd.read_csv("dataset.csv", skiprows=3)
cols = list(dataset)[5:11]
dataset = dataset[cols].astype(float)

## Normalising dataset using MinMaxScalar
scaler = MinMaxScalar()
scaler = scaler.fit(dataset)
dataset = scaler.transform(dataset)

## Processsing dataset into suitable format
dataset = np.array(dataset)
train_x , train_y = [], []
test_x , test_y = [], []

x = 12         # x rows of data in an hour
y = 288*30     # y rows of data in a month
for i in range(y*6-x):
    train_x.append(dataset[i:i + x])
    train_y.append(dataset[i + x:i + x + 1, 2])

for i in range(y*6-x, y*11-x):
    test_x.append(dataset[i:i + x])
    test_y.append(dataset[i + x:i + x + 1, 2])
train_x , train_y = np.array(train_x).astype("float32"), np.array(train_y).astype("float32")
test_x , test_y = np.array(test_x).astype("float32"), np.array(test_y).astype("float32")


## Trining dataset using Sequential model, it is a LSTM regression model
model = Sequential()
model.add(LSTM(256, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
model.add(Dense(8, kernel_initializer='normal',activation='relu'))
model.add(Dense(6, kernel_initializer='normal',activation='relu'))
model.add(Dense(1, kernel_initializer='normal',activation='linear'))
model.compile(optimizer="adam", loss="mse")
model.summary()

history = model.fit(train_x, train_y, epochs = 20, validation_data=(test_x, test_y) ,batch_size= 100, verbose = 1)

pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()


prediction = model.predict(test_x)
y = []
for i in range(prediction.shape[0]):
  y.append(prediction[i][0])

mse = mean_squared_error(y, test_y)
print('Test MSE: %.3f' % mse)
rmse = sqrt(mean_squared_error(y , test_y))
print('Test RMSE: %.3f' % rmse)
mae = mean_absolute_error(y, test_y)
print('Test MAE: %.3f' % mae)
