import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM, Dropout
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from pandas import DataFrame

dataset= pd.read_csv("dataset.csv", skiprows = 3)
cols = list(dataset)[5:11]
dataset = dataset[cols].astype(float)
dataset = np.array(dataset)


class Adaptive_Standard( layers.Layer ):
    def __init__(self , dim , features , **kwargs):
        super( Adaptive_Standard , self ).__init__( **kwargs )
        self.dim = dim
        self.features = features
        self.eps = 1e-8

        self.shifting_layer = layers.Dense( features , use_bias=False , kernel_initializer="identity" )
        self.scaling_layer = layers.Dense( features , use_bias=False , kernel_initializer="identity" )

        self.transpose = layers.Permute( (2 , 1) )
        self.reshape = layers.Reshape( (dim , features) )

    def call(self , inputs):
        ## Adaptive Shifting
        inputs = self.transpose( inputs )
        avg = backend.mean( inputs , axis=2 )
        shift_operator = self.shifting_layer( avg )
        shift_operator = backend.reshape( shift_operator , (-1 , self.features , 1) )
        inputs -= shift_operator

        ## Adatiptive Scaling
        std = backend.mean( inputs ** 2 , axis=2 )
        std = backend.sqrt( std + self.eps )
        scale_operator = self.scaling_layer( std )
        fn = lambda elem: backend.switch( backend.less_equal( elem , 1.0 ) , backend.ones_like( elem ) , elem )
        scale_operator = backend.map_fn( fn , scale_operator )
        scale_operator = backend.reshape( scale_operator , (-1 , self.features , 1) )
        inputs /= scale_operator

        return inputs

## For scaling the targets
def normalise_output(train_y):
  scaler = MinMaxScaler()
  scaler = scaler.fit(train_y)
  train_y = scaler.transform(train_y)
  return train_y

model = Sequential()
model.add(Adaptive_Standard(12,6))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(Dense(8, kernel_initializer='normal',activation='relu'))
model.add(Dense(1, kernel_initializer='normal',activation='linear'))

## one split consist of data of one month
for split in range(1,12):
  train_x , train_y = [], []
  test_x , test_y = [], []
  x = 12
  y = 288*30
  for i in range(y*split-x):
    train_x.append(dataset[i : i+x])
    train_y.append(dataset[i+x : i+x+1, 2])

  for i in range(y*split-x, y*(split+1)-x):
    test_x.append(dataset[i : i+x])
    test_y.append(dataset[i+x : i+x+1, 2])
  train_x , train_y = np.array(train_x).astype("float32"), np.array(train_y).astype("float32")
  test_x , test_y = np.array(test_x).astype("float32"), np.array(test_y).astype("float32")
  train_y = normalise_output(train_y)
  test_y = normalise_output(test_y)
  model.compile(optimizer="adam", loss="mse")
  model.build(train_x.shape)
  model.summary()
  history = model.fit(train_x, train_y, epochs = 10,validation_data=(test_x, test_y) ,batch_size= 100, verbose = 1)

from matplotlib import pyplot
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()

predictions = model.predict(test_x)
y = []
for i in range(predictions.shape[0]):
  y.append(predictions[i][5])
y = np.array(y)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
mse = mean_squared_error(y, test_y)
print('Test MSE: %.3f' % mse)
rmse = sqrt(mean_squared_error(y , test_y))
print('Test RMSE: %.3f' % rmse)
mae = mean_absolute_error(y, test_y)
print('Test MAE: %.3f' % mae)
