# -*- coding: utf-8 -*-
"""StockEX.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1duYS11KjumY5qfOkCzyHhFMZ5huSi0_a
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import numpy as np

"""## Data Preprocessing part

From this section, data preprocessing and analysis part starts here...
"""

data = pd.read_csv("./ITC.csv")
data.head()

from sklearn.preprocessing import MinMaxScaler

scale = MinMaxScaler(feature_range=(0,1))

data['Date'] = pd.to_datetime(data.Date, format="%Y-%m-%d")
data.index = data['Date']

plt.plot(data['Close'], label="Close price")

new_dat = pd.DataFrame(index=range(0, len(data)), columns=['Date', 'Close'])
for i in range(0, len(data)):
  new_dat['Date'][i] = data['Date'][i]
  new_dat['Close'][i] = data['Close'][i]
  
new_dat.index = new_dat.Date
new_dat.drop('Date', axis=1, inplace=True)

# Split
from sklearn.model_selection import train_test_split

X = []
Y = []
window_size=50
for i in range(0 , len(data) - window_size -1 , 1):
    first = data.iloc[i, 4]
    temp = []
    temp2 = []
    for j in range(window_size):
        temp.append((data.iloc[i + j, 8] - first) / first)
    # for j in range(week):
    temp2.append((data.iloc[i +window_size, 8] - first) / first)
    # X.append(np.array(stock.iloc[i:i+window_size,4]).reshape(50,1))
    # Y.append(np.array(stock.iloc[i+window_size,4]).reshape(1,1))
    # print(stock2.iloc[i:i+window_size,4])
    X.append(np.array(temp).reshape(50, 1))
    Y.append(np.array(temp2).reshape(1,1))

x_train, x_test, train_label, test_label = train_test_split(X, Y, test_size=0.2, shuffle=True)

train_X = np.array(x_train)
test_X = np.array(x_test)
train_label = np.array(train_label)
test_label = np.array(test_label)

train_X = train_X.reshape(train_X.shape[0],1,50,1)
test_X = test_X.reshape(test_X.shape[0],1,50,1)

"""## Training part
From this section, model creation, training is done...
"""

from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation, Flatten, Bidirectional, TimeDistributed, MaxPool2D
from keras.layers import GlobalAveragePooling2D

model = keras.Sequential()
model.add(TimeDistributed(Conv1D(128, kernel_size=1, activation='relu', input_shape=(None,50,1))))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(256, kernel_size=1, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(512, kernel_size=1, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(LSTM(200,return_sequences=True)))
model.add(Dropout(0.25))
model.add(Bidirectional(LSTM(200,return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='RMSprop', loss='mse')
model.fit(train_X, train_label, validation_data=(test_X,test_label), epochs=40,batch_size=64,shuffle =False)

"""## Testing
From this section, testing needs to be done... Needs with help of a mate
"""

predicted  = model.predict(test_X)
test_label = (test_label)
predicted = np.array(predicted[:,0]).reshape(-1,1)
len_t = len(train_X)
for j in range(len_t , len_t + len(test_X)):
    temp =data.iloc[j,8]
    test_label[j - len_t] = test_label[j - len_t] * temp + temp
    predicted[j - len_t] = predicted[j - len_t] * temp + temp
plt.plot(predicted, color = 'green', label = 'Predicted  Stock Price')
plt.title(' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(' Stock Price')
plt.legend()
plt.show()