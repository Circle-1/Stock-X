# %% [markdown]
# # Stock Market Analysis using CNN-LSTM model
# This project is about analysis of Stock Market and providing suggestions and predictions to the stockholders. For this, we used CNN-LSTM approach to create a blank model, then use it to train on stock market data. Further implementation is discussed below...

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-12T07:15:22.948041Z","iopub.execute_input":"2021-11-12T07:15:22.948626Z","iopub.status.idle":"2021-11-12T07:15:22.954283Z","shell.execute_reply.started":"2021-11-12T07:15:22.948590Z","shell.execute_reply":"2021-11-12T07:15:22.953421Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# ## Data Preprocessing and Analysis

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-12T07:15:22.956025Z","iopub.execute_input":"2021-11-12T07:15:22.956411Z","iopub.status.idle":"2021-11-12T07:15:22.967962Z","shell.execute_reply.started":"2021-11-12T07:15:22.956374Z","shell.execute_reply":"2021-11-12T07:15:22.967073Z"}}
import math
import seaborn as sns
import datetime as dt
from datetime import datetime    
sns.set_style("whitegrid")
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use("ggplot")

# %% [markdown]
# First we'd read the CSV file and then drop the null columns. Then we'd check the columns (some not all)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-12T07:15:22.970618Z","iopub.execute_input":"2021-11-12T07:15:22.971041Z","iopub.status.idle":"2021-11-12T07:15:23.007962Z","shell.execute_reply.started":"2021-11-12T07:15:22.970976Z","shell.execute_reply":"2021-11-12T07:15:23.007019Z"}}
# For data preprocessing and analysis part
data = pd.read_csv('../input/price-volume-data-for-all-us-stocks-etfs/Stocks/abe.us.txt')
#data = pd.read_csv('../input/nifty50-stock-market-data/COALINDIA.csv')
#data = pd.read_csv('../input/stock-market-data/stock_market_data/nasdaq/csv/ABCO.csv')
# Any CSV or TXT file can be added here....
data.dropna(inplace=True)
data.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-12T07:15:23.009978Z","iopub.execute_input":"2021-11-12T07:15:23.010378Z","iopub.status.idle":"2021-11-12T07:15:23.030373Z","shell.execute_reply.started":"2021-11-12T07:15:23.010320Z","shell.execute_reply":"2021-11-12T07:15:23.028450Z"}}
data.info()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-12T07:15:23.031853Z","iopub.execute_input":"2021-11-12T07:15:23.032151Z","iopub.status.idle":"2021-11-12T07:15:23.064819Z","shell.execute_reply.started":"2021-11-12T07:15:23.032116Z","shell.execute_reply":"2021-11-12T07:15:23.063958Z"}}
data.describe()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-12T07:15:23.066362Z","iopub.execute_input":"2021-11-12T07:15:23.066652Z","iopub.status.idle":"2021-11-12T07:15:23.077345Z","shell.execute_reply.started":"2021-11-12T07:15:23.066616Z","shell.execute_reply":"2021-11-12T07:15:23.076359Z"}}
data.isnull().sum()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-12T07:15:23.079443Z","iopub.execute_input":"2021-11-12T07:15:23.079964Z","iopub.status.idle":"2021-11-12T07:15:23.102727Z","shell.execute_reply.started":"2021-11-12T07:15:23.079927Z","shell.execute_reply":"2021-11-12T07:15:23.102030Z"}}
data.reset_index(drop=True, inplace=True)
data.fillna(data.mean(), inplace=True)
data.head()

# %% [markdown]
# After that, we'll visualize the data for understanding, this is shown below...

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-12T07:15:23.104178Z","iopub.execute_input":"2021-11-12T07:15:23.104455Z","iopub.status.idle":"2021-11-12T07:15:23.935768Z","shell.execute_reply.started":"2021-11-12T07:15:23.104419Z","shell.execute_reply":"2021-11-12T07:15:23.935045Z"}}
cols_plot = ['Open', 'Close', 'High','Low']
axes = data[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Daily trade')

# %% [markdown]
# Then we'd print the data after making changes and dropping null data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-12T07:15:23.937329Z","iopub.execute_input":"2021-11-12T07:15:23.937848Z","iopub.status.idle":"2021-11-12T07:15:24.238850Z","shell.execute_reply.started":"2021-11-12T07:15:23.937809Z","shell.execute_reply":"2021-11-12T07:15:24.238182Z"}}
plt.plot(data['Close'], label="Close price")
plt.xlabel("Timestamp")
plt.ylabel("Closing price")

df = data.drop('Date', axis=1)
print(df)

# %% [markdown]
# The data has been analysed but it must be converted into data of shape [100,1] to make it easier for CNN to train on... Else it won't select necessary features and the model will fail

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-12T07:15:24.241214Z","iopub.execute_input":"2021-11-12T07:15:24.241503Z","iopub.status.idle":"2021-11-12T07:15:33.628595Z","shell.execute_reply.started":"2021-11-12T07:15:24.241451Z","shell.execute_reply":"2021-11-12T07:15:33.627853Z"}}
from sklearn.model_selection import train_test_split

X = []
Y = []
window_size=100
for i in range(1 , len(df) - window_size -1 , 1):
    first = df.iloc[i,3]
    temp = []
    temp2 = []
    for j in range(window_size):
        temp.append((df.iloc[i + j, 3] - first) / first)
    temp2.append((df.iloc[i + window_size, 3] - first) / first)
    X.append(np.array(temp).reshape(100, 1))
    Y.append(np.array(temp2).reshape(1, 1))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

train_X = np.array(x_train)
test_X = np.array(x_test)
train_Y = np.array(y_train)
test_Y = np.array(y_test)

train_X = train_X.reshape(train_X.shape[0],1,100,1)
test_X = test_X.reshape(test_X.shape[0],1,100,1)

print(len(train_X))
print(len(test_X))

# %% [markdown]
# ## Training part

# %% [markdown]
# This part has 2 subparts: CNN and LSTM
# 
# For CNN, the layers are created with sizes 64,128,64. In every layer, TimeDistributed function is added to track the features with respect to time. In between them, Pooling layers are added.
# 
# Then a dense layer of 5 neurons with L1 Kernel regularizer is added
# 
# After that, it's passed to Bi-LSTM layers

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-12T07:24:06.575812Z","iopub.execute_input":"2021-11-12T07:24:06.576090Z","iopub.status.idle":"2021-11-12T07:24:41.302363Z","shell.execute_reply.started":"2021-11-12T07:24:06.576060Z","shell.execute_reply":"2021-11-12T07:24:41.301634Z"}}
# For creating model and training
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, TimeDistributed
from tensorflow.keras.layers import MaxPooling1D, Flatten
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.metrics import RootMeanSquaredError

model = tf.keras.Sequential()

# Creating the Neural Network model here...
model.add(TimeDistributed(Conv1D(64, kernel_size=1, activation='relu', input_shape=(None, 100, 1))))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(128, kernel_size=1, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(64, kernel_size=1, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Flatten()))
# model.add(Dense(5, kernel_regularizer=L2(0.01)))
model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(Dropout(0.25))
model.add(Bidirectional(LSTM(100, return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae', 'mape'])

history = model.fit(train_X, train_Y, validation_data=(test_X,test_Y), epochs=40,batch_size=40, verbose=1, shuffle =True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-12T07:24:41.306101Z","iopub.execute_input":"2021-11-12T07:24:41.306341Z","iopub.status.idle":"2021-11-12T07:24:41.627666Z","shell.execute_reply.started":"2021-11-12T07:24:41.306314Z","shell.execute_reply":"2021-11-12T07:24:41.627016Z"}}
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-12T07:24:41.628952Z","iopub.execute_input":"2021-11-12T07:24:41.629465Z","iopub.status.idle":"2021-11-12T07:24:41.948542Z","shell.execute_reply.started":"2021-11-12T07:24:41.629428Z","shell.execute_reply":"2021-11-12T07:24:41.947844Z"}}
plt.plot(history.history['mse'], label='train mse')
plt.plot(history.history['val_mse'], label='val mse')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-12T07:24:41.950194Z","iopub.execute_input":"2021-11-12T07:24:41.950877Z","iopub.status.idle":"2021-11-12T07:24:42.269485Z","shell.execute_reply.started":"2021-11-12T07:24:41.950833Z","shell.execute_reply":"2021-11-12T07:24:42.268775Z"}}
plt.plot(history.history['mae'], label='train mae')
plt.plot(history.history['val_mae'], label='val mae')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-12T07:21:40.917993Z","iopub.execute_input":"2021-11-12T07:21:40.918574Z","iopub.status.idle":"2021-11-12T07:21:41.241719Z","shell.execute_reply.started":"2021-11-12T07:21:40.918537Z","shell.execute_reply":"2021-11-12T07:21:41.240997Z"}}
plt.plot(history.history['mape'], label='train mape')
plt.plot(history.history['val_mape'], label='val mape')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-12T07:15:36.695424Z","iopub.status.idle":"2021-11-12T07:15:36.695906Z","shell.execute_reply.started":"2021-11-12T07:15:36.695668Z","shell.execute_reply":"2021-11-12T07:15:36.695698Z"}}
# After the model has been constructed, we need to train
from tensorflow.keras.utils import plot_model
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-12T07:15:36.698070Z","iopub.status.idle":"2021-11-12T07:15:36.698776Z","shell.execute_reply.started":"2021-11-12T07:15:36.698513Z","shell.execute_reply":"2021-11-12T07:15:36.698537Z"}}
model.evaluate(test_X, test_Y)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-12T07:26:20.485682Z","iopub.execute_input":"2021-11-12T07:26:20.486401Z","iopub.status.idle":"2021-11-12T07:26:22.011937Z","shell.execute_reply.started":"2021-11-12T07:26:20.486364Z","shell.execute_reply":"2021-11-12T07:26:22.011245Z"}}
predicted  = model.predict(test_X)
test_label = test_Y.reshape(-1,1)
predicted = np.array(predicted[:,0]).reshape(-1,1)
len_t = len(train_X)
for j in range(len_t , len_t + len(test_X)):
    temp = data.iloc[j,3]
    test_label[j - len_t] = test_label[j - len_t] * temp + temp
    predicted[j - len_t] = predicted[j - len_t] * temp + temp
plt.plot(predicted, color = 'green', label = 'Predicted  Stock Price')
plt.plot(test_label, color = 'red', label = 'Real Stock Price')
plt.title(' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(' Stock Price')
plt.legend()
plt.show()

# %% [markdown]
# ## Testing part

# %% [markdown]
# In this part, the model is saved and loaded back again. Then, it's made to train again but with different data to check it's loss and prediction

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-12T07:15:36.702119Z","iopub.status.idle":"2021-11-12T07:15:36.702794Z","shell.execute_reply.started":"2021-11-12T07:15:36.702541Z","shell.execute_reply":"2021-11-12T07:15:36.702564Z"}}
# First we need to save a model
model.save("model.h5")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-12T07:15:36.704044Z","iopub.status.idle":"2021-11-12T07:15:36.704771Z","shell.execute_reply.started":"2021-11-12T07:15:36.704518Z","shell.execute_reply":"2021-11-12T07:15:36.704551Z"}}
# Load model
new_model = tf.keras.models.load_model("./model.h5")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-12T07:15:36.706009Z","iopub.status.idle":"2021-11-12T07:15:36.706681Z","shell.execute_reply.started":"2021-11-12T07:15:36.706431Z","shell.execute_reply":"2021-11-12T07:15:36.706454Z"}}
new_model.summary()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-12T07:28:04.894547Z","iopub.execute_input":"2021-11-12T07:28:04.895483Z","iopub.status.idle":"2021-11-12T07:28:12.217724Z","shell.execute_reply.started":"2021-11-12T07:28:04.895435Z","shell.execute_reply":"2021-11-12T07:28:12.216877Z"}}
# For data preprocessing and analysis part
#data2 = pd.read_csv('../input/price-volume-data-for-all-us-stocks-etfs/Stocks/aaoi.us.txt')
data2 = pd.read_csv('../input/nifty50-stock-market-data/SBIN.csv')
#data2 = pd.read_csv('../input/stock-market-data/stock_market_data/nasdaq/csv/ACTG.csv')
# Any CSV or TXT file can be added here....
data2.dropna(inplace=True)
data2.head()

data2.reset_index(drop=True, inplace=True)
data2.fillna(data.mean(), inplace=True)
data2.head()

df2 = data2.drop('Date', axis=1)
print(df2)

X = []
Y = []
window_size=100
for i in range(1 , len(df2) - window_size -1 , 1):
    first = df2.iloc[i,3]
    temp = []
    temp2 = []
    for j in range(window_size):
        temp.append((df2.iloc[i + j, 3] - first) / first)
    # for j in range(week):
    temp2.append((df2.iloc[i + window_size, 3] - first) / first)
    # X.append(np.array(stock.iloc[i:i+window_size,4]).reshape(50,1))
    # Y.append(np.array(stock.iloc[i+window_size,4]).reshape(1,1))
    # print(stock2.iloc[i:i+window_size,4])
    X.append(np.array(temp).reshape(100, 1))
    Y.append(np.array(temp2).reshape(1, 1))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

train_X = np.array(x_train)
test_X = np.array(x_test)
train_Y = np.array(y_train)
test_Y = np.array(y_test)

train_X = train_X.reshape(train_X.shape[0],1,100,1)
test_X = test_X.reshape(test_X.shape[0],1,100,1)

print(len(train_X))
print(len(test_X))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-12T07:28:12.219426Z","iopub.execute_input":"2021-11-12T07:28:12.219897Z","iopub.status.idle":"2021-11-12T07:28:12.617964Z","shell.execute_reply.started":"2021-11-12T07:28:12.219858Z","shell.execute_reply":"2021-11-12T07:28:12.617254Z"}}
predicted  = model.predict(test_X)
test_label = test_Y.reshape(-1,1)
predicted = np.array(predicted[:,0]).reshape(-1,1)
len_t = len(train_X)
for j in range(len_t , len_t + len(test_X)):
    temp = data2.iloc[j,3]
    test_label[j - len_t] = test_label[j - len_t] * temp + temp
    predicted[j - len_t] = predicted[j - len_t] * temp + temp
plt.plot(predicted, color = 'green', label = 'Predicted  Stock Price')
plt.plot(test_label, color = 'red', label = 'Real Stock Price')
plt.title(' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(' Stock Price')
plt.legend()
plt.show()