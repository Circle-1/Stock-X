# %% [markdown]
# # Stock Market Analysis using CNN-LSTM model
# This project is about analysis of Stock Market and providing suggestions and predictions to the stockholders. For this, we used CNN-LSTM approach to create a blank model, then use it to train on stock market data. Further implementation is discussed below...

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-09T13:21:29.174017Z","iopub.execute_input":"2021-11-09T13:21:29.174376Z","iopub.status.idle":"2021-11-09T13:21:29.280247Z","shell.execute_reply.started":"2021-11-09T13:21:29.174294Z","shell.execute_reply":"2021-11-09T13:21:29.279248Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-09T13:21:29.282057Z","iopub.execute_input":"2021-11-09T13:21:29.282946Z","iopub.status.idle":"2021-11-09T13:21:29.994565Z","shell.execute_reply.started":"2021-11-09T13:21:29.282888Z","shell.execute_reply":"2021-11-09T13:21:29.993861Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-09T13:21:29.995636Z","iopub.execute_input":"2021-11-09T13:21:29.995913Z","iopub.status.idle":"2021-11-09T13:21:30.076032Z","shell.execute_reply.started":"2021-11-09T13:21:29.995863Z","shell.execute_reply":"2021-11-09T13:21:30.075029Z"}}
# For data preprocessing and analysis part
data = pd.read_csv('../input/price-volume-data-for-all-us-stocks-etfs/Stocks/abe.us.txt')
# Any CSV or TXT file can be added here....
data.dropna(inplace=True)
data.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-09T13:21:30.081203Z","iopub.execute_input":"2021-11-09T13:21:30.081550Z","iopub.status.idle":"2021-11-09T13:21:30.112977Z","shell.execute_reply.started":"2021-11-09T13:21:30.081516Z","shell.execute_reply":"2021-11-09T13:21:30.112251Z"}}
data.info()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-09T13:21:30.114838Z","iopub.execute_input":"2021-11-09T13:21:30.115304Z","iopub.status.idle":"2021-11-09T13:21:30.146510Z","shell.execute_reply.started":"2021-11-09T13:21:30.115268Z","shell.execute_reply":"2021-11-09T13:21:30.145859Z"}}
data.describe()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-09T13:21:30.147724Z","iopub.execute_input":"2021-11-09T13:21:30.147977Z","iopub.status.idle":"2021-11-09T13:21:30.157261Z","shell.execute_reply.started":"2021-11-09T13:21:30.147942Z","shell.execute_reply":"2021-11-09T13:21:30.156343Z"}}
data.isnull().sum()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-09T13:21:30.158678Z","iopub.execute_input":"2021-11-09T13:21:30.159346Z","iopub.status.idle":"2021-11-09T13:21:30.181326Z","shell.execute_reply.started":"2021-11-09T13:21:30.159300Z","shell.execute_reply":"2021-11-09T13:21:30.180699Z"}}
data.reset_index(drop=True, inplace=True)
data.fillna(data.mean(), inplace=True)
data.head()

# %% [markdown]
# After that, we'll visualize the data for understanding, this is shown below...

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-09T13:21:30.182282Z","iopub.execute_input":"2021-11-09T13:21:30.182521Z","iopub.status.idle":"2021-11-09T13:21:31.127862Z","shell.execute_reply.started":"2021-11-09T13:21:30.182490Z","shell.execute_reply":"2021-11-09T13:21:31.127024Z"}}
cols_plot = ['Open', 'Close', 'High','Low']
axes = data[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Daily trade')

# %% [markdown]
# Then we'd print the data after making changes and dropping null data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-09T13:21:31.129078Z","iopub.execute_input":"2021-11-09T13:21:31.129324Z","iopub.status.idle":"2021-11-09T13:21:31.571162Z","shell.execute_reply.started":"2021-11-09T13:21:31.129291Z","shell.execute_reply":"2021-11-09T13:21:31.570472Z"}}
plt.plot(data['Close'], label="Close price")
plt.xlabel("Timestamp")
plt.ylabel("Closing price")

df = data.drop('Date', axis=1)
print(df)

# %% [markdown]
# The data has been analysed but it must be converted into data of shape [100,1] to make it easier for CNN to train on... Else it won't select necessary features and the model will fail

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-09T13:21:31.576847Z","iopub.execute_input":"2021-11-09T13:21:31.578974Z","iopub.status.idle":"2021-11-09T13:21:41.191489Z","shell.execute_reply.started":"2021-11-09T13:21:31.578932Z","shell.execute_reply":"2021-11-09T13:21:41.190653Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-09T13:21:41.194090Z","iopub.execute_input":"2021-11-09T13:21:41.194342Z","iopub.status.idle":"2021-11-09T13:22:25.684450Z","shell.execute_reply.started":"2021-11-09T13:21:41.194310Z","shell.execute_reply":"2021-11-09T13:22:25.683646Z"}}
# For creating model and training
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, TimeDistributed
from tensorflow.keras.layers import MaxPooling1D, Flatten
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras.metrics import Accuracy

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
model.add(Bidirectional(LSTM(200, return_sequences=True)))
model.add(Dropout(0.25))
model.add(Bidirectional(LSTM(200, return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

history = model.fit(train_X, train_Y, validation_data=(test_X,test_Y), epochs=40,batch_size=40, verbose=1, shuffle =True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-09T13:22:25.686424Z","iopub.execute_input":"2021-11-09T13:22:25.686714Z","iopub.status.idle":"2021-11-09T13:22:26.004151Z","shell.execute_reply.started":"2021-11-09T13:22:25.686678Z","shell.execute_reply":"2021-11-09T13:22:26.003460Z"}}
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-09T13:22:26.005289Z","iopub.execute_input":"2021-11-09T13:22:26.005965Z","iopub.status.idle":"2021-11-09T13:22:26.948672Z","shell.execute_reply.started":"2021-11-09T13:22:26.005918Z","shell.execute_reply":"2021-11-09T13:22:26.947835Z"}}
# After the model has been constructed, we need to train
from tensorflow.keras.utils import plot_model
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-09T13:22:26.950576Z","iopub.execute_input":"2021-11-09T13:22:26.950863Z","iopub.status.idle":"2021-11-09T13:22:27.177784Z","shell.execute_reply.started":"2021-11-09T13:22:26.950824Z","shell.execute_reply":"2021-11-09T13:22:27.177118Z"}}
model.evaluate(test_X, test_Y)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-09T13:22:27.178861Z","iopub.execute_input":"2021-11-09T13:22:27.180003Z","iopub.status.idle":"2021-11-09T13:22:28.895400Z","shell.execute_reply.started":"2021-11-09T13:22:27.179965Z","shell.execute_reply":"2021-11-09T13:22:28.894660Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-09T13:26:38.356138Z","iopub.execute_input":"2021-11-09T13:26:38.356918Z","iopub.status.idle":"2021-11-09T13:26:38.481490Z","shell.execute_reply.started":"2021-11-09T13:26:38.356861Z","shell.execute_reply":"2021-11-09T13:26:38.480621Z"}}
# First we need to save a model
model.save("model.h5")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-09T13:26:51.575999Z","iopub.execute_input":"2021-11-09T13:26:51.576572Z","iopub.status.idle":"2021-11-09T13:26:52.619503Z","shell.execute_reply.started":"2021-11-09T13:26:51.576534Z","shell.execute_reply":"2021-11-09T13:26:52.618707Z"}}
# Load model
new_model = tf.keras.models.load_model("./model.h5")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-09T13:26:52.621283Z","iopub.execute_input":"2021-11-09T13:26:52.621542Z","iopub.status.idle":"2021-11-09T13:26:52.634999Z","shell.execute_reply.started":"2021-11-09T13:26:52.621508Z","shell.execute_reply":"2021-11-09T13:26:52.634177Z"}}
new_model.summary()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-09T13:27:01.551554Z","iopub.execute_input":"2021-11-09T13:27:01.551852Z","iopub.status.idle":"2021-11-09T13:27:04.615300Z","shell.execute_reply.started":"2021-11-09T13:27:01.551815Z","shell.execute_reply":"2021-11-09T13:27:04.614554Z"}}
# For data preprocessing and analysis part
data2 = pd.read_csv('../input/price-volume-data-for-all-us-stocks-etfs/Stocks/aaoi.us.txt')
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-09T13:27:29.993587Z","iopub.execute_input":"2021-11-09T13:27:29.993873Z","iopub.status.idle":"2021-11-09T13:27:39.865258Z","shell.execute_reply.started":"2021-11-09T13:27:29.993843Z","shell.execute_reply":"2021-11-09T13:27:39.864595Z"}}
new_model.fit(train_X, train_Y, validation_data=(test_X,test_Y), epochs=40,batch_size=64, verbose=1, shuffle =True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-09T13:27:45.521456Z","iopub.execute_input":"2021-11-09T13:27:45.522235Z","iopub.status.idle":"2021-11-09T13:27:45.868333Z","shell.execute_reply.started":"2021-11-09T13:27:45.522174Z","shell.execute_reply":"2021-11-09T13:27:45.867642Z"}}
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