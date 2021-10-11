# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-10-11T14:08:21.753592Z","iopub.execute_input":"2021-10-11T14:08:21.754525Z","iopub.status.idle":"2021-10-11T14:08:21.760651Z","shell.execute_reply.started":"2021-10-11T14:08:21.75448Z","shell.execute_reply":"2021-10-11T14:08:21.758994Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2021-10-11T14:11:03.580507Z","iopub.execute_input":"2021-10-11T14:11:03.581236Z","iopub.status.idle":"2021-10-11T14:11:03.70209Z","shell.execute_reply.started":"2021-10-11T14:11:03.581198Z","shell.execute_reply":"2021-10-11T14:11:03.701438Z"}}
import math
import seaborn as sns
import datetime as dt
from datetime import datetime    
sns.set_style("whitegrid")
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use("ggplot")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-10-11T14:11:07.422866Z","iopub.execute_input":"2021-10-11T14:11:07.42356Z","iopub.status.idle":"2021-10-11T14:11:07.450209Z","shell.execute_reply.started":"2021-10-11T14:11:07.423522Z","shell.execute_reply":"2021-10-11T14:11:07.449297Z"}}
# For data preprocessing and analysis part
data = pd.read_csv('../input/price-volume-data-for-all-us-stocks-etfs/Stocks/abe.us.txt')
# Any CSV or TXT file can be added here....
data.dropna(inplace=True)
data.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-10-11T14:11:17.178736Z","iopub.execute_input":"2021-10-11T14:11:17.179231Z","iopub.status.idle":"2021-10-11T14:11:17.208706Z","shell.execute_reply.started":"2021-10-11T14:11:17.179182Z","shell.execute_reply":"2021-10-11T14:11:17.207884Z"}}
data.info()

# %% [code] {"execution":{"iopub.status.busy":"2021-10-11T14:11:28.33675Z","iopub.execute_input":"2021-10-11T14:11:28.337475Z","iopub.status.idle":"2021-10-11T14:11:28.369154Z","shell.execute_reply.started":"2021-10-11T14:11:28.337436Z","shell.execute_reply":"2021-10-11T14:11:28.368506Z"}}
data.describe()

# %% [code] {"execution":{"iopub.status.busy":"2021-10-11T14:11:46.139168Z","iopub.execute_input":"2021-10-11T14:11:46.139778Z","iopub.status.idle":"2021-10-11T14:11:46.14842Z","shell.execute_reply.started":"2021-10-11T14:11:46.139738Z","shell.execute_reply":"2021-10-11T14:11:46.147393Z"}}
data.isnull().sum()

# %% [code] {"execution":{"iopub.status.busy":"2021-10-11T14:11:57.544429Z","iopub.execute_input":"2021-10-11T14:11:57.544722Z","iopub.status.idle":"2021-10-11T14:11:57.570212Z","shell.execute_reply.started":"2021-10-11T14:11:57.54469Z","shell.execute_reply":"2021-10-11T14:11:57.569556Z"}}
data.reset_index(drop=True, inplace=True)
data.fillna(data.mean(), inplace=True)
data.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-10-11T14:12:47.460503Z","iopub.execute_input":"2021-10-11T14:12:47.46121Z","iopub.status.idle":"2021-10-11T14:12:48.168105Z","shell.execute_reply.started":"2021-10-11T14:12:47.461176Z","shell.execute_reply":"2021-10-11T14:12:48.167412Z"}}
cols_plot = ['Open', 'Close', 'High','Low']
axes = df[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Daily trade')

# %% [code] {"execution":{"iopub.status.busy":"2021-10-11T14:08:25.758534Z","iopub.execute_input":"2021-10-11T14:08:25.759125Z","iopub.status.idle":"2021-10-11T14:08:25.978112Z","shell.execute_reply.started":"2021-10-11T14:08:25.759087Z","shell.execute_reply":"2021-10-11T14:08:25.977428Z"}}
plt.plot(data['Close'], label="Close price")
plt.xlabel("Timestamp")
plt.ylabel("Closing price")

df = data.drop('Date', axis=1)
print(df)

# %% [code] {"execution":{"iopub.status.busy":"2021-10-11T14:15:35.239329Z","iopub.execute_input":"2021-10-11T14:15:35.240054Z","iopub.status.idle":"2021-10-11T14:15:44.798693Z","shell.execute_reply.started":"2021-10-11T14:15:35.240015Z","shell.execute_reply":"2021-10-11T14:15:44.797899Z"}}
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X = []
Y = []
window_size=100
for i in range(1 , len(df) - window_size -1 , 1):
    first = df.iloc[i,3]
    temp = []
    temp2 = []
    for j in range(window_size):
        temp.append((df.iloc[i + j, 3] - first) / first)
    # for j in range(week):
    temp2.append((df.iloc[i + window_size, 3] - first) / first)
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

# %% [markdown]
# ## Training part

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-10-11T14:15:47.837196Z","iopub.execute_input":"2021-10-11T14:15:47.837947Z","iopub.status.idle":"2021-10-11T14:16:06.35578Z","shell.execute_reply.started":"2021-10-11T14:15:47.837911Z","shell.execute_reply":"2021-10-11T14:16:06.355093Z"}}
# For creating model and training
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, TimeDistributed
from tensorflow.keras.layers import MaxPooling1D, Flatten
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras.metrics import Accuracy

model = tf.keras.Sequential()

# Creating the Neural Network model here...
model.add(TimeDistributed(Conv1D(64, kernel_size=1, activation='relu', input_shape=(None, 50, 1))))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(128, kernel_size=1, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(64, kernel_size=1, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Flatten()))
model.add(Dense(5, kernel_regularizer=L1(0.01)))
model.add(Bidirectional(LSTM(200, return_sequences=True)))
model.add(Dropout(0.25))
model.add(Bidirectional(LSTM(200, return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=[Accuracy()])

history = model.fit(train_X, train_Y, validation_data=(test_X,test_Y), epochs=25,batch_size=64, verbose=1, shuffle =True)

# %% [code] {"execution":{"iopub.status.busy":"2021-10-11T14:13:55.978348Z","iopub.execute_input":"2021-10-11T14:13:55.978608Z","iopub.status.idle":"2021-10-11T14:13:56.270413Z","shell.execute_reply.started":"2021-10-11T14:13:55.978579Z","shell.execute_reply":"2021-10-11T14:13:56.269602Z"}}
# After the model has been constructed, we need to train
from tensorflow.keras.utils import plot_model
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# %% [code] {"execution":{"iopub.status.busy":"2021-10-11T14:14:00.344815Z","iopub.execute_input":"2021-10-11T14:14:00.34551Z","iopub.status.idle":"2021-10-11T14:14:00.641421Z","shell.execute_reply.started":"2021-10-11T14:14:00.345466Z","shell.execute_reply":"2021-10-11T14:14:00.640743Z"}}
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()

# %% [code] {"execution":{"iopub.status.busy":"2021-10-11T14:14:07.439628Z","iopub.execute_input":"2021-10-11T14:14:07.440257Z","iopub.status.idle":"2021-10-11T14:14:07.587684Z","shell.execute_reply.started":"2021-10-11T14:14:07.440217Z","shell.execute_reply":"2021-10-11T14:14:07.587004Z"}}
model.evaluate(test_X, test_Y)

# %% [code] {"execution":{"iopub.status.busy":"2021-10-11T14:14:09.154538Z","iopub.execute_input":"2021-10-11T14:14:09.155232Z","iopub.status.idle":"2021-10-11T14:14:10.677942Z","shell.execute_reply.started":"2021-10-11T14:14:09.155197Z","shell.execute_reply":"2021-10-11T14:14:10.67725Z"}}
predicted  = model.predict(test_X)
test_label = (test_Y)
predicted = np.array(predicted[:,0]).reshape(-1,1)
len_t = len(train_X)
for j in range(len_t , len_t + len(test_X)):
    temp =data.iloc[j,3]
    test_label[j - len_t] = test_label[j - len_t] * temp + temp
    predicted[j - len_t] = predicted[j - len_t] * temp + temp
plt.plot(predicted, color = 'green', label = 'Predicted  Stock Price')
plt.title(' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(' Stock Price')
plt.legend()
plt.show()

# %% [markdown]
# ## Testing part

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-10-11T14:09:13.504766Z","iopub.execute_input":"2021-10-11T14:09:13.505504Z","iopub.status.idle":"2021-10-11T14:09:13.600594Z","shell.execute_reply.started":"2021-10-11T14:09:13.505465Z","shell.execute_reply":"2021-10-11T14:09:13.599773Z"}}
# First we need to save a model
model.save("saved_model.h5")

# %% [code] {"execution":{"iopub.status.busy":"2021-10-11T14:09:16.112009Z","iopub.execute_input":"2021-10-11T14:09:16.112769Z","iopub.status.idle":"2021-10-11T14:09:17.710421Z","shell.execute_reply.started":"2021-10-11T14:09:16.11272Z","shell.execute_reply":"2021-10-11T14:09:17.709681Z"}}
# Load model
new_model = tf.keras.models.load_model("./saved_model.h5")

# %% [code] {"execution":{"iopub.status.busy":"2021-10-11T14:09:18.563967Z","iopub.execute_input":"2021-10-11T14:09:18.564207Z","iopub.status.idle":"2021-10-11T14:09:18.579847Z","shell.execute_reply.started":"2021-10-11T14:09:18.564179Z","shell.execute_reply":"2021-10-11T14:09:18.57919Z"}}
new_model.summary()

# %% [code]
