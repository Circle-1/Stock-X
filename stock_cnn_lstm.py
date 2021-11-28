# %% [markdown]
# # Stock Market Analysis using CNN-LSTM model
# This project is about analysis of Stock Market and providing suggestions and predictions to the stockholders. For this, we used CNN-LSTM approach to create a blank model, then use it to train on stock market data. Further implementation is discussed below...

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-28T06:34:28.116467Z","iopub.execute_input":"2021-11-28T06:34:28.116849Z","iopub.status.idle":"2021-11-28T06:34:28.122578Z","shell.execute_reply.started":"2021-11-28T06:34:28.116757Z","shell.execute_reply":"2021-11-28T06:34:28.121521Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-28T06:34:28.124670Z","iopub.execute_input":"2021-11-28T06:34:28.125326Z","iopub.status.idle":"2021-11-28T06:34:28.142670Z","shell.execute_reply.started":"2021-11-28T06:34:28.125281Z","shell.execute_reply":"2021-11-28T06:34:28.141520Z"}}
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

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2021-11-28T06:34:28.146347Z","iopub.execute_input":"2021-11-28T06:34:28.146984Z","iopub.status.idle":"2021-11-28T06:34:28.159466Z","shell.execute_reply.started":"2021-11-28T06:34:28.146935Z","shell.execute_reply":"2021-11-28T06:34:28.158426Z"}}
#1DP18XAREYFRWP4I
import requests
import csv
from tqdm import tqdm
key = "1DP18XAREYFRWP4I"

def request_stock_price_list(symbol, size, token):
    q_string = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={}&outputsize={}&apikey={}'

    print("Retrieving stock price data from Alpha Vantage (This may take a while)...")
    r = requests.get(q_string.format(symbol, size, token))
    print("Data has been successfully downloaded...")
    date = []
    colnames = list(range(0, 7))
    df = pd.DataFrame(columns = colnames)
    print("Sorting the retrieved data into a dataframe...")
    for i in tqdm(r.json()['Time Series (Daily)'].keys()):
        date.append(i)
        row = pd.DataFrame.from_dict(r.json()['Time Series (Daily)'][i], orient='index').reset_index().T[1:]
        df = pd.concat([df, row], ignore_index=True)
    df.columns = ["open", "high", "low", "close", "adjusted close", "volume", "dividend amount", "split cf"]
    df['date'] = date
    return df

# %% [code] {"execution":{"iopub.status.busy":"2021-11-28T06:34:28.161973Z","iopub.execute_input":"2021-11-28T06:34:28.162506Z","iopub.status.idle":"2021-11-28T06:36:32.863536Z","shell.execute_reply.started":"2021-11-28T06:34:28.162459Z","shell.execute_reply":"2021-11-28T06:36:32.862428Z"}}
cv1 = request_stock_price_list('IBM', 'full', key)
print(cv1.head)
cv1.to_csv('data.csv')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-28T06:36:32.865362Z","iopub.execute_input":"2021-11-28T06:36:32.866215Z","iopub.status.idle":"2021-11-28T06:36:32.903318Z","shell.execute_reply.started":"2021-11-28T06:36:32.866169Z","shell.execute_reply":"2021-11-28T06:36:32.902157Z"}}
# For data preprocessing and analysis part
data = pd.read_csv('../input/price-volume-data-for-all-us-stocks-etfs/Stocks/abe.us.txt')
#data = pd.read_csv('../input/nifty50-stock-market-data/COALINDIA.csv')
#data = pd.read_csv('../input/stock-market-data/stock_market_data/nasdaq/csv/ABCO.csv')
#data = pd.read_csv('./data.csv')
# Any CSV or TXT file can be added here....
data.dropna(inplace=True)
data.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-28T06:36:32.904932Z","iopub.execute_input":"2021-11-28T06:36:32.905350Z","iopub.status.idle":"2021-11-28T06:36:32.927988Z","shell.execute_reply.started":"2021-11-28T06:36:32.905306Z","shell.execute_reply":"2021-11-28T06:36:32.926993Z"}}
data.info()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-28T06:36:32.931328Z","iopub.execute_input":"2021-11-28T06:36:32.932006Z","iopub.status.idle":"2021-11-28T06:36:32.970247Z","shell.execute_reply.started":"2021-11-28T06:36:32.931963Z","shell.execute_reply":"2021-11-28T06:36:32.968967Z"}}
data.describe()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-28T06:36:32.972045Z","iopub.execute_input":"2021-11-28T06:36:32.972715Z","iopub.status.idle":"2021-11-28T06:36:32.986005Z","shell.execute_reply.started":"2021-11-28T06:36:32.972669Z","shell.execute_reply":"2021-11-28T06:36:32.984696Z"}}
data.isnull().sum()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-28T06:36:32.987767Z","iopub.execute_input":"2021-11-28T06:36:32.988431Z","iopub.status.idle":"2021-11-28T06:36:33.016457Z","shell.execute_reply.started":"2021-11-28T06:36:32.988388Z","shell.execute_reply":"2021-11-28T06:36:33.015453Z"}}
data.reset_index(drop=True, inplace=True)
data.fillna(data.mean(), inplace=True)
data.head()

# %% [markdown]
# After that, we'll visualize the data for understanding, this is shown below...

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-28T06:36:33.017981Z","iopub.execute_input":"2021-11-28T06:36:33.018993Z","iopub.status.idle":"2021-11-28T06:36:33.985477Z","shell.execute_reply.started":"2021-11-28T06:36:33.018950Z","shell.execute_reply":"2021-11-28T06:36:33.983664Z"}}
cols_plot = ['Open', 'High', 'Low','Close']
axes = data[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Daily trade')

# %% [markdown]
# Then we'd print the data after making changes and dropping null data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-28T06:36:33.987235Z","iopub.execute_input":"2021-11-28T06:36:33.990100Z","iopub.status.idle":"2021-11-28T06:36:34.353571Z","shell.execute_reply.started":"2021-11-28T06:36:33.990054Z","shell.execute_reply":"2021-11-28T06:36:34.352516Z"}}
plt.plot(data['Close'], label="Close price")
plt.xlabel("Timestamp")
plt.ylabel("Closing price")
df = data
print(df)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-28T06:36:34.355413Z","iopub.execute_input":"2021-11-28T06:36:34.356084Z","iopub.status.idle":"2021-11-28T06:36:34.362796Z","shell.execute_reply.started":"2021-11-28T06:36:34.356035Z","shell.execute_reply":"2021-11-28T06:36:34.361460Z"}}
# Make DataFrame of the given data
#data = pd.DataFrame({"Date":['2005-02-25','2005-02-2','2005-03-01','2005-03-02','2005-03-01'],
#                    "Open":[6.4987,6.6072,6.6391,6.5753,6.5753],
#                     "High":[6.6009,6.7669,6.6773,6.6072,6.6135],
#                     "Low":[6.4668,6.5944,6.6072,6.5434,6.5562],
#                    "Close":[6.5753,6.6263,6.6072,6.5816,6.5944],
#                    "Volume":[55766,49343,31643,27101,17387],
#                    "OpenInt":[0,0,0,0,0]})
#data['Date'] = pd.to_numeric(data['Date'], errors='coerce')
#data['Date'] = data['Date'].astype(float)

#from sklearn.preprocessing import OrdinalEncoder
#ord_enc = OrdinalEncoder()
  
# Transform the data
#data[["Open","Close","OpenInt"]] = ord_enc.fit_transform(data[["Open","Close","OpenInt"]])

# import VarianceThreshold
#from sklearn.feature_selection import VarianceThreshold
#var_threshold = VarianceThreshold(threshold=0)   # threshold = 0 for constant
  
# fit the data
#var_threshold.fit(data)
  
# We can check the variance of different features as
#print(var_threshold.variances_)

#print(var_threshold.transform(data))
#print('' * 10,"Separator",'' * 10)
  
# shapes of data before transformed and after transformed
#print("EARLIER Shape of the DATA: ", data.shape)
#print("Shape after transformation: ", var_threshold.transform(data).shape)

# %% [markdown]
# The data has been analysed but it must be converted into data of shape [100,1] to make it easier for CNN to train on... Else it won't select necessary features and the model will fail

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-28T06:36:34.364738Z","iopub.execute_input":"2021-11-28T06:36:34.365455Z","iopub.status.idle":"2021-11-28T06:36:48.443119Z","shell.execute_reply.started":"2021-11-28T06:36:34.365410Z","shell.execute_reply":"2021-11-28T06:36:48.442126Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-28T06:36:48.444803Z","iopub.execute_input":"2021-11-28T06:36:48.445259Z","iopub.status.idle":"2021-11-28T06:37:33.257004Z","shell.execute_reply.started":"2021-11-28T06:36:48.445213Z","shell.execute_reply":"2021-11-28T06:37:33.255960Z"}}
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
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

history = model.fit(train_X, train_Y, validation_data=(test_X,test_Y), epochs=40,batch_size=40, verbose=1, shuffle =True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-28T06:37:33.259538Z","iopub.execute_input":"2021-11-28T06:37:33.259890Z","iopub.status.idle":"2021-11-28T06:37:33.668506Z","shell.execute_reply.started":"2021-11-28T06:37:33.259839Z","shell.execute_reply":"2021-11-28T06:37:33.667530Z"}}
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-28T06:37:33.672887Z","iopub.execute_input":"2021-11-28T06:37:33.673466Z","iopub.status.idle":"2021-11-28T06:37:34.179653Z","shell.execute_reply.started":"2021-11-28T06:37:33.673419Z","shell.execute_reply":"2021-11-28T06:37:34.178608Z"}}
plt.plot(history.history['mse'], label='train mse')
plt.plot(history.history['val_mse'], label='val mse')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-28T06:37:34.182120Z","iopub.execute_input":"2021-11-28T06:37:34.183514Z","iopub.status.idle":"2021-11-28T06:37:34.755047Z","shell.execute_reply.started":"2021-11-28T06:37:34.183340Z","shell.execute_reply":"2021-11-28T06:37:34.753982Z"}}
plt.plot(history.history['mae'], label='train mae')
plt.plot(history.history['val_mae'], label='val mae')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-28T06:37:34.756873Z","iopub.execute_input":"2021-11-28T06:37:34.757327Z","iopub.status.idle":"2021-11-28T06:37:35.089191Z","shell.execute_reply.started":"2021-11-28T06:37:34.757281Z","shell.execute_reply":"2021-11-28T06:37:35.088018Z"}}
# After the model has been constructed, we need to train
from tensorflow.keras.utils import plot_model
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-28T06:37:35.091298Z","iopub.execute_input":"2021-11-28T06:37:35.091939Z","iopub.status.idle":"2021-11-28T06:37:35.275864Z","shell.execute_reply.started":"2021-11-28T06:37:35.091871Z","shell.execute_reply":"2021-11-28T06:37:35.274871Z"}}
model.evaluate(test_X, test_Y)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-28T06:37:35.277814Z","iopub.execute_input":"2021-11-28T06:37:35.279009Z","iopub.status.idle":"2021-11-28T06:37:37.333830Z","shell.execute_reply.started":"2021-11-28T06:37:35.278964Z","shell.execute_reply":"2021-11-28T06:37:37.332878Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-28T06:37:37.335508Z","iopub.execute_input":"2021-11-28T06:37:37.336620Z","iopub.status.idle":"2021-11-28T06:37:37.438424Z","shell.execute_reply.started":"2021-11-28T06:37:37.336566Z","shell.execute_reply":"2021-11-28T06:37:37.437284Z"}}
# First we need to save a model
model.save("model.h5")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-28T06:37:37.440385Z","iopub.execute_input":"2021-11-28T06:37:37.440805Z","iopub.status.idle":"2021-11-28T06:37:38.802568Z","shell.execute_reply.started":"2021-11-28T06:37:37.440762Z","shell.execute_reply":"2021-11-28T06:37:38.801587Z"}}
# Load model
new_model = tf.keras.models.load_model("./model.h5")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-28T06:37:38.804463Z","iopub.execute_input":"2021-11-28T06:37:38.804781Z","iopub.status.idle":"2021-11-28T06:37:38.823022Z","shell.execute_reply.started":"2021-11-28T06:37:38.804739Z","shell.execute_reply":"2021-11-28T06:37:38.822027Z"}}
new_model.summary()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-28T06:37:38.824264Z","iopub.execute_input":"2021-11-28T06:37:38.824797Z","iopub.status.idle":"2021-11-28T06:38:03.646696Z","shell.execute_reply.started":"2021-11-28T06:37:38.824753Z","shell.execute_reply":"2021-11-28T06:38:03.644741Z"}}
# For data preprocessing and analysis part
#data2 = pd.read_csv('../input/price-volume-data-for-all-us-stocks-etfs/Stocks/aaoi.us.txt')
#data2 = pd.read_csv('../input/nifty50-stock-market-data/SBIN.csv')
#data2 = pd.read_csv('../input/stock-market-data/stock_market_data/nasdaq/csv/ACTG.csv')
data2 = pd.read_csv('./data.csv')
# Any CSV or TXT file can be added here....
data2.dropna(inplace=True)
data2.head()

data2.reset_index(drop=True, inplace=True)
data2.fillna(data.mean(), inplace=True)
data2.head()
df2 = data2.drop('date', axis=1)

print(df2)

X = []
Y = []
window_size=100
for i in range(1 , len(df2) - window_size -1 , 1):
    first = df2.iloc[i,4]
    temp = []
    temp2 = []
    for j in range(window_size):
        temp.append((df2.iloc[i + j, 4] - first) / first)
    # for j in range(week):
    temp2.append((df2.iloc[i + window_size, 4] - first) / first)
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-28T06:38:03.648932Z","iopub.execute_input":"2021-11-28T06:38:03.649725Z","iopub.status.idle":"2021-11-28T06:38:04.245123Z","shell.execute_reply.started":"2021-11-28T06:38:03.649677Z","shell.execute_reply":"2021-11-28T06:38:04.244134Z"}}
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
