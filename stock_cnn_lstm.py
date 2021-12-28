# %% [markdown]
# # Stock Market Prediction using CNN-LSTM model
# This project is about analysis of Stock Market and providing predictions to the stockholders. For this, we used CNN-LSTM approach to create a blank model, then use it to train on stock market data. Further implementation is discussed below...

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:35:47.563443Z","iopub.execute_input":"2021-12-21T04:35:47.56378Z","iopub.status.idle":"2021-12-21T04:35:47.568846Z","shell.execute_reply.started":"2021-12-21T04:35:47.563747Z","shell.execute_reply":"2021-12-21T04:35:47.568215Z"}}
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
# # Data Preprocessing and Analysis

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:35:47.579406Z","iopub.execute_input":"2021-12-21T04:35:47.579971Z","iopub.status.idle":"2021-12-21T04:35:47.589144Z","shell.execute_reply.started":"2021-12-21T04:35:47.579926Z","shell.execute_reply":"2021-12-21T04:35:47.588306Z"}}
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
# Before preprocessing data, a function to fetch real-time stock data (using Alpha Vantage API) is made

# %% [code] {"_kg_hide-input":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:35:47.590602Z","iopub.execute_input":"2021-12-21T04:35:47.591151Z","iopub.status.idle":"2021-12-21T04:35:47.801833Z","shell.execute_reply.started":"2021-12-21T04:35:47.591118Z","shell.execute_reply":"2021-12-21T04:35:47.801065Z"}}
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
key = user_secrets.get_secret("api")

import requests
import csv
from tqdm import tqdm

def request_stock_price_list(symbol, size, token):
    q_string = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&outputsize={}&apikey={}'
    
    print("Retrieving stock price data from Alpha Vantage (This may take a while)...")
    r = requests.get(q_string.format(symbol, size, token))
    print("Data has been successfully downloaded...")
    date = []
    colnames = list(range(0, 5))
    df = pd.DataFrame(columns = colnames)
    print("Sorting the retrieved data into a dataframe...")
    for i in tqdm(r.json()['Time Series (Daily)'].keys()):
        date.append(i)
        row = pd.DataFrame.from_dict(r.json()['Time Series (Daily)'][i], orient='index').reset_index().T[1:]
        df = pd.concat([df, row], ignore_index=True)
    df.columns = ["open", "high", "low", "close", "volume"]
    df['date'] = date
    return df

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:35:47.803236Z","iopub.execute_input":"2021-12-21T04:35:47.803522Z","iopub.status.idle":"2021-12-21T04:38:02.748172Z","shell.execute_reply.started":"2021-12-21T04:35:47.803495Z","shell.execute_reply":"2021-12-21T04:38:02.746951Z"}}
# UNCOMMENT THE CELL IF DATA IS NEEDED TO BE LOADED FOR 1ST TIME

cv1 = request_stock_price_list('IBM', 'full', key)
print(cv1.head)
cv1.to_csv('data.csv')

# %% [markdown]
# Then the datasets are loaded

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:38:02.750106Z","iopub.execute_input":"2021-12-21T04:38:02.750426Z","iopub.status.idle":"2021-12-21T04:38:02.786025Z","shell.execute_reply.started":"2021-12-21T04:38:02.750394Z","shell.execute_reply":"2021-12-21T04:38:02.785121Z"}}
# For data preprocessing and analysis part
data = pd.read_csv('../input/price-volume-data-for-all-us-stocks-etfs/Stocks/abe.us.txt')
#data = pd.read_csv('../input/nifty50-stock-market-data/COALINDIA.csv')
#data = pd.read_csv('../input/stock-market-data/stock_market_data/nasdaq/csv/ABCO.csv')
#data = pd.read_csv('./data.csv')
# Any CSV or TXT file can be added here....
data.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:38:02.788328Z","iopub.execute_input":"2021-12-21T04:38:02.78856Z","iopub.status.idle":"2021-12-21T04:38:02.803746Z","shell.execute_reply.started":"2021-12-21T04:38:02.788533Z","shell.execute_reply":"2021-12-21T04:38:02.802996Z"}}
data.info()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:38:02.804952Z","iopub.execute_input":"2021-12-21T04:38:02.805183Z","iopub.status.idle":"2021-12-21T04:38:02.839221Z","shell.execute_reply.started":"2021-12-21T04:38:02.805156Z","shell.execute_reply":"2021-12-21T04:38:02.838306Z"}}
data.describe()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:38:02.840716Z","iopub.execute_input":"2021-12-21T04:38:02.840993Z","iopub.status.idle":"2021-12-21T04:38:02.851535Z","shell.execute_reply.started":"2021-12-21T04:38:02.840958Z","shell.execute_reply":"2021-12-21T04:38:02.850823Z"}}
data.isnull().sum()

# %% [markdown]
# Filling null columns with mean values....

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:38:02.852917Z","iopub.execute_input":"2021-12-21T04:38:02.853485Z","iopub.status.idle":"2021-12-21T04:38:02.87935Z","shell.execute_reply.started":"2021-12-21T04:38:02.85345Z","shell.execute_reply":"2021-12-21T04:38:02.878453Z"}}
data.reset_index(drop=True, inplace=True)
data.fillna(data.mean(), inplace=True)
data.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:38:02.88105Z","iopub.execute_input":"2021-12-21T04:38:02.882042Z","iopub.status.idle":"2021-12-21T04:38:06.934658Z","shell.execute_reply.started":"2021-12-21T04:38:02.881998Z","shell.execute_reply":"2021-12-21T04:38:06.93371Z"}}
data.plot(legend=True,subplots=True, figsize = (12, 6))
plt.show()
#data['Close'].plot(legend=True, figsize = (12, 6))
#plt.show()
#data['Volume'].plot(legend=True,figsize=(12,7))
#plt.show()

data.shape
data.size
data.describe(include='all').T
data.dtypes
data.nunique()
ma_day = [10,50,100]

for ma in ma_day:
    column_name = "MA for %s days" %(str(ma))
    data[column_name]=pd.DataFrame.rolling(data['Close'],ma).mean()

data['Daily Return'] = data['Close'].pct_change()
# plot the daily return percentage
data['Daily Return'].plot(figsize=(12,5),legend=True,linestyle=':',marker='o')
plt.show()

sns.displot(data['Daily Return'].dropna(),bins=100,color='green')
plt.show()

date=pd.DataFrame(data['Date'])
closing_df1 = pd.DataFrame(data['Close'])
close1  = closing_df1.rename(columns={"Close": "data_close"})
close2=pd.concat([date,close1],axis=1)
close2.head()

data.reset_index(drop=True, inplace=True)
data.fillna(data.mean(), inplace=True)
data.head()

data.nunique()

data.sort_index(axis=1,ascending=True)

cols_plot = ['Open', 'High', 'Low','Close','Volume','MA for 10 days','MA for 50 days','MA for 100 days','Daily Return']
axes = data[cols_plot].plot(marker='.', alpha=0.7, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Daily trade')

plt.plot(data['Close'], label="Close price")
plt.xlabel("Timestamp")
plt.ylabel("Closing price")
df = data
print(df)

data.isnull().sum()

# %% [markdown]
# After that, we'll visualize the data for understanding, this is shown below...

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:38:06.935983Z","iopub.execute_input":"2021-12-21T04:38:06.936236Z","iopub.status.idle":"2021-12-21T04:38:07.764826Z","shell.execute_reply.started":"2021-12-21T04:38:06.936208Z","shell.execute_reply":"2021-12-21T04:38:07.763962Z"}}
cols_plot = ['Open', 'High', 'Low','Close']
axes = data[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Daily trade')

# %% [markdown]
# Then we'd print the data after making changes and dropping null data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:38:07.767589Z","iopub.execute_input":"2021-12-21T04:38:07.767945Z","iopub.status.idle":"2021-12-21T04:38:08.111112Z","shell.execute_reply.started":"2021-12-21T04:38:07.767913Z","shell.execute_reply":"2021-12-21T04:38:08.110498Z"}}
plt.plot(data['Close'], label="Close price")
plt.xlabel("Timestamp")
plt.ylabel("Closing price")
df = data
print(df)

df.describe().transpose()

# %% [markdown]
# The data has been analysed but it must be converted into data of shape [100,1] to make it easier for CNN to train on... Else it won't select necessary features and the model will fail

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:38:08.112119Z","iopub.execute_input":"2021-12-21T04:38:08.112695Z","iopub.status.idle":"2021-12-21T04:38:19.32783Z","shell.execute_reply.started":"2021-12-21T04:38:08.112664Z","shell.execute_reply":"2021-12-21T04:38:19.326917Z"}}
from sklearn.model_selection import train_test_split

X = []
Y = []
window_size=100
for i in range(1 , len(df) - window_size -1 , 1):
    first = df.iloc[i,2]
    temp = []
    temp2 = []
    for j in range(window_size):
        temp.append((df.iloc[i + j, 2] - first) / first)
    temp2.append((df.iloc[i + window_size, 2] - first) / first)
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
# # Training part

# %% [markdown]
# This part has 2 subparts: CNN and LSTM
# 
# For CNN, the layers are created with sizes 64,128,64 with kernel size = 3. In every layer, TimeDistributed function is added to track the features for every temporal slice of data with respect to time. In between, MaxPooling layers are added.
# 
# After that, it's passed to Bi-LSTM layers

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:38:19.328887Z","iopub.execute_input":"2021-12-21T04:38:19.329103Z","iopub.status.idle":"2021-12-21T04:39:24.139033Z","shell.execute_reply.started":"2021-12-21T04:38:19.329078Z","shell.execute_reply":"2021-12-21T04:39:24.138214Z"}}
# For creating model and training
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, TimeDistributed
from tensorflow.keras.layers import MaxPooling1D, Flatten
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.metrics import RootMeanSquaredError

model = tf.keras.Sequential()

# Creating the Neural Network model here...
# CNN layers
model.add(TimeDistributed(Conv1D(64, kernel_size=3, activation='relu', input_shape=(None, 100, 1))))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(128, kernel_size=3, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(64, kernel_size=3, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Flatten()))
# model.add(Dense(5, kernel_regularizer=L2(0.01)))

# LSTM layers
model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(100, return_sequences=False)))
model.add(Dropout(0.5))

#Final layers
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

history = model.fit(train_X, train_Y, validation_data=(test_X,test_Y), epochs=40,batch_size=40, verbose=1, shuffle =True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:39:24.142878Z","iopub.execute_input":"2021-12-21T04:39:24.143438Z","iopub.status.idle":"2021-12-21T04:39:24.457427Z","shell.execute_reply.started":"2021-12-21T04:39:24.143401Z","shell.execute_reply":"2021-12-21T04:39:24.456538Z"}}
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:39:24.458694Z","iopub.execute_input":"2021-12-21T04:39:24.458929Z","iopub.status.idle":"2021-12-21T04:39:24.765864Z","shell.execute_reply.started":"2021-12-21T04:39:24.458903Z","shell.execute_reply":"2021-12-21T04:39:24.76517Z"}}
plt.plot(history.history['mse'], label='train mse')
plt.plot(history.history['val_mse'], label='val mse')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:39:24.767204Z","iopub.execute_input":"2021-12-21T04:39:24.767475Z","iopub.status.idle":"2021-12-21T04:39:25.125249Z","shell.execute_reply.started":"2021-12-21T04:39:24.767446Z","shell.execute_reply":"2021-12-21T04:39:25.124392Z"}}
plt.plot(history.history['mae'], label='train mae')
plt.plot(history.history['val_mae'], label='val mae')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:39:25.126549Z","iopub.execute_input":"2021-12-21T04:39:25.126793Z","iopub.status.idle":"2021-12-21T04:39:25.406779Z","shell.execute_reply.started":"2021-12-21T04:39:25.126766Z","shell.execute_reply":"2021-12-21T04:39:25.405778Z"}}
# After the model has been constructed, we'll summarise it
from tensorflow.keras.utils import plot_model
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:39:25.408492Z","iopub.execute_input":"2021-12-21T04:39:25.408783Z","iopub.status.idle":"2021-12-21T04:39:25.603139Z","shell.execute_reply.started":"2021-12-21T04:39:25.408745Z","shell.execute_reply":"2021-12-21T04:39:25.602561Z"}}
model.evaluate(test_X, test_Y)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:39:25.604095Z","iopub.execute_input":"2021-12-21T04:39:25.604823Z","iopub.status.idle":"2021-12-21T04:39:27.157648Z","shell.execute_reply.started":"2021-12-21T04:39:25.604787Z","shell.execute_reply":"2021-12-21T04:39:27.156796Z"}}
from sklearn.metrics import explained_variance_score, mean_poisson_deviance, mean_gamma_deviance
from sklearn.metrics import r2_score
from sklearn.metrics import max_error

# predict probabilities for test set
yhat_probs = model.predict(test_X, verbose=0)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]

var = explained_variance_score(test_Y.reshape(-1,1), yhat_probs)
print('Variance: %f' % var)

r2 = r2_score(test_Y.reshape(-1,1), yhat_probs)
print('R2 Score: %f' % var)

var2 = max_error(test_Y.reshape(-1,1), yhat_probs)
print('Max Error: %f' % var2)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:39:27.159335Z","iopub.execute_input":"2021-12-21T04:39:27.160388Z","iopub.status.idle":"2021-12-21T04:39:27.682411Z","shell.execute_reply.started":"2021-12-21T04:39:27.160351Z","shell.execute_reply":"2021-12-21T04:39:27.681563Z"}}
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
# # Testing part

# %% [markdown]
# In this part, the model is saved and loaded back again. Then, it's made to train again but with different data to check it's loss and prediction

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:39:27.684065Z","iopub.execute_input":"2021-12-21T04:39:27.684368Z","iopub.status.idle":"2021-12-21T04:39:27.774175Z","shell.execute_reply.started":"2021-12-21T04:39:27.684329Z","shell.execute_reply":"2021-12-21T04:39:27.773153Z"}}
# First we need to save a model
model.save("model.h5")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:39:27.775944Z","iopub.execute_input":"2021-12-21T04:39:27.776253Z","iopub.status.idle":"2021-12-21T04:39:29.045776Z","shell.execute_reply.started":"2021-12-21T04:39:27.776215Z","shell.execute_reply":"2021-12-21T04:39:29.044992Z"}}
# Load model
new_model = tf.keras.models.load_model("./model.h5")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:39:29.047094Z","iopub.execute_input":"2021-12-21T04:39:29.047319Z","iopub.status.idle":"2021-12-21T04:39:29.059186Z","shell.execute_reply.started":"2021-12-21T04:39:29.047294Z","shell.execute_reply":"2021-12-21T04:39:29.056578Z"}}
new_model.summary()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:39:29.060545Z","iopub.execute_input":"2021-12-21T04:39:29.060811Z","iopub.status.idle":"2021-12-21T04:39:49.022632Z","shell.execute_reply.started":"2021-12-21T04:39:29.060783Z","shell.execute_reply":"2021-12-21T04:39:49.021546Z"}}
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

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

train_X = np.array(x_train)
test_X = np.array(x_test)
train_Y = np.array(y_train)
test_Y = np.array(y_test)

train_X = train_X.reshape(train_X.shape[0],1,100,1)
test_X = test_X.reshape(test_X.shape[0],1,100,1)

print(len(train_X))
print(len(test_X))

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T04:39:49.023842Z","iopub.execute_input":"2021-12-21T04:39:49.024067Z","iopub.status.idle":"2021-12-21T04:39:49.291965Z","shell.execute_reply.started":"2021-12-21T04:39:49.024041Z","shell.execute_reply":"2021-12-21T04:39:49.291234Z"}}
model.evaluate(test_X, test_Y)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:39:49.293123Z","iopub.execute_input":"2021-12-21T04:39:49.293364Z","iopub.status.idle":"2021-12-21T04:39:49.872487Z","shell.execute_reply.started":"2021-12-21T04:39:49.293337Z","shell.execute_reply":"2021-12-21T04:39:49.8719Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T04:39:49.873683Z","iopub.execute_input":"2021-12-21T04:39:49.874074Z","iopub.status.idle":"2021-12-21T04:40:02.588984Z","shell.execute_reply.started":"2021-12-21T04:39:49.87403Z","shell.execute_reply":"2021-12-21T04:40:02.587937Z"}}
# Converting model from HDF5 format to TFJS format...
!pip install tensorflowjs[wizard]
# Need to be done on a CLI and not in notebook
!tensorflowjs_converter --input_format=keras /kaggle/working/model.h5 /kaggle/working/model-tjs

# %% [markdown]
# # EDA

# %% [markdown]
# This section is exploratory data analysis on the dataset collected. This is just for analysing the data...

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:40:02.590615Z","iopub.execute_input":"2021-12-21T04:40:02.590901Z","iopub.status.idle":"2021-12-21T04:40:02.634536Z","shell.execute_reply.started":"2021-12-21T04:40:02.590871Z","shell.execute_reply":"2021-12-21T04:40:02.633915Z"}}
dataX = pd.read_csv('./data.csv')
dataY = pd.read_csv('./data.csv')
dataX.info()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:40:02.639954Z","iopub.execute_input":"2021-12-21T04:40:02.640224Z","iopub.status.idle":"2021-12-21T04:40:02.658405Z","shell.execute_reply.started":"2021-12-21T04:40:02.640195Z","shell.execute_reply":"2021-12-21T04:40:02.657623Z"}}
dataX.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:40:02.65954Z","iopub.execute_input":"2021-12-21T04:40:02.659803Z","iopub.status.idle":"2021-12-21T04:40:02.695784Z","shell.execute_reply.started":"2021-12-21T04:40:02.659774Z","shell.execute_reply":"2021-12-21T04:40:02.694856Z"}}
start_date = '2020-01-01'
end_date = '2021-11-29'

start = '2018-01-01'
end = '2020-01-01'

fill = (dataX['date']>=start_date) & (dataX['date']<=end_date)
dataX = dataX.loc[fill]
dataX

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:40:02.69779Z","iopub.execute_input":"2021-12-21T04:40:02.69812Z","iopub.status.idle":"2021-12-21T04:40:02.731473Z","shell.execute_reply.started":"2021-12-21T04:40:02.698079Z","shell.execute_reply":"2021-12-21T04:40:02.730836Z"}}
fill2 = (dataY['date']>=start) & (dataY['date']<=end)
dataY = dataY.loc[fill2]
dataY

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:40:02.732625Z","iopub.execute_input":"2021-12-21T04:40:02.732904Z","iopub.status.idle":"2021-12-21T04:40:02.771293Z","shell.execute_reply.started":"2021-12-21T04:40:02.732876Z","shell.execute_reply":"2021-12-21T04:40:02.770398Z"}}
dataX.describe()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:40:02.772861Z","iopub.execute_input":"2021-12-21T04:40:02.77334Z","iopub.status.idle":"2021-12-21T04:40:02.809578Z","shell.execute_reply.started":"2021-12-21T04:40:02.773299Z","shell.execute_reply":"2021-12-21T04:40:02.808936Z"}}
dataY.describe()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:40:02.811199Z","iopub.execute_input":"2021-12-21T04:40:02.811788Z","iopub.status.idle":"2021-12-21T04:40:03.215199Z","shell.execute_reply.started":"2021-12-21T04:40:02.81171Z","shell.execute_reply":"2021-12-21T04:40:03.214249Z"}}
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import r2_score,mean_squared_error

sns_plot = sns.distplot(dataX['close'])
sns_plot2 = sns.distplot(dataY['close'])

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:40:03.218469Z","iopub.execute_input":"2021-12-21T04:40:03.218744Z","iopub.status.idle":"2021-12-21T04:40:05.820017Z","shell.execute_reply.started":"2021-12-21T04:40:03.21869Z","shell.execute_reply":"2021-12-21T04:40:05.819097Z"}}
fig, ax = plt.subplots(4, 2, figsize = (15, 13))
sns.boxplot(x= dataX["close"], ax = ax[0,0])
sns.distplot(dataX['close'], ax = ax[0,1])
sns.boxplot(x= dataX["open"], ax = ax[1,0])
sns.distplot(dataX['open'], ax = ax[1,1])
sns.boxplot(x= dataX["high"], ax = ax[2,0])
sns.distplot(dataX['high'], ax = ax[2,1])
sns.boxplot(x= dataX["low"], ax = ax[3,0])
sns.distplot(dataX['low'], ax = ax[3,1])
plt.tight_layout()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:40:05.821397Z","iopub.execute_input":"2021-12-21T04:40:05.821638Z","iopub.status.idle":"2021-12-21T04:40:08.100887Z","shell.execute_reply.started":"2021-12-21T04:40:05.821613Z","shell.execute_reply":"2021-12-21T04:40:08.099988Z"}}
fig, ax = plt.subplots(4, 2, figsize = (15, 13))
sns.boxplot(x= dataY["close"], ax = ax[0,0])
sns.distplot(dataY['close'], ax = ax[0,1])
sns.boxplot(x= dataY["open"], ax = ax[1,0])
sns.distplot(dataY['open'], ax = ax[1,1])
sns.boxplot(x= dataY["high"], ax = ax[2,0])
sns.distplot(dataY['high'], ax = ax[2,1])
sns.boxplot(x= dataY["low"], ax = ax[3,0])
sns.distplot(dataY['low'], ax = ax[3,1])
plt.tight_layout()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:40:08.1021Z","iopub.execute_input":"2021-12-21T04:40:08.10234Z","iopub.status.idle":"2021-12-21T04:40:08.852576Z","shell.execute_reply.started":"2021-12-21T04:40:08.102312Z","shell.execute_reply":"2021-12-21T04:40:08.851707Z"}}
plt.figure(figsize=(10,6))
sns.heatmap(dataX.corr(),cmap=plt.cm.Reds,annot=True)
plt.title('Heatmap displaying the relationship between the features of the data (During COVID)',
         fontsize=13)
plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:40:08.853701Z","iopub.execute_input":"2021-12-21T04:40:08.854219Z","iopub.status.idle":"2021-12-21T04:40:09.522562Z","shell.execute_reply.started":"2021-12-21T04:40:08.854166Z","shell.execute_reply":"2021-12-21T04:40:09.521652Z"}}
plt.figure(figsize=(10,6))
sns.heatmap(dataY.corr(),cmap=plt.cm.Blues,annot=True)
plt.title('Heatmap displaying the relationship between the features of the data (Before COVID)',
         fontsize=13)
plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-12-21T04:40:09.523955Z","iopub.execute_input":"2021-12-21T04:40:09.524501Z","iopub.status.idle":"2021-12-21T04:41:28.796626Z","shell.execute_reply.started":"2021-12-21T04:40:09.524457Z","shell.execute_reply":"2021-12-21T04:41:28.795613Z"}}
# For other company....

# UNCOMMENT IF NEEDED...
cv2 = request_stock_price_list('RELIANCE.BSE', 'full', key)
print(cv2.head)
cv2.to_csv('data2.csv')

dataX = pd.read_csv('./data2.csv')
dataY = pd.read_csv('./data2.csv')
dataX.info()

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T04:41:28.798149Z","iopub.execute_input":"2021-12-21T04:41:28.798509Z","iopub.status.idle":"2021-12-21T04:41:28.828654Z","shell.execute_reply.started":"2021-12-21T04:41:28.798476Z","shell.execute_reply":"2021-12-21T04:41:28.827766Z"}}
start_date = '2020-01-01'
end_date = '2021-11-29'

start = '2018-01-01'
end = '2020-01-01'

fill = (dataX['date']>=start_date) & (dataX['date']<=end_date)
dataX = dataX.loc[fill]
dataX

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T04:41:28.829858Z","iopub.execute_input":"2021-12-21T04:41:28.830483Z","iopub.status.idle":"2021-12-21T04:41:28.864628Z","shell.execute_reply.started":"2021-12-21T04:41:28.830447Z","shell.execute_reply":"2021-12-21T04:41:28.863737Z"}}
fill2 = (dataY['date']>=start) & (dataY['date']<=end)
dataY = dataY.loc[fill2]
dataY

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T04:41:28.865927Z","iopub.execute_input":"2021-12-21T04:41:28.866195Z","iopub.status.idle":"2021-12-21T04:41:28.906445Z","shell.execute_reply.started":"2021-12-21T04:41:28.866158Z","shell.execute_reply":"2021-12-21T04:41:28.9055Z"}}
dataX.describe()

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T04:41:28.907836Z","iopub.execute_input":"2021-12-21T04:41:28.908282Z","iopub.status.idle":"2021-12-21T04:41:28.944283Z","shell.execute_reply.started":"2021-12-21T04:41:28.90825Z","shell.execute_reply":"2021-12-21T04:41:28.94341Z"}}
dataY.describe()

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T04:41:28.947389Z","iopub.execute_input":"2021-12-21T04:41:28.947953Z","iopub.status.idle":"2021-12-21T04:41:29.329513Z","shell.execute_reply.started":"2021-12-21T04:41:28.947918Z","shell.execute_reply":"2021-12-21T04:41:29.328893Z"}}
sns_plot = sns.distplot(dataX['close'])
sns_plot2 = sns.distplot(dataY['close'])

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T04:41:29.330517Z","iopub.execute_input":"2021-12-21T04:41:29.331083Z","iopub.status.idle":"2021-12-21T04:41:31.918911Z","shell.execute_reply.started":"2021-12-21T04:41:29.331048Z","shell.execute_reply":"2021-12-21T04:41:31.918041Z"}}
fig, ax = plt.subplots(4, 2, figsize = (15, 13))
sns.boxplot(x= dataX["close"], ax = ax[0,0])
sns.distplot(dataX['close'], ax = ax[0,1])
sns.boxplot(x= dataX["open"], ax = ax[1,0])
sns.distplot(dataX['open'], ax = ax[1,1])
sns.boxplot(x= dataX["high"], ax = ax[2,0])
sns.distplot(dataX['high'], ax = ax[2,1])
sns.boxplot(x= dataX["low"], ax = ax[3,0])
sns.distplot(dataX['low'], ax = ax[3,1])
plt.tight_layout()

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T04:41:31.920248Z","iopub.execute_input":"2021-12-21T04:41:31.920576Z","iopub.status.idle":"2021-12-21T04:41:33.999393Z","shell.execute_reply.started":"2021-12-21T04:41:31.920529Z","shell.execute_reply":"2021-12-21T04:41:33.99855Z"}}
fig, ax = plt.subplots(4, 2, figsize = (15, 13))
sns.boxplot(x= dataY["close"], ax = ax[0,0])
sns.distplot(dataY['close'], ax = ax[0,1])
sns.boxplot(x= dataY["open"], ax = ax[1,0])
sns.distplot(dataY['open'], ax = ax[1,1])
sns.boxplot(x= dataY["high"], ax = ax[2,0])
sns.distplot(dataY['high'], ax = ax[2,1])
sns.boxplot(x= dataY["low"], ax = ax[3,0])
sns.distplot(dataY['low'], ax = ax[3,1])
plt.tight_layout()

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T04:41:34.000619Z","iopub.execute_input":"2021-12-21T04:41:34.001084Z","iopub.status.idle":"2021-12-21T04:41:34.749598Z","shell.execute_reply.started":"2021-12-21T04:41:34.001053Z","shell.execute_reply":"2021-12-21T04:41:34.74867Z"}}
plt.figure(figsize=(10,6))
sns.heatmap(dataX.corr(),cmap=plt.cm.Reds,annot=True)
plt.title('Heatmap displaying the relationship between the features of the data (During COVID)',
         fontsize=13)
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2021-12-21T04:41:34.75088Z","iopub.execute_input":"2021-12-21T04:41:34.751248Z","iopub.status.idle":"2021-12-21T04:41:35.426433Z","shell.execute_reply.started":"2021-12-21T04:41:34.751213Z","shell.execute_reply":"2021-12-21T04:41:35.425818Z"}}
plt.figure(figsize=(10,6))
sns.heatmap(dataY.corr(),cmap=plt.cm.Blues,annot=True)
plt.title('Heatmap displaying the relationship between the features of the data (Before COVID)',
         fontsize=13)
plt.show()

# %% [code]
