# %% [markdown]
# # Stock Market Analysis using CNN-LSTM model
# This project is about analysis of Stock Market and providing suggestions and predictions to the stockholders. For this, we used CNN-LSTM approach to create a blank model, then use it to train on stock market data. Further implementation is discussed below...

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:27:50.497697Z","iopub.execute_input":"2021-11-30T00:27:50.497978Z","iopub.status.idle":"2021-11-30T00:27:50.597648Z","shell.execute_reply.started":"2021-11-30T00:27:50.497903Z","shell.execute_reply":"2021-11-30T00:27:50.596977Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:27:50.599512Z","iopub.execute_input":"2021-11-30T00:27:50.59981Z","iopub.status.idle":"2021-11-30T00:27:51.386526Z","shell.execute_reply.started":"2021-11-30T00:27:50.599777Z","shell.execute_reply":"2021-11-30T00:27:51.385531Z"}}
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

# %% [code] {"_kg_hide-input":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:27:51.392224Z","iopub.execute_input":"2021-11-30T00:27:51.392524Z","iopub.status.idle":"2021-11-30T00:27:51.412896Z","shell.execute_reply.started":"2021-11-30T00:27:51.392484Z","shell.execute_reply":"2021-11-30T00:27:51.411899Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:27:51.418749Z","iopub.execute_input":"2021-11-30T00:27:51.419144Z","iopub.status.idle":"2021-11-30T00:29:40.058709Z","shell.execute_reply.started":"2021-11-30T00:27:51.41911Z","shell.execute_reply":"2021-11-30T00:29:40.057926Z"}}
cv1 = request_stock_price_list('IBM', 'full', key)
print(cv1.head)
cv1.to_csv('data.csv')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:29:40.060156Z","iopub.execute_input":"2021-11-30T00:29:40.060799Z","iopub.status.idle":"2021-11-30T00:29:40.121198Z","shell.execute_reply.started":"2021-11-30T00:29:40.060757Z","shell.execute_reply":"2021-11-30T00:29:40.120409Z"}}
# For data preprocessing and analysis part
data = pd.read_csv('../input/price-volume-data-for-all-us-stocks-etfs/Stocks/abe.us.txt')
#data = pd.read_csv('../input/nifty50-stock-market-data/COALINDIA.csv')
#data = pd.read_csv('../input/stock-market-data/stock_market_data/nasdaq/csv/ABCO.csv')
#data = pd.read_csv('./data.csv')
# Any CSV or TXT file can be added here....
data.dropna(inplace=True)
data.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:29:40.122598Z","iopub.execute_input":"2021-11-30T00:29:40.122893Z","iopub.status.idle":"2021-11-30T00:29:40.143151Z","shell.execute_reply.started":"2021-11-30T00:29:40.122856Z","shell.execute_reply":"2021-11-30T00:29:40.142393Z"}}
data.info()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:29:40.144532Z","iopub.execute_input":"2021-11-30T00:29:40.145001Z","iopub.status.idle":"2021-11-30T00:29:40.181397Z","shell.execute_reply.started":"2021-11-30T00:29:40.14496Z","shell.execute_reply":"2021-11-30T00:29:40.180645Z"}}
data.describe()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:29:40.182766Z","iopub.execute_input":"2021-11-30T00:29:40.183055Z","iopub.status.idle":"2021-11-30T00:29:40.193634Z","shell.execute_reply.started":"2021-11-30T00:29:40.183022Z","shell.execute_reply":"2021-11-30T00:29:40.192757Z"}}
data.isnull().sum()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:29:40.195386Z","iopub.execute_input":"2021-11-30T00:29:40.196251Z","iopub.status.idle":"2021-11-30T00:29:40.224296Z","shell.execute_reply.started":"2021-11-30T00:29:40.196215Z","shell.execute_reply":"2021-11-30T00:29:40.223475Z"}}
data.reset_index(drop=True, inplace=True)
data.fillna(data.mean(), inplace=True)
data.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:29:40.228014Z","iopub.execute_input":"2021-11-30T00:29:40.228971Z","iopub.status.idle":"2021-11-30T00:29:44.024048Z","shell.execute_reply.started":"2021-11-30T00:29:40.228932Z","shell.execute_reply":"2021-11-30T00:29:44.02076Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:29:44.033492Z","iopub.execute_input":"2021-11-30T00:29:44.034278Z","iopub.status.idle":"2021-11-30T00:29:44.846261Z","shell.execute_reply.started":"2021-11-30T00:29:44.034231Z","shell.execute_reply":"2021-11-30T00:29:44.845554Z"}}
cols_plot = ['Open', 'High', 'Low','Close']
axes = data[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Daily trade')

# %% [markdown]
# Then we'd print the data after making changes and dropping null data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:29:44.847737Z","iopub.execute_input":"2021-11-30T00:29:44.848257Z","iopub.status.idle":"2021-11-30T00:29:45.183855Z","shell.execute_reply.started":"2021-11-30T00:29:44.848222Z","shell.execute_reply":"2021-11-30T00:29:45.183076Z"}}
plt.plot(data['Close'], label="Close price")
plt.xlabel("Timestamp")
plt.ylabel("Closing price")
df = data
print(df)

df.describe().transpose()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:29:45.185259Z","iopub.execute_input":"2021-11-30T00:29:45.185674Z","iopub.status.idle":"2021-11-30T00:29:46.155798Z","shell.execute_reply.started":"2021-11-30T00:29:45.185635Z","shell.execute_reply":"2021-11-30T00:29:46.155005Z"}}
X = data.drop(['Date', 'Close'], axis=1)
Y = data['Close']

X.shape,Y.shape

from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression

lreg = LinearRegression()
sfs1 = sfs(lreg, k_features=2, forward=False, verbose=2, scoring='neg_mean_squared_error')

sfs1 = sfs1.fit(X, Y)

feat_names = list(sfs1.k_feature_names_)
print(feat_names)

# creating a new dataframe using the above variables and adding the target variable
new_data = data[feat_names]
new_data['Close'] = data['Close']

# first five rows of the new data
new_data.head()

new_data.shape, data.shape

df = new_data

# %% [markdown]
# The data has been analysed but it must be converted into data of shape [100,1] to make it easier for CNN to train on... Else it won't select necessary features and the model will fail

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:29:46.157247Z","iopub.execute_input":"2021-11-30T00:29:46.157713Z","iopub.status.idle":"2021-11-30T00:29:55.574983Z","shell.execute_reply.started":"2021-11-30T00:29:46.157657Z","shell.execute_reply":"2021-11-30T00:29:55.574154Z"}}
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
# ## Training part

# %% [markdown]
# This part has 2 subparts: CNN and LSTM
# 
# For CNN, the layers are created with sizes 64,128,64. In every layer, TimeDistributed function is added to track the features with respect to time. In between them, Pooling layers are added.
# 
# After that, it's passed to Bi-LSTM layers

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:29:55.576169Z","iopub.execute_input":"2021-11-30T00:29:55.577005Z","iopub.status.idle":"2021-11-30T00:30:42.387455Z","shell.execute_reply.started":"2021-11-30T00:29:55.576965Z","shell.execute_reply":"2021-11-30T00:30:42.386656Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:30:42.389187Z","iopub.execute_input":"2021-11-30T00:30:42.389874Z","iopub.status.idle":"2021-11-30T00:30:42.711243Z","shell.execute_reply.started":"2021-11-30T00:30:42.389835Z","shell.execute_reply":"2021-11-30T00:30:42.710554Z"}}
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:30:42.71256Z","iopub.execute_input":"2021-11-30T00:30:42.713097Z","iopub.status.idle":"2021-11-30T00:30:43.032936Z","shell.execute_reply.started":"2021-11-30T00:30:42.713059Z","shell.execute_reply":"2021-11-30T00:30:43.032156Z"}}
plt.plot(history.history['mse'], label='train mse')
plt.plot(history.history['val_mse'], label='val mse')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:30:43.034562Z","iopub.execute_input":"2021-11-30T00:30:43.035018Z","iopub.status.idle":"2021-11-30T00:30:43.356383Z","shell.execute_reply.started":"2021-11-30T00:30:43.034981Z","shell.execute_reply":"2021-11-30T00:30:43.355742Z"}}
plt.plot(history.history['mae'], label='train mae')
plt.plot(history.history['val_mae'], label='val mae')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:30:43.357648Z","iopub.execute_input":"2021-11-30T00:30:43.358041Z","iopub.status.idle":"2021-11-30T00:30:44.323425Z","shell.execute_reply.started":"2021-11-30T00:30:43.358006Z","shell.execute_reply":"2021-11-30T00:30:44.322588Z"}}
# After the model has been constructed, we need to train
from tensorflow.keras.utils import plot_model
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:30:44.325061Z","iopub.execute_input":"2021-11-30T00:30:44.325946Z","iopub.status.idle":"2021-11-30T00:30:44.494354Z","shell.execute_reply.started":"2021-11-30T00:30:44.325907Z","shell.execute_reply":"2021-11-30T00:30:44.493555Z"}}
model.evaluate(test_X, test_Y)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:30:44.49592Z","iopub.execute_input":"2021-11-30T00:30:44.496435Z","iopub.status.idle":"2021-11-30T00:30:45.842253Z","shell.execute_reply.started":"2021-11-30T00:30:44.496399Z","shell.execute_reply":"2021-11-30T00:30:45.841411Z"}}
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import max_error

# predict probabilities for test set
yhat_probs = model.predict(test_X, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(test_X, verbose=0)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]

var = explained_variance_score(test_Y.reshape(-1,1), yhat_probs)
print('Variance: %f' % var)

r2 = r2_score(test_Y.reshape(-1,1), yhat_probs)
print('R2 Score: %f' % var)

var2 = max_error(test_Y.reshape(-1,1), yhat_probs)
print('Max Error: %f' % var2)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:30:45.843592Z","iopub.execute_input":"2021-11-30T00:30:45.844369Z","iopub.status.idle":"2021-11-30T00:30:46.252205Z","shell.execute_reply.started":"2021-11-30T00:30:45.844327Z","shell.execute_reply":"2021-11-30T00:30:46.251541Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:30:46.253495Z","iopub.execute_input":"2021-11-30T00:30:46.254343Z","iopub.status.idle":"2021-11-30T00:30:46.330456Z","shell.execute_reply.started":"2021-11-30T00:30:46.254307Z","shell.execute_reply":"2021-11-30T00:30:46.329751Z"}}
# First we need to save a model
model.save("model.h5")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:30:46.331751Z","iopub.execute_input":"2021-11-30T00:30:46.332024Z","iopub.status.idle":"2021-11-30T00:30:47.61497Z","shell.execute_reply.started":"2021-11-30T00:30:46.331991Z","shell.execute_reply":"2021-11-30T00:30:47.614193Z"}}
# Load model
new_model = tf.keras.models.load_model("./model.h5")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:30:47.619112Z","iopub.execute_input":"2021-11-30T00:30:47.619389Z","iopub.status.idle":"2021-11-30T00:30:47.634084Z","shell.execute_reply.started":"2021-11-30T00:30:47.619355Z","shell.execute_reply":"2021-11-30T00:30:47.63318Z"}}
new_model.summary()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:30:47.635384Z","iopub.execute_input":"2021-11-30T00:30:47.635817Z","iopub.status.idle":"2021-11-30T00:31:04.609129Z","shell.execute_reply.started":"2021-11-30T00:30:47.635781Z","shell.execute_reply":"2021-11-30T00:31:04.607737Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:31:04.610441Z","iopub.execute_input":"2021-11-30T00:31:04.610867Z","iopub.status.idle":"2021-11-30T00:31:05.077092Z","shell.execute_reply.started":"2021-11-30T00:31:04.610822Z","shell.execute_reply":"2021-11-30T00:31:05.076072Z"}}
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

# %% [markdown]
# # EDA

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-30T00:41:04.131327Z","iopub.execute_input":"2021-11-30T00:41:04.131641Z","iopub.status.idle":"2021-11-30T00:41:04.167147Z","shell.execute_reply.started":"2021-11-30T00:41:04.131612Z","shell.execute_reply":"2021-11-30T00:41:04.166248Z"}}
dataX = pd.read_csv('./data.csv')
dataY = pd.read_csv('./data.csv')
dataX.info()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T00:41:07.441649Z","iopub.execute_input":"2021-11-30T00:41:07.442353Z","iopub.status.idle":"2021-11-30T00:41:07.458392Z","shell.execute_reply.started":"2021-11-30T00:41:07.442321Z","shell.execute_reply":"2021-11-30T00:41:07.457719Z"}}
dataX.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T00:41:10.802318Z","iopub.execute_input":"2021-11-30T00:41:10.803193Z","iopub.status.idle":"2021-11-30T00:41:10.838462Z","shell.execute_reply.started":"2021-11-30T00:41:10.803154Z","shell.execute_reply":"2021-11-30T00:41:10.837748Z"}}
start_date = '2020-01-01'
end_date = '2021-11-29'

start = '2018-01-01'
end = '2020-01-01'

fill = (dataX['date']>=start_date) & (dataX['date']<=end_date)
dataX = dataX.loc[fill]
dataX


# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T00:41:13.838014Z","iopub.execute_input":"2021-11-30T00:41:13.838274Z","iopub.status.idle":"2021-11-30T00:41:13.870051Z","shell.execute_reply.started":"2021-11-30T00:41:13.838249Z","shell.execute_reply":"2021-11-30T00:41:13.869107Z"}}
fill2 = (dataY['date']>=start) & (dataY['date']<=end)
dataY = dataY.loc[fill2]
dataY

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T00:41:17.009971Z","iopub.execute_input":"2021-11-30T00:41:17.010531Z","iopub.status.idle":"2021-11-30T00:41:17.051379Z","shell.execute_reply.started":"2021-11-30T00:41:17.010474Z","shell.execute_reply":"2021-11-30T00:41:17.050509Z"}}
dataX.describe()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T00:41:19.598566Z","iopub.execute_input":"2021-11-30T00:41:19.599179Z","iopub.status.idle":"2021-11-30T00:41:19.644959Z","shell.execute_reply.started":"2021-11-30T00:41:19.599141Z","shell.execute_reply":"2021-11-30T00:41:19.64422Z"}}
dataY.describe()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T00:42:27.313943Z","iopub.execute_input":"2021-11-30T00:42:27.314244Z","iopub.status.idle":"2021-11-30T00:42:27.670252Z","shell.execute_reply.started":"2021-11-30T00:42:27.314215Z","shell.execute_reply":"2021-11-30T00:42:27.669576Z"}}
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import r2_score,mean_squared_error

sns_plot = sns.distplot(dataX['close'])
sns_plot2 = sns.distplot(dataY['close'])

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T00:44:44.730718Z","iopub.execute_input":"2021-11-30T00:44:44.73141Z","iopub.status.idle":"2021-11-30T00:44:46.789448Z","shell.execute_reply.started":"2021-11-30T00:44:44.731376Z","shell.execute_reply":"2021-11-30T00:44:46.788715Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T00:45:22.898902Z","iopub.execute_input":"2021-11-30T00:45:22.899712Z","iopub.status.idle":"2021-11-30T00:45:24.913058Z","shell.execute_reply.started":"2021-11-30T00:45:22.899676Z","shell.execute_reply":"2021-11-30T00:45:24.912176Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T00:48:14.241054Z","iopub.execute_input":"2021-11-30T00:48:14.242004Z","iopub.status.idle":"2021-11-30T00:48:14.913312Z","shell.execute_reply.started":"2021-11-30T00:48:14.241963Z","shell.execute_reply":"2021-11-30T00:48:14.910242Z"}}
plt.figure(figsize=(10,6))
sns.heatmap(dataX.corr(),cmap=plt.cm.Reds,annot=True)
plt.title('Heatmap displaying the relationship between the features of the data (During COVID)',
         fontsize=13)
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-30T00:48:39.280052Z","iopub.execute_input":"2021-11-30T00:48:39.280311Z","iopub.status.idle":"2021-11-30T00:48:39.872445Z","shell.execute_reply.started":"2021-11-30T00:48:39.280284Z","shell.execute_reply":"2021-11-30T00:48:39.87159Z"}}
plt.figure(figsize=(10,6))
sns.heatmap(dataY.corr(),cmap=plt.cm.Blues,annot=True)
plt.title('Heatmap displaying the relationship between the features of the data (Before COVID)',
         fontsize=13)
plt.show()

# %% [code]
