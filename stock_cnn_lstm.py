# %% [markdown]
# # Stock Market Analysis using CNN-LSTM model
# This project is about analysis of Stock Market and providing suggestions and predictions to the stockholders. For this, we used CNN-LSTM approach to create a blank model, then use it to train on stock market data. Further implementation is discussed below...

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:47:50.820187Z","iopub.execute_input":"2021-11-29T11:47:50.820801Z","iopub.status.idle":"2021-11-29T11:47:50.825269Z","shell.execute_reply.started":"2021-11-29T11:47:50.820743Z","shell.execute_reply":"2021-11-29T11:47:50.824484Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:47:50.835758Z","iopub.execute_input":"2021-11-29T11:47:50.835970Z","iopub.status.idle":"2021-11-29T11:47:50.848974Z","shell.execute_reply.started":"2021-11-29T11:47:50.835948Z","shell.execute_reply":"2021-11-29T11:47:50.848218Z"}}
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

# %% [code] {"_kg_hide-input":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:47:50.852563Z","iopub.execute_input":"2021-11-29T11:47:50.852877Z","iopub.status.idle":"2021-11-29T11:47:50.860831Z","shell.execute_reply.started":"2021-11-29T11:47:50.852851Z","shell.execute_reply":"2021-11-29T11:47:50.859967Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:47:50.865172Z","iopub.execute_input":"2021-11-29T11:47:50.865830Z","iopub.status.idle":"2021-11-29T11:49:35.584715Z","shell.execute_reply.started":"2021-11-29T11:47:50.865795Z","shell.execute_reply":"2021-11-29T11:49:35.583887Z"}}
cv1 = request_stock_price_list('IBM', 'full', key)
print(cv1.head)
cv1.to_csv('data.csv')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:49:35.586527Z","iopub.execute_input":"2021-11-29T11:49:35.586830Z","iopub.status.idle":"2021-11-29T11:49:35.613479Z","shell.execute_reply.started":"2021-11-29T11:49:35.586795Z","shell.execute_reply":"2021-11-29T11:49:35.612687Z"}}
# For data preprocessing and analysis part
data = pd.read_csv('../input/price-volume-data-for-all-us-stocks-etfs/Stocks/abe.us.txt')
#data = pd.read_csv('../input/nifty50-stock-market-data/COALINDIA.csv')
#data = pd.read_csv('../input/stock-market-data/stock_market_data/nasdaq/csv/ABCO.csv')
#data = pd.read_csv('./data.csv')
# Any CSV or TXT file can be added here....
data.dropna(inplace=True)
data.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:49:35.614531Z","iopub.execute_input":"2021-11-29T11:49:35.615131Z","iopub.status.idle":"2021-11-29T11:49:35.630304Z","shell.execute_reply.started":"2021-11-29T11:49:35.615094Z","shell.execute_reply":"2021-11-29T11:49:35.629643Z"}}
data.info()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:49:35.632354Z","iopub.execute_input":"2021-11-29T11:49:35.633032Z","iopub.status.idle":"2021-11-29T11:49:35.660603Z","shell.execute_reply.started":"2021-11-29T11:49:35.632997Z","shell.execute_reply":"2021-11-29T11:49:35.659673Z"}}
data.describe()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:49:35.661816Z","iopub.execute_input":"2021-11-29T11:49:35.662061Z","iopub.status.idle":"2021-11-29T11:49:35.671552Z","shell.execute_reply.started":"2021-11-29T11:49:35.662029Z","shell.execute_reply":"2021-11-29T11:49:35.670543Z"}}
data.isnull().sum()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:49:35.673276Z","iopub.execute_input":"2021-11-29T11:49:35.673525Z","iopub.status.idle":"2021-11-29T11:49:35.693157Z","shell.execute_reply.started":"2021-11-29T11:49:35.673493Z","shell.execute_reply":"2021-11-29T11:49:35.692377Z"}}
data.reset_index(drop=True, inplace=True)
data.fillna(data.mean(), inplace=True)
data.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:49:35.694376Z","iopub.execute_input":"2021-11-29T11:49:35.694685Z","iopub.status.idle":"2021-11-29T11:49:38.978148Z","shell.execute_reply.started":"2021-11-29T11:49:35.694650Z","shell.execute_reply":"2021-11-29T11:49:38.977468Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:49:38.979375Z","iopub.execute_input":"2021-11-29T11:49:38.979961Z","iopub.status.idle":"2021-11-29T11:49:39.668971Z","shell.execute_reply.started":"2021-11-29T11:49:38.979925Z","shell.execute_reply":"2021-11-29T11:49:39.668283Z"}}
cols_plot = ['Open', 'High', 'Low','Close']
axes = data[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Daily trade')

# %% [markdown]
# Then we'd print the data after making changes and dropping null data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:49:39.669994Z","iopub.execute_input":"2021-11-29T11:49:39.670363Z","iopub.status.idle":"2021-11-29T11:49:39.982338Z","shell.execute_reply.started":"2021-11-29T11:49:39.670329Z","shell.execute_reply":"2021-11-29T11:49:39.981674Z"}}
plt.plot(data['Close'], label="Close price")
plt.xlabel("Timestamp")
plt.ylabel("Closing price")
df = data
print(df)

df.describe().transpose()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:49:39.985253Z","iopub.execute_input":"2021-11-29T11:49:39.985471Z","iopub.status.idle":"2021-11-29T11:49:40.575017Z","shell.execute_reply.started":"2021-11-29T11:49:39.985440Z","shell.execute_reply":"2021-11-29T11:49:40.574242Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:49:40.576159Z","iopub.execute_input":"2021-11-29T11:49:40.576423Z","iopub.status.idle":"2021-11-29T11:49:49.686255Z","shell.execute_reply.started":"2021-11-29T11:49:40.576386Z","shell.execute_reply":"2021-11-29T11:49:49.685428Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:49:49.687430Z","iopub.execute_input":"2021-11-29T11:49:49.689134Z","iopub.status.idle":"2021-11-29T11:50:22.041867Z","shell.execute_reply.started":"2021-11-29T11:49:49.689090Z","shell.execute_reply":"2021-11-29T11:50:22.041168Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:50:22.043225Z","iopub.execute_input":"2021-11-29T11:50:22.043985Z","iopub.status.idle":"2021-11-29T11:50:22.346441Z","shell.execute_reply.started":"2021-11-29T11:50:22.043934Z","shell.execute_reply":"2021-11-29T11:50:22.345646Z"}}
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:50:22.347661Z","iopub.execute_input":"2021-11-29T11:50:22.347924Z","iopub.status.idle":"2021-11-29T11:50:22.645362Z","shell.execute_reply.started":"2021-11-29T11:50:22.347891Z","shell.execute_reply":"2021-11-29T11:50:22.644686Z"}}
plt.plot(history.history['mse'], label='train mse')
plt.plot(history.history['val_mse'], label='val mse')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:50:22.646463Z","iopub.execute_input":"2021-11-29T11:50:22.646708Z","iopub.status.idle":"2021-11-29T11:50:22.953977Z","shell.execute_reply.started":"2021-11-29T11:50:22.646675Z","shell.execute_reply":"2021-11-29T11:50:22.953210Z"}}
plt.plot(history.history['mae'], label='train mae')
plt.plot(history.history['val_mae'], label='val mae')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:50:22.955289Z","iopub.execute_input":"2021-11-29T11:50:22.955728Z","iopub.status.idle":"2021-11-29T11:50:23.226515Z","shell.execute_reply.started":"2021-11-29T11:50:22.955689Z","shell.execute_reply":"2021-11-29T11:50:23.225682Z"}}
# After the model has been constructed, we need to train
from tensorflow.keras.utils import plot_model
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:50:23.228065Z","iopub.execute_input":"2021-11-29T11:50:23.228669Z","iopub.status.idle":"2021-11-29T11:50:23.361634Z","shell.execute_reply.started":"2021-11-29T11:50:23.228625Z","shell.execute_reply":"2021-11-29T11:50:23.361001Z"}}
model.evaluate(test_X, test_Y)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-29T11:50:23.363126Z","iopub.execute_input":"2021-11-29T11:50:23.363411Z","iopub.status.idle":"2021-11-29T11:50:24.647024Z","shell.execute_reply.started":"2021-11-29T11:50:23.363373Z","shell.execute_reply":"2021-11-29T11:50:24.644729Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:50:24.648545Z","iopub.execute_input":"2021-11-29T11:50:24.649089Z","iopub.status.idle":"2021-11-29T11:50:25.043323Z","shell.execute_reply.started":"2021-11-29T11:50:24.649053Z","shell.execute_reply":"2021-11-29T11:50:25.042654Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:50:25.044727Z","iopub.execute_input":"2021-11-29T11:50:25.045508Z","iopub.status.idle":"2021-11-29T11:50:25.117068Z","shell.execute_reply.started":"2021-11-29T11:50:25.045471Z","shell.execute_reply":"2021-11-29T11:50:25.116430Z"}}
# First we need to save a model
model.save("model.h5")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:50:25.118216Z","iopub.execute_input":"2021-11-29T11:50:25.118476Z","iopub.status.idle":"2021-11-29T11:50:26.089270Z","shell.execute_reply.started":"2021-11-29T11:50:25.118440Z","shell.execute_reply":"2021-11-29T11:50:26.088429Z"}}
# Load model
new_model = tf.keras.models.load_model("./model.h5")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:50:26.090741Z","iopub.execute_input":"2021-11-29T11:50:26.091061Z","iopub.status.idle":"2021-11-29T11:50:26.104857Z","shell.execute_reply.started":"2021-11-29T11:50:26.091023Z","shell.execute_reply":"2021-11-29T11:50:26.104130Z"}}
new_model.summary()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:50:26.106061Z","iopub.execute_input":"2021-11-29T11:50:26.106487Z","iopub.status.idle":"2021-11-29T11:50:42.645240Z","shell.execute_reply.started":"2021-11-29T11:50:26.106450Z","shell.execute_reply":"2021-11-29T11:50:42.643860Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-29T11:50:42.646454Z","iopub.execute_input":"2021-11-29T11:50:42.647286Z","iopub.status.idle":"2021-11-29T11:50:43.111315Z","shell.execute_reply.started":"2021-11-29T11:50:42.647246Z","shell.execute_reply":"2021-11-29T11:50:43.110619Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false}}
