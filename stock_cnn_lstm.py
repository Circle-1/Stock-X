# %% [markdown]
# # Stock-CNN-LSTM
# This project is about analysis of Stock Market and providing suggestions and predictions to the stockholders. For this, we used CNN-LSTM approach to create a blank model, then use it to train on stock market data. Further implementation is discussed below...

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-10-23T06:18:26.092077Z","iopub.execute_input":"2021-10-23T06:18:26.092651Z","iopub.status.idle":"2021-10-23T06:18:26.191618Z","shell.execute_reply.started":"2021-10-23T06:18:26.092561Z","shell.execute_reply":"2021-10-23T06:18:26.190971Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-10-23T06:18:26.193297Z","iopub.execute_input":"2021-10-23T06:18:26.193802Z","iopub.status.idle":"2021-10-23T06:18:26.899435Z","shell.execute_reply.started":"2021-10-23T06:18:26.193767Z","shell.execute_reply":"2021-10-23T06:18:26.898685Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-10-23T06:18:26.900670Z","iopub.execute_input":"2021-10-23T06:18:26.900948Z","iopub.status.idle":"2021-10-23T06:18:26.962090Z","shell.execute_reply.started":"2021-10-23T06:18:26.900917Z","shell.execute_reply":"2021-10-23T06:18:26.961408Z"}}
# For data preprocessing and analysis part
data = pd.read_csv('../input/price-volume-data-for-all-us-stocks-etfs/Stocks/abe.us.txt')
# Any CSV or TXT file can be added here....
data.dropna(inplace=True)
data.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-10-23T06:18:26.964152Z","iopub.execute_input":"2021-10-23T06:18:26.964591Z","iopub.status.idle":"2021-10-23T06:18:26.985851Z","shell.execute_reply.started":"2021-10-23T06:18:26.964552Z","shell.execute_reply":"2021-10-23T06:18:26.985156Z"}}
data.info()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-10-23T06:18:26.986927Z","iopub.execute_input":"2021-10-23T06:18:26.987652Z","iopub.status.idle":"2021-10-23T06:18:27.019524Z","shell.execute_reply.started":"2021-10-23T06:18:26.987613Z","shell.execute_reply":"2021-10-23T06:18:27.018653Z"}}
data.describe()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-10-23T06:18:27.020797Z","iopub.execute_input":"2021-10-23T06:18:27.021048Z","iopub.status.idle":"2021-10-23T06:18:27.030252Z","shell.execute_reply.started":"2021-10-23T06:18:27.021016Z","shell.execute_reply":"2021-10-23T06:18:27.029322Z"}}
data.isnull().sum()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-10-23T06:18:27.031614Z","iopub.execute_input":"2021-10-23T06:18:27.031878Z","iopub.status.idle":"2021-10-23T06:18:27.053046Z","shell.execute_reply.started":"2021-10-23T06:18:27.031846Z","shell.execute_reply":"2021-10-23T06:18:27.052322Z"}}
data.reset_index(drop=True, inplace=True)
data.fillna(data.mean(), inplace=True)
data.head()

# %% [markdown]
# After that, we'll visualize the data for understanding, this is shown below...

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-10-23T06:18:27.054515Z","iopub.execute_input":"2021-10-23T06:18:27.054943Z","iopub.status.idle":"2021-10-23T06:18:27.878823Z","shell.execute_reply.started":"2021-10-23T06:18:27.054911Z","shell.execute_reply":"2021-10-23T06:18:27.878126Z"}}
cols_plot = ['Open', 'Close', 'High','Low']
axes = data[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Daily trade')

# %% [markdown]
# Then we'd print the data after making changes and dropping null data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-10-23T06:18:27.879749Z","iopub.execute_input":"2021-10-23T06:18:27.879969Z","iopub.status.idle":"2021-10-23T06:18:28.257317Z","shell.execute_reply.started":"2021-10-23T06:18:27.879938Z","shell.execute_reply":"2021-10-23T06:18:28.256420Z"}}
plt.plot(data['Close'], label="Close price")
plt.xlabel("Timestamp")
plt.ylabel("Closing price")

df = data.drop('Date', axis=1)
print(df)

# %% [markdown]
# The data has been analysed but it must be converted into data of shape [100,1] to make it easier for CNN to train on... Else it won't select necessary features and the model will fail

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-10-23T06:18:28.262176Z","iopub.execute_input":"2021-10-23T06:18:28.262621Z","iopub.status.idle":"2021-10-23T06:18:37.804188Z","shell.execute_reply.started":"2021-10-23T06:18:28.262589Z","shell.execute_reply":"2021-10-23T06:18:37.803434Z"}}
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

# %% [markdown]
# This part has 2 subparts: CNN and LSTM
# 
# For CNN, the layers are created with sizes 64,128,64. In every layer, TimeDistributed function is added to track the features with respect to time. In between them, Pooling layers are added.
# 
# Then a dense layer of 5 neurons with L1 Kernel regularizer is added
# 
# After that, it's passed to Bi-LSTM layers

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-10-23T06:18:37.805459Z","iopub.execute_input":"2021-10-23T06:18:37.805708Z","iopub.status.idle":"2021-10-23T06:19:09.122393Z","shell.execute_reply.started":"2021-10-23T06:18:37.805677Z","shell.execute_reply":"2021-10-23T06:19:09.121610Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-10-23T06:19:09.124398Z","iopub.execute_input":"2021-10-23T06:19:09.124662Z","iopub.status.idle":"2021-10-23T06:19:10.123501Z","shell.execute_reply.started":"2021-10-23T06:19:09.124627Z","shell.execute_reply":"2021-10-23T06:19:10.122754Z"}}
# After the model has been constructed, we need to train
from tensorflow.keras.utils import plot_model
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-10-23T06:19:10.125217Z","iopub.execute_input":"2021-10-23T06:19:10.125593Z","iopub.status.idle":"2021-10-23T06:19:10.429445Z","shell.execute_reply.started":"2021-10-23T06:19:10.125559Z","shell.execute_reply":"2021-10-23T06:19:10.428534Z"}}
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-10-23T06:19:10.430777Z","iopub.execute_input":"2021-10-23T06:19:10.431034Z","iopub.status.idle":"2021-10-23T06:19:10.597644Z","shell.execute_reply.started":"2021-10-23T06:19:10.430998Z","shell.execute_reply":"2021-10-23T06:19:10.596948Z"}}
model.evaluate(test_X, test_Y)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-10-23T06:19:10.599090Z","iopub.execute_input":"2021-10-23T06:19:10.599368Z","iopub.status.idle":"2021-10-23T06:19:12.195334Z","shell.execute_reply.started":"2021-10-23T06:19:10.599333Z","shell.execute_reply":"2021-10-23T06:19:12.194624Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2021-10-23T06:19:12.196791Z","iopub.execute_input":"2021-10-23T06:19:12.197255Z","iopub.status.idle":"2021-10-23T06:19:15.436003Z","shell.execute_reply.started":"2021-10-23T06:19:12.197218Z","shell.execute_reply":"2021-10-23T06:19:15.435137Z"}}
pip freeze > requirements.txt

# %% [markdown]
# ## Testing part

# %% [markdown]
# In this part, the model is saved and loaded back again. Then, it's made to train again but with different data to check it's loss and prediction

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-10-23T06:19:15.437410Z","iopub.execute_input":"2021-10-23T06:19:15.437703Z","iopub.status.idle":"2021-10-23T06:19:15.569305Z","shell.execute_reply.started":"2021-10-23T06:19:15.437665Z","shell.execute_reply":"2021-10-23T06:19:15.568535Z"}}
# First we need to save a model
model.save("model.h5")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-10-23T06:19:15.573797Z","iopub.execute_input":"2021-10-23T06:19:15.575747Z","iopub.status.idle":"2021-10-23T06:19:16.782492Z","shell.execute_reply.started":"2021-10-23T06:19:15.575694Z","shell.execute_reply":"2021-10-23T06:19:16.781727Z"}}
# Load model
new_model = tf.keras.models.load_model("./model.h5")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-10-23T06:19:16.783982Z","iopub.execute_input":"2021-10-23T06:19:16.784250Z","iopub.status.idle":"2021-10-23T06:19:16.799449Z","shell.execute_reply.started":"2021-10-23T06:19:16.784216Z","shell.execute_reply":"2021-10-23T06:19:16.798369Z"}}
new_model.summary()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-10-23T06:19:16.800701Z","iopub.execute_input":"2021-10-23T06:19:16.800957Z","iopub.status.idle":"2021-10-23T06:19:19.594305Z","shell.execute_reply.started":"2021-10-23T06:19:16.800924Z","shell.execute_reply":"2021-10-23T06:19:19.593558Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2021-10-23T06:19:19.595462Z","iopub.execute_input":"2021-10-23T06:19:19.595872Z","iopub.status.idle":"2021-10-23T06:19:28.282361Z","shell.execute_reply.started":"2021-10-23T06:19:19.595836Z","shell.execute_reply":"2021-10-23T06:19:28.281582Z"}}
new_model.fit(train_X, train_Y, validation_data=(test_X,test_Y), epochs=25,batch_size=64, verbose=1, shuffle =True)

# %% [code] {"execution":{"iopub.status.busy":"2021-10-23T06:19:28.283649Z","iopub.execute_input":"2021-10-23T06:19:28.284001Z","iopub.status.idle":"2021-10-23T06:19:28.631170Z","shell.execute_reply.started":"2021-10-23T06:19:28.283964Z","shell.execute_reply":"2021-10-23T06:19:28.630429Z"}}
predicted  = model.predict(test_X)
test_label = (test_Y)
predicted = np.array(predicted[:,0]).reshape(-1,1)
len_t = len(train_X)
for j in range(len_t , len_t + len(test_X)):
    temp =data.iloc[j,3]
    test_label[j - len_t] = test_label[j - len_t] * temp + temp
    predicted[j - len_t] = predicted[j - len_t] * temp + temp
plt.plot(predicted, color = 'green', label = 'Predicted Stock Price')
plt.title(' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()