# Stock-X

![License](https://img.shields.io/github/license/Circle-1/Stock-X)
![Stars](https://img.shields.io/github/stars/Circle-1/Stock-X)
![Release](https://img.shields.io/github/v/release/Circle-1/Stock-X)
[![Heroku](https://img.shields.io/badge/Heroku-Active-blue?logo=heroku)](https://stock-x-proj.herokuapp.com/)

This project is all about analysis of Stock Market and providing suggestions to stockholders to invest in right company

Note: The notebook used here (IPYNB) is made using Kaggle, a data-science and ML community website which provides free Jupyter Notebook environment to work on programs and GPUs and TPUs to work on Neural Networks easily.

Here's the ref link to [Kaggle](https://www.kaggle.com/)

Notebook link for CNN-LSTM: [Click here](https://www.kaggle.com/aadhityaa/stock-cnn-lstm)

Docker Image link (contains bundled libraries): [Click here](https://hub.docker.com/r/aerox86/stock-x) ![Size](https://img.shields.io/docker/image-size/aerox86/stock-x/latest-stable)

Helm charts: [![Artifact Hub](https://img.shields.io/endpoint?url=https://artifacthub.io/badge/repository/stock-x)](https://artifacthub.io/packages/search?repo=stock-x)

## Libraries used:
    - Tensorflow
    - Keras
    - Pandas
    - Scikit-learn
    - Matplotlib
    - Seaborn

## Neural Network type

Here CNN (with Time Distributed function) and Bi-LSTM combined Neural Network is used to train. Other algorithms like XGBoost, RNN-LSTM, LSTM-GRU are also added for comparison. Here are the links to view the notebooks directly. You can also view the results in the app created using [Mercury](https://mljar.com/mercury/) which is deployed over [Heroku (free dyno)](https://stock-x-proj.herokuapp.com/).

 - [CNN-LSTM](stock-market-prediction-using-cnn-lstm.ipynb)
 - [LSTM-GRU](lstm_gru_model.ipynb)
 - [RNN-LSTM](RNN-LSTM.ipynb)
 - [XGBoost](regressor-model.ipynb)
