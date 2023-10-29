#!/usr/bin/env python
# coding: utf-8

# Source codes for Python Machine Learning By Example 4th Edition (Packt Publishing)
# 
# Chapter 12 Making Predictions with Sequences Using Recurrent Neural Networks
# 
# Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)

# # Revisiting stock price forecasting with LSTM

import pandas as pd
import torch
import torch.nn as nn


# Reusing the feature generation function we developed
def generate_features(df):
    """
    Generate features for a stock/index based on historical price and performance
    @param df: dataframe with columns "Open", "Close", "High", "Low", "Volume", "Adj Close"
    @return: dataframe, data set with new features
    """
    df_new = pd.DataFrame()
    # 6 original features
    df_new['open'] = df['Open']
    df_new['open_1'] = df['Open'].shift(1)
    df_new['close_1'] = df['Close'].shift(1)
    df_new['high_1'] = df['High'].shift(1)
    df_new['low_1'] = df['Low'].shift(1)
    df_new['volume_1'] = df['Volume'].shift(1)
    # # 31 generated features
    # # average price
    df_new['avg_price_5'] = df['Close'].rolling(5).mean().shift(1)
    df_new['avg_price_30'] = df['Close'].rolling(21).mean().shift(1)
    df_new['avg_price_365'] = df['Close'].rolling(252).mean().shift(1)
    df_new['ratio_avg_price_5_30'] = df_new['avg_price_5'] / df_new['avg_price_30']
    df_new['ratio_avg_price_5_365'] = df_new['avg_price_5'] / df_new['avg_price_365']
    df_new['ratio_avg_price_30_365'] = df_new['avg_price_30'] / df_new['avg_price_365']
    # # average volume
    df_new['avg_volume_5'] = df['Volume'].rolling(5).mean().shift(1)
    df_new['avg_volume_30'] = df['Volume'].rolling(21).mean().shift(1)
    df_new['avg_volume_365'] = df['Volume'].rolling(252).mean().shift(1)
    df_new['ratio_avg_volume_5_30'] = df_new['avg_volume_5'] / df_new['avg_volume_30']
    df_new['ratio_avg_volume_5_365'] = df_new['avg_volume_5'] / df_new['avg_volume_365']
    df_new['ratio_avg_volume_30_365'] = df_new['avg_volume_30'] / df_new['avg_volume_365']
    # # standard deviation of prices
    df_new['std_price_5'] = df['Close'].rolling(5).std().shift(1)
    df_new['std_price_30'] = df['Close'].rolling(21).std().shift(1)
    df_new['std_price_365'] = df['Close'].rolling(252).std().shift(1)
    df_new['ratio_std_price_5_30'] = df_new['std_price_5'] / df_new['std_price_30']
    df_new['ratio_std_price_5_365'] = df_new['std_price_5'] / df_new['std_price_365']
    df_new['ratio_std_price_30_365'] = df_new['std_price_30'] / df_new['std_price_365']
    # # standard deviation of volumes
    df_new['std_volume_5'] = df['Volume'].rolling(5).std().shift(1)
    df_new['std_volume_30'] = df['Volume'].rolling(21).std().shift(1)
    df_new['std_volume_365'] = df['Volume'].rolling(252).std().shift(1)
    df_new['ratio_std_volume_5_30'] = df_new['std_volume_5'] / df_new['std_volume_30']
    df_new['ratio_std_volume_5_365'] = df_new['std_volume_5'] / df_new['std_volume_365']
    df_new['ratio_std_volume_30_365'] = df_new['std_volume_30'] / df_new['std_volume_365']
    # # # return
    df_new['return_1'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).shift(1)
    df_new['return_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)).shift(1)
    df_new['return_30'] = ((df['Close'] - df['Close'].shift(21)) / df['Close'].shift(21)).shift(1)
    df_new['return_365'] = ((df['Close'] - df['Close'].shift(252)) / df['Close'].shift(252)).shift(1)
    df_new['moving_avg_5'] = df_new['return_1'].rolling(5).mean().shift(1)
    df_new['moving_avg_30'] = df_new['return_1'].rolling(21).mean().shift(1)
    df_new['moving_avg_365'] = df_new['return_1'].rolling(252).mean().shift(1)
    # the target
    df_new['close'] = df['Close']
    df_new = df_new.dropna(axis=0)
    return df_new


data_raw = pd.read_csv('19900101_20230630.csv', index_col='Date')
data = generate_features(data_raw)

start_train = '1990-01-01'
end_train = '2022-12-31'

start_test = '2023-01-01'
end_test = '2023-06-30'

data_train = data.loc[start_train:end_train]
X_train = data_train.drop('close', axis=1).values
y_train = data_train['close'].values

data_test = data.loc[start_test:end_test]
X_test = data_test.drop('close', axis=1).values
y_test = data_test['close'].values


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_scaled_train = torch.FloatTensor(scaler.fit_transform(X_train))
X_scaled_test = torch.FloatTensor(scaler.transform(X_test))

y_train_torch = torch.FloatTensor(y_train)
y_test_torch = torch.FloatTensor(y_test)


# Define a function to create sequences
def create_sequences(data, labels, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = labels[i+seq_length-1]
        sequences.append((seq, label))
    return sequences

    
# Create sequences with a sequence length of 5
seq_length = 5
sequence_train = create_sequences(X_scaled_train, y_train_torch, seq_length)
sequence_test = create_sequences(X_scaled_test, y_test_torch, seq_length)


from torch.utils.data import DataLoader
torch.manual_seed(0)

batch_size = 128  
train_dl = DataLoader(sequence_train, batch_size=batch_size,
                      shuffle=True)


class RNN(nn.Module):
    def __init__(self, input_dim, rnn_hidden_dim, fc_hidden_dim):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, rnn_hidden_dim, 2,
                           batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_dim, fc_hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_dim, 1)

    def forward(self, x):
        out, (hidden, cell) = self.rnn(x)
        out = hidden[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


torch.manual_seed(42)
rnn_hidden_dim = 16
fc_hidden_dim = 16
model = RNN(X_train.shape[1], rnn_hidden_dim, fc_hidden_dim) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)


def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for seq, label in dataloader:
        optimizer.zero_grad()
        pred = model(seq.to(device))[:, 0]
        loss = loss_fn(pred, label.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*label.size(0)
    return total_loss/len(dataloader.dataset)


num_epochs = 1000 
for epoch in range(num_epochs):
    loss = train(model, train_dl, optimizer)
    if epoch % 100 == 0:
        print(f'Epoch {epoch+1} - loss: {loss:.4f}')


predictions, y = [], []
 
for seq, label in sequence_test:
    with torch.no_grad():
        pred = model.cpu()(seq.view(1, seq_length, X_test.shape[1]))[:, 0]
        predictions.append(pred)
        y.append(label)


from sklearn.metrics import r2_score
print(f'R^2: {r2_score(y, predictions):.3f}')


# ---

# Readers may ignore the next cell.

get_ipython().system('jupyter nbconvert --to python ch12_part2.ipynb --TemplateExporter.exclude_input_prompt=True')




