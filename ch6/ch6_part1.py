#!/usr/bin/env python
# coding: utf-8

# 
# Source codes for Python Machine Learning By Example 4th Edition (Packt Publishing)
# 
# Chapter 6 Predicting Stock Prices with Artificial Neural Networks
# 
# Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
# 

# # Demystifying neural networks 

# ## Starting with a single-layer neural network

# ### Layers in neural networks

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

z = np.linspace(-8, 8, 1000)
y = sigmoid(z)
plt.plot(z, y)
plt.xlabel('z')
plt.ylabel('y(z)')
plt.title('logistic')
plt.grid()
plt.show()


def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

z = np.linspace(-8, 8, 1000)
y = tanh(z)
plt.plot(z, y)
plt.xlabel('z')
plt.ylabel('y(z)')
plt.title('tanh')
plt.grid()
plt.show()


def relu(z):
    return np.maximum(np.zeros_like(z), z)


z = np.linspace(-8, 8, 1000)
y = relu(z)
plt.plot(z, y)
plt.xlabel('z')
plt.ylabel('y(z)')
plt.title('relu')
plt.grid()
plt.show()


# # Building neural networks 

# ## Implementing neural networks from scratch 

def sigmoid_derivative(z):
    return sigmoid(z) * (1.0 - sigmoid(z))


def train(X, y, n_hidden, learning_rate, n_iter):
    m, n_input = X.shape
    W1 = np.random.randn(n_input, n_hidden)
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, 1)
    b2 = np.zeros((1, 1))
    for i in range(1, n_iter+1):
        Z2 = np.matmul(X, W1) + b1
        A2 = sigmoid(Z2)
        Z3 = np.matmul(A2, W2) + b2
        A3 = Z3

        dZ3 = A3 - y
        dW2 = np.matmul(A2.T, dZ3)
        db2 = np.sum(dZ3, axis=0, keepdims=True)

        dZ2 = np.matmul(dZ3, W2.T) * sigmoid_derivative(Z2)
        dW1 = np.matmul(X.T, dZ2)
        db1 = np.sum(dZ2, axis=0)

        W2 = W2 - learning_rate * dW2 / m
        b2 = b2 - learning_rate * db2 / m
        W1 = W1 - learning_rate * dW1 / m
        b1 = b1 - learning_rate * db1 / m

        if i % 100 == 0:
            cost = np.mean((y - A3) ** 2)
            print('Iteration %i, training loss: %f' % (i, cost))

    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model


from sklearn import datasets
housing = datasets.fetch_california_housing()

num_test = 10  # the last 10 samples as testing set

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()

X_train = housing.data[:-num_test, :]
X_train = scaler.fit_transform(X_train)
y_train = housing.target[:-num_test].reshape(-1, 1)
X_test = housing.data[-num_test:, :]
X_test = scaler.transform(X_test)
y_test = housing.target[-num_test:]


n_hidden = 20
learning_rate = 0.1
n_iter = 2000

model = train(X_train, y_train, n_hidden, learning_rate, n_iter)


def predict(x, model):
    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']
    A2 = sigmoid(np.matmul(x, W1) + b1)
    A3 = np.matmul(A2, W2) + b2
    return A3


predictions = predict(X_test, model)
print(predictions[:, 0])
print(y_test)


# ## Implementing neural networks with scikit-learn 

from sklearn.neural_network import MLPRegressor
nn_scikit = MLPRegressor(hidden_layer_sizes=(16, 8), 
                         activation='relu', 
                         solver='adam',
                         learning_rate_init=0.001, 
                         random_state=42, 
                         max_iter=2000)


nn_scikit.fit(X_train, y_train.ravel())
predictions = nn_scikit.predict(X_test)
print(predictions)


from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, predictions))


# ## Implementing neural networks with TensorFlow

import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)


model = keras.Sequential([
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=8, activation='relu'),
    keras.layers.Dense(units=1)
])


model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.01))


model.fit(X_train, y_train, epochs=300)


predictions = model.predict(X_test)[:, 0]
print(predictions)

print(mean_squared_error(y_test, predictions))


# ## Implementing neural networks with PyTorch

import torch
import torch.nn as nn


torch.manual_seed(42)
model = nn.Sequential(nn.Linear(X_train.shape[1], 16),
                      nn.ReLU(),
                      nn.Linear(16, 8),
                      nn.ReLU(),
                      nn.Linear(8, 1))


loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


X_train_torch = torch.from_numpy(X_train.astype(np.float32))
y_train_torch = torch.from_numpy(y_train.astype(np.float32))


def train_step(model, X_train, y_train, loss_function, optimizer):
    pred_train = model(X_train)
    loss = loss_function(pred_train, y_train)
 
    model.zero_grad()
    loss.backward()

    optimizer.step()
    
    return loss.item()


for epoch in range(500):
    loss = train_step(model, X_train_torch, y_train_torch, loss_function, optimizer)

    if epoch % 100 == 0:
        print(f"Epoch {epoch} - loss: {loss}")
        


X_test_torch = torch.from_numpy(X_test.astype(np.float32))
predictions = model(X_test_torch).detach().numpy()[:, 0]
print(predictions)

print(mean_squared_error(y_test, predictions))


# # Preventing overfitting in neural networks 

# ## Dropout

torch.manual_seed(42)
model_with_dropout = nn.Sequential(nn.Linear(X_train.shape[1], 16),
                                   nn.ReLU(),
                                   nn.Dropout(0.1),
                                   nn.Linear(16, 8),
                                   nn.ReLU(),
                                   nn.Linear(8, 1))


optimizer = torch.optim.Adam(model_with_dropout.parameters(), lr=0.01)


for epoch in range(1000):
    loss = train_step(model_with_dropout, X_train_torch, y_train_torch, loss_function, optimizer)

    if epoch % 100 == 0:
        print(f"Epoch {epoch} - loss: {loss}")
 


model_with_dropout.eval()
predictions = model_with_dropout(X_test_torch).detach().numpy()[:, 0]

print(mean_squared_error(y_test, predictions))


# ## Early stopping 

torch.manual_seed(42)
model = nn.Sequential(nn.Linear(X_train.shape[1], 16),
                      nn.ReLU(),
                      nn.Linear(16, 8),
                      nn.ReLU(),
                      nn.Linear(8, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


patience = 100
epochs_no_improve = 0
best_test_loss = float('inf')


import copy

best_model = model

for epoch in range(500):
    loss = train_step(model, X_train_torch, y_train_torch, loss_function, optimizer)
        
    predictions = model(X_test_torch).detach().numpy()[:, 0]
    test_loss = mean_squared_error(y_test, predictions)
    if test_loss > best_test_loss:
        epochs_no_improve += 1
        if epochs_no_improve > patience:
            print(f"Early stopped at epoch {epoch}")
            break
    else:
        epochs_no_improve = 0
        best_test_loss = test_loss
        best_model = copy.deepcopy(model)
            


predictions = best_model(X_test_torch).detach().numpy()[:, 0] 

print(mean_squared_error(y_test, predictions))


# ---

# Readers may ignore the next cell.

get_ipython().system('jupyter nbconvert --to python ch6_part1.ipynb --TemplateExporter.exclude_input_prompt=True')

