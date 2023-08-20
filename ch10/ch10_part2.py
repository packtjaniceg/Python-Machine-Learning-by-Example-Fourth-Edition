#!/usr/bin/env python
# coding: utf-8

# Source codes for Python Machine Learning By Example 4th Edition (Packt Publishing)
# 
# Chapter 10 Machine Learning Best Practices
# 
# Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)

# ## Best practice 14 – Extracting features from text data 

# ### Word embedding

from gensim.models import Word2Vec


# Sample sentences for training
sentences = [
    ["i", "love", "machine", "learning", "by", "example"],
    ["machine", "learning", "and", "deep", "learning", "are", "fascinating"],
    ["word", "embedding", "is", "essential", "for", "many", "nlp", "tasks"],
    ["word2vec", "produces", "word", "embeddings"]
]

# Create and train Word2Vec model
model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, sg=0)

# Access word vectors
vector = model.wv["machine"]
print("Vector for 'machine':", vector)


import torch
import torch.nn as nn

# Sample data
input_data = torch.LongTensor([[1, 2, 3, 4], [5, 1, 6, 3]])

# Define the embedding layer
vocab_size = 10  # Total number of unique words
embedding_dim = 3  # Dimensionality of the embeddings
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# Pass input data through the embedding layer
embedded_data = embedding_layer(input_data)

# Print the embedded data
print("Embedded Data:\n", embedded_data)


# # Best practices in the deployment and monitoring stage 

# # Best practice 19 – Saving, loading, and reusing models 

# ### Saving and restoring models using pickle 

from sklearn import datasets
dataset = datasets.load_diabetes()
X, y = dataset.data, dataset.target

num_new = 30    # the last 30 samples as new data set
X_train = X[:-num_new, :]
y_train = y[:-num_new]
X_new = X[-num_new:, :]
y_new = y[-num_new:]


# Data pre-processing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)


import pickle
# Save the scaler
pickle.dump(scaler, open("scaler.p", "wb" ))


X_scaled_train = scaler.transform(X_train)


# Regression model training
from sklearn.svm import SVR
regressor = SVR(C=20)
regressor.fit(X_scaled_train, y_train)


# Save the regressor
pickle.dump(regressor, open("regressor.p", "wb"))


# Deployment
my_scaler = pickle.load(open("scaler.p", "rb" ))
my_regressor = pickle.load(open("regressor.p", "rb"))


X_scaled_new = my_scaler.transform(X_new)
predictions = my_regressor.predict(X_scaled_new)


# Monitor
from sklearn.metrics import r2_score
print(f'Health check on the model, R^2: {r2_score(y_new, predictions):.3f}')


# ### Saving and restoring models in TensorFlow 

import tensorflow as tf
from tensorflow import keras

cancer_data = datasets.load_breast_cancer()
X = cancer_data.data
X = scaler.fit_transform(X)
y = cancer_data.target


learning_rate = 0.005
n_iter = 10

tf.random.set_seed(42)

model = keras.Sequential([
    keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate))


model.fit(X, y, epochs=n_iter)


model.summary()


path = './model_tf'
model.save(path)


new_model = tf.keras.models.load_model(path)

new_model.summary()


# ### Saving and restoring models in PyTorch

X_torch = torch.FloatTensor(X)
y_torch = torch.FloatTensor(y.reshape(y.shape[0], 1))


torch.manual_seed(42)
 
model = nn.Sequential(nn.Linear(X.shape[1], 1),
                      nn.Sigmoid())
 
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_step(model, X_train, y_train, loss_function, optimizer):
    pred_train = model(X_train)
    loss = loss_function(pred_train, y_train)
    model.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


for epoch in range(n_iter):
    loss = train_step(model, X_torch, y_torch, loss_function, optimizer)
    print(f"Epoch {epoch} - loss: {loss}")


print(model)


path = './model.pth'
torch.save(model, path)


new_model = torch.load(path)
print(new_model)


# ---

# Readers may ignore the next cell.

get_ipython().system('jupyter nbconvert --to python ch10_part2.ipynb --TemplateExporter.exclude_input_prompt=True')

