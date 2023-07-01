#!/usr/bin/env python
# coding: utf-8

# 
# Source codes for Python Machine Learning By Example 4th Edition (Packt Publishing)
# 
# Chapter 4 Predicting Online Ad Click-Through with Tree-Based Algorithms 
# 
# Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
# 

# # Training on large datasets with online learning 

import numpy as np
import pandas as pd
import timeit
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder


n_rows = 100000 * 11
df = pd.read_csv("train.csv", nrows=n_rows)

X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values

n_train = 100000 * 10
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]


enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X_train)


# The number of iterations is set to 1 if using partial_fit.
sgd_lr_online = SGDClassifier(loss='log_loss', 
                              penalty=None, 
                              fit_intercept=True, 
                              max_iter=1, 
                              learning_rate='constant',
                              eta0=0.01, 
                              random_state=42)


start_time = timeit.default_timer()

# Use the first 1,000,000 samples for training, and the next 100,000 for testing
for i in range(10):
    x_train = X_train[i*100000:(i+1)*100000]
    y_train = Y_train[i*100000:(i+1)*100000]
    x_train_enc = enc.transform(x_train)
    sgd_lr_online.partial_fit(x_train_enc.toarray(), y_train, classes=[0, 1])

print(f"--- {(timeit.default_timer() - start_time):.3f} seconds ---")


x_test_enc = enc.transform(X_test)

pred = sgd_lr_online.predict_proba(x_test_enc.toarray())[:, 1]
print(f'Training samples: {n_train * 10}, AUC on testing set: {roc_auc_score(Y_test, pred):.3f}')


# # Handling multiclass classification 

from sklearn import datasets
digits = datasets.load_digits()
n_samples = len(digits.images)


X = digits.images.reshape((n_samples, -1))
Y = digits.target


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


from sklearn.model_selection import GridSearchCV
parameters = {'penalty': ['l2', None],
              'alpha': [1e-07, 1e-06, 1e-05, 1e-04],
              'eta0': [0.01, 0.1, 1, 10]}

sgd_lr = SGDClassifier(loss='log_loss', 
                       learning_rate='constant', 
                       fit_intercept=True, 
                       max_iter=50,
                       random_state=42)

grid_search = GridSearchCV(sgd_lr, parameters, n_jobs=-1, cv=5)

grid_search.fit(X_train, Y_train)
print(grid_search.best_params_)


sgd_lr_best = grid_search.best_estimator_
accuracy = sgd_lr_best.score(X_test, Y_test)
print(f'The accuracy on testing set is: {accuracy*100:.1f}%')


# # Implementing logistic regression using TensorFlow 

import tensorflow as tf


n_rows = 100000
df = pd.read_csv("train.csv", nrows=n_rows)

X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values

n_train = int(n_rows * 0.9)
X_train = X[:n_train]
Y_train = Y[:n_train] 
X_test = X[n_train:]
Y_test = Y[n_train:] 


enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train).toarray().astype('float32')
X_test_enc = enc.transform(X_test).toarray().astype('float32')
Y_train = Y_train.astype('float32')
Y_test = Y_test.astype('float32')


batch_size = 1000
train_data = tf.data.Dataset.from_tensor_slices((X_train_enc, Y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)


n_features = X_train_enc.shape[1]
W = tf.Variable(tf.zeros([n_features, 1]))
b = tf.Variable(tf.zeros([1]))


learning_rate = 0.001
optimizer = tf.optimizers.Adam(learning_rate)


def run_optimization(x, y):
    with tf.GradientTape() as tape:
        logits = tf.add(tf.matmul(x, W), b)[:, 0]
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
    # Update the parameters with respect to the gradient calculations
    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))
    


training_steps = 5000
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    run_optimization(batch_x, batch_y)
    if step % 500 == 0:
        logits = tf.add(tf.matmul(batch_x, W), b)[:, 0]
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_y, logits=logits))
        print("step: %i, loss: %f" % (step, loss))


logits = tf.add(tf.matmul(X_test_enc, W), b)[:, 0]
pred = tf.nn.sigmoid(logits)
auc_metric = tf.keras.metrics.AUC()
auc_metric.update_state(Y_test, pred)

print(f'AUC on testing set: {auc_metric.result().numpy():.3f}')


# # Feature selection using random forest 

X_train = X
Y_train = Y

enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)


# Feature selection with random forest

from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=30, n_jobs=-1, random_state=42)
random_forest.fit(X_train_enc.toarray(), Y_train)


feature_imp = random_forest.feature_importances_
print(feature_imp)


# bottom 10 weights and the corresponding 10 least important features
feature_names = enc.get_feature_names_out()
print(np.sort(feature_imp)[:10])
bottom_10 = np.argsort(feature_imp)[:10]
print('10 least important features are:\n', feature_names[bottom_10])


# top 10 weights and the corresponding 10 most important features
print(np.sort(feature_imp)[-10:])
top_10 = np.argsort(feature_imp)[-10:]
print('10 most important features are:\n', feature_names[top_10])


# ---

# Readers may ignore the next cell.

get_ipython().system('jupyter nbconvert --to python ch4_part2.ipynb --TemplateExporter.exclude_input_prompt=True')

