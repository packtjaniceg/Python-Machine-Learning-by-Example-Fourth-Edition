#!/usr/bin/env python
# coding: utf-8

# 
# Source codes for Python Machine Learning By Example 4th Edition (Packt Publishing)
# 
# Chapter 3 Predicting Online Ad Click-Through with Tree-Based Algorithms 
# 
# Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)

# # Predicting ad click-through with a decision tree

import pandas as pd
n_rows = 300000
df = pd.read_csv("train.csv", nrows=n_rows)
print(df.head(5))


X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values

print(X.shape)


n_train = int(n_rows * 0.9)
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)

X_train_enc[0]
print(X_train_enc[0])


X_test_enc = enc.transform(X_test)


from sklearn.tree import DecisionTreeClassifier
parameters = {'max_depth': [3, 10, None]}
decision_tree = DecisionTreeClassifier(criterion='gini', min_samples_split=30)

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(decision_tree, parameters, n_jobs=-1, cv=3, scoring='roc_auc')

grid_search.fit(X_train_enc, Y_train)
print(grid_search.best_params_)


decision_tree_best = grid_search.best_estimator_
pos_prob = decision_tree_best.predict_proba(X_test_enc)[:, 1]

from sklearn.metrics import roc_auc_score
print(f'The ROC AUC on testing set is: {roc_auc_score(Y_test, pos_prob):.3f}')


import numpy as np
pos_prob = np.zeros(len(Y_test))
click_index = np.random.choice(len(Y_test), int(len(Y_test) *  51211.0/300000), replace=False)
pos_prob[click_index] = 1

print(f'The ROC AUC on testing set using random selection is: {roc_auc_score(Y_test, pos_prob):.3f}')


# # Ensembling decision trees – random forest  

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=30, n_jobs=-1)
grid_search = GridSearchCV(random_forest, parameters, n_jobs=-1, cv=3, scoring='roc_auc')
grid_search.fit(X_train_enc, Y_train)
print(grid_search.best_params_)


random_forest_best = grid_search.best_estimator_
pos_prob = random_forest_best.predict_proba(X_test_enc)[:, 1]
print(f'The ROC AUC on testing set using random forest is: {roc_auc_score(Y_test, pos_prob):.3f}')


# # Ensembling decision trees – gradient boosted trees

import xgboost as xgb
model = xgb.XGBClassifier(learning_rate=0.1, max_depth=10, n_estimators=1000)

model.fit(X_train_enc, Y_train)
pos_prob = model.predict_proba(X_test_enc)[:, 1]

print(f'The ROC AUC on testing set using GBT is: {roc_auc_score(Y_test, pos_prob):.3f}')


# ---

# Readers may ignore the next cell.

get_ipython().system('jupyter nbconvert --to python ch3_part2.ipynb --TemplateExporter.exclude_input_prompt=True')

