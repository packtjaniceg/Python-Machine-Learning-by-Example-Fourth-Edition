#!/usr/bin/env python
# coding: utf-8

# Source codes for Python Machine Learning By Example 4th Edition (Packt Publishing)
# 
# Chapter 10 Machine Learning Best Practices
# 
# Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)

# # Best practices in the data preparation stage

# ## Best practice 4 – Dealing with missing data

import numpy as np
from sklearn.impute import SimpleImputer


data_origin = [[30, 100],
               [20, 50],
               [35, np.nan],
               [25, 80],
               [30, 70],
               [40, 60]]


imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(data_origin)


data_mean_imp = imp_mean.transform(data_origin)
print(data_mean_imp)


imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
imp_median.fit(data_origin)
data_median_imp = imp_median.transform(data_origin)
print(data_median_imp)


# New samples
new = [[20, np.nan],
       [30, np.nan],
       [np.nan, 70],
       [np.nan, np.nan]]
new_mean_imp = imp_mean.transform(new)
print(new_mean_imp)


# Effects of discarding missing values and imputation
from sklearn import datasets
dataset = datasets.load_diabetes()
X_full, y = dataset.data, dataset.target


m, n = X_full.shape
m_missing = int(m * 0.25)
print(m, m_missing)


np.random.seed(42)
missing_samples = np.array([True] * m_missing + [False] * (m - m_missing))
np.random.shuffle(missing_samples)


missing_features = np.random.randint(low=0, high=n, size=m_missing)


X_missing = X_full.copy()
X_missing[np.where(missing_samples)[0], missing_features] = np.nan


# Discard samples containing missing values
X_rm_missing = X_missing[~missing_samples, :]
y_rm_missing = y[~missing_samples]


# Estimate R^2 on the data set with missing samples removed
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
regressor = RandomForestRegressor(random_state=42, max_depth=10, n_estimators=100)
score_rm_missing = cross_val_score(regressor, X_rm_missing, y_rm_missing).mean()
print(f'Score with the data set with missing samples removed: {score_rm_missing:.2f}')


# Imputation with mean value
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X_mean_imp = imp_mean.fit_transform(X_missing)


# Estimate R^2 on the data set with missing samples removed
regressor = RandomForestRegressor(random_state=42, max_depth=10, n_estimators=100)
score_mean_imp = cross_val_score(regressor, X_mean_imp, y).mean()
print(f'Score with the data set with missing values replaced by mean: {score_mean_imp:.2f}')


# Estimate R^2 on the full data set
regressor = RandomForestRegressor(random_state=42, max_depth=10, n_estimators=500)
score_full = cross_val_score(regressor, X_full, y).mean()
print(f'Score with the full data set: {score_full:.2f}')


# # Best practices in the training sets generation stage

# ## Best practice 8 – Deciding whether to select features, and if so, how to do so 

from sklearn.datasets import load_digits
dataset = load_digits()
X, y = dataset.data, dataset.target
print(X.shape)


# Estimate accuracy on the original data set
from sklearn.svm import SVC
classifier = SVC(gamma=0.005, random_state=42)
score = cross_val_score(classifier, X, y).mean()
print(f'Score with the original data set: {score:.2f}')


# Feature selection with random forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, criterion='gini', n_jobs=-1, random_state=42)
random_forest.fit(X, y)

# Sort features based on their importancies
feature_sorted = np.argsort(random_forest.feature_importances_)


# Select different number of top features
K = [10, 15, 25, 35, 45]
for k in K:
    top_K_features = feature_sorted[-k:]
    X_k_selected = X[:, top_K_features]
    # Estimate accuracy on the data set with k selected features
    classifier = SVC(gamma=0.005)
    score_k_features = cross_val_score(classifier, X_k_selected, y).mean()
    print(f'Score with the dataset of top {k} features: {score_k_features:.2f}')


# ## Best practice 9 – Deciding whether to reduce dimensionality, and if so, how to do so! 

from sklearn.decomposition import PCA

# Keep different number of top components
N = [10, 15, 25, 35, 45]
for n in N:
    pca = PCA(n_components=n)
    X_n_kept = pca.fit_transform(X)
    # Estimate accuracy on the data set with top n components
    classifier = SVC(gamma=0.005)
    score_n_components = cross_val_score(classifier, X_n_kept, y).mean()
    print(f'Score with the dataset of top {n} components: {score_n_components:.2f}')


# ## Best practice 12 – Performing feature engineering without domain expertise 

# ### Binarization and discretization 

from sklearn.preprocessing import Binarizer
X = [[4], [1], [3], [0]]
binarizer = Binarizer(threshold=2.9)
X_new = binarizer.fit_transform(X)
print(X_new)


# ### Polynomial transformation

from sklearn.preprocessing import PolynomialFeatures
X = [[2, 4],
     [1, 3],
     [3, 2],
     [0, 3]]
poly = PolynomialFeatures(degree=2)
X_new = poly.fit_transform(X)
print(X_new)


# ---

# Readers may ignore the next cell.

get_ipython().system('jupyter nbconvert --to python ch10_part1.ipynb --TemplateExporter.exclude_input_prompt=True')

