#!/usr/bin/env python
# coding: utf-8

# Source codes for Python Machine Learning By Example 4th Edition (Packt Publishing)
# 
# Chapter 8 Discovering Underlying Topics in the Newsgroups Dataset with Clustering and Topic Modeling
# 
# Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)

# # Getting started with k-means clustering

# ## Implementing k-means from scratch

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, 2:4]
y = iris.target


import numpy as np
from matplotlib import pyplot as plt
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()


k = 3
np.random.seed(0)
random_index = np.random.choice(range(len(X)), k)
centroids = X[random_index]


def visualize_centroids(X, centroids):
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
    plt.show()

visualize_centroids(X, centroids)


def dist(a, b):
    return np.linalg.norm(a - b, axis=1)


def assign_cluster(x, centroids):
    distances = dist(x, centroids)
    cluster = np.argmin(distances)
    return cluster


def update_centroids(X, centroids, clusters):
    for i in range(k):
        cluster_i = np.where(clusters == i)
        centroids[i] = np.mean(X[cluster_i], axis=0)


tol = 0.0001
max_iter = 100

iter = 0
centroids_diff = 100000
clusters = np.zeros(len(X))


from copy import deepcopy
while iter < max_iter and centroids_diff > tol:
    for i in range(len(X)):
        clusters[i] = assign_cluster(X[i], centroids)
    centroids_prev = deepcopy(centroids)
    update_centroids(X, centroids, clusters)
    iter += 1
    centroids_diff = np.linalg.norm(centroids - centroids_prev)
    print('Iteration:', str(iter))
    print('Centroids:\n', centroids)
    print(f'Centroids move: {centroids_diff:5.4f}')
    visualize_centroids(X, centroids)


plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='r')
plt.show()


# ## Implementing k-means with scikit-learn

from sklearn.cluster import KMeans
kmeans_sk = KMeans(n_clusters=3, n_init='auto', random_state=42)


kmeans_sk.fit(X)


clusters_sk = kmeans_sk.labels_
centroids_sk = kmeans_sk.cluster_centers_


plt.scatter(X[:, 0], X[:, 1], c=clusters_sk)
plt.scatter(centroids_sk[:, 0], centroids_sk[:, 1], marker='*', s=200, c='r')
plt.show()


# ## Choosing the value of k 

X = iris.data
y = iris.target
k_list = list(range(1, 7))
sse_list = [0] * len(k_list)


for k_ind, k in enumerate(k_list):
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit(X)
    clusters = kmeans.labels_
    centroids = kmeans.cluster_centers_

    sse = 0
    for i in range(k):
        cluster_i = np.where(clusters == i)

        sse += np.linalg.norm(X[cluster_i] - centroids[i])

    print(f'k={k}, SSE={sse}')
    sse_list[k_ind] = sse


plt.plot(k_list, sse_list)
plt.show()


# ---

# Readers may ignore the next cell.

get_ipython().system('jupyter nbconvert --to python ch8_part1.ipynb --TemplateExporter.exclude_input_prompt=True')

