#!/usr/bin/env python
# coding: utf-8

# Source codes for Python Machine Learning By Example 4th Edition (Packt Publishing)
# 
# Chapter 9 Recognizing Faces with Support Vector Machine
# 
# Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)

# # Classifying face images with SVM

# ## Exploring the face image dataset 

from sklearn.datasets import fetch_lfw_people

# face_data = fetch_lfw_people(min_faces_per_person=80)
face_data = fetch_lfw_people(data_home='./', min_faces_per_person=80, download_if_missing=False)


X = face_data.data
Y = face_data.target

print('Input data size :', X.shape)
print('Output data size :', Y.shape)
print('Label names:', face_data.target_names)


for i in range(5):
    print(f'Class {i} has {(Y == i).sum()} samples.')


import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 4)
for i, axi in enumerate(ax.flat):
    axi.imshow(face_data.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[],
            xlabel=face_data.target_names[face_data.target[i]])

plt.show()


# ## Building an SVM-based image classifier

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)


from sklearn.svm import SVC
clf = SVC(class_weight='balanced', random_state=42)


from sklearn.model_selection import GridSearchCV
parameters = {'C': [10, 100, 300],
              'gamma': [0.0001,  0.0003, 0.001],
              'kernel' : ['rbf', 'linear'] }

grid_search = GridSearchCV(clf, parameters, n_jobs=-1, cv=5)


grid_search.fit(X_train, Y_train)


print('The best model:\n', grid_search.best_params_)


print('The best averaged performance:', grid_search.best_score_)


clf_best = grid_search.best_estimator_

print(f'The accuracy is: {clf_best.score(X_test, Y_test)*100:.1f}%')


pred = clf_best.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(Y_test, pred, target_names=face_data.target_names))


# ## Boosting image classification performance with PCA 

from sklearn.decomposition import PCA
pca = PCA(n_components=100, whiten=True, random_state=42)
svc = SVC(class_weight='balanced', kernel='rbf', random_state=42)

from sklearn.pipeline import Pipeline
model = Pipeline([('pca', pca),
                  ('svc', svc)])


parameters_pipeline = {'svc__C': [1, 3, 10],
                       'svc__gamma': [0.01,  0.03, 0.003]}
grid_search = GridSearchCV(model, parameters_pipeline, n_jobs=-1, cv=5)

grid_search.fit(X_train, Y_train)


print('The best model:\n', grid_search.best_params_)
print('The best averaged performance:', grid_search.best_score_)

model_best = grid_search.best_estimator_
print(f'The accuracy is: {model_best.score(X_test, Y_test)*100:.1f}%')
pred = model_best.predict(X_test)
print(classification_report(Y_test, pred, target_names=face_data.target_names))


# # Estimating with support vector regression 

# ## Implementing SVR 

from sklearn import datasets
diabetes = datasets.load_diabetes()

X = diabetes.data
Y = diabetes.target

print('Input data size :', X.shape)
print('Output data size :', Y.shape)
 


num_test = 30    # the last 30 samples as testing set
X_train = diabetes.data[:-num_test, :]
y_train = diabetes.target[:-num_test]
X_test = diabetes.data[-num_test:, :]
y_test = diabetes.target[-num_test:]


from sklearn.svm import SVR
regressor = SVR(C=100, kernel='linear')
regressor.fit(X_train, y_train)


from sklearn.metrics import r2_score
predictions = regressor.predict(X_test)
print(r2_score(y_test, predictions))


parameters = {'C': [300, 500, 700],
              'gamma': [0.3, 0.6, 1],
              'kernel' : ['rbf', 'linear']}

regressor = SVR()
grid_search = GridSearchCV(regressor, parameters, n_jobs=-1, cv=5)


grid_search.fit(X_train, y_train)


print('The best model:\n', grid_search.best_params_)


model_best = grid_search.best_estimator_
predictions = model_best.predict(X_test)

print(r2_score(y_test, predictions))


# ---

# Readers may ignore the next cell.

get_ipython().system('jupyter nbconvert --to python ch9_part2.ipynb --TemplateExporter.exclude_input_prompt=True')

