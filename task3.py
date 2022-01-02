# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 14:49:27 2022

@author: Gada
"""

#Importing the libraries
import pandas as pd
import numpy as np

#Importing the dataset
dataset=pd.read_csv("diabetes.csv")
x=dataset.iloc[:,0:8].values
y=dataset.iloc[:, 8].values

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_predD= classifier.predict(x_test)
#accuracy
from sklearn.metrics import accuracy_score
print('Accuracy: ',accuracy_score(y_test,y_predD))
#Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
clf.fit(x_train, y_train)

#Predicting the Test set results
y_predR= clf.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predR)
#accuracy
print('Accuracy: ',accuracy_score(y_test,y_predR))

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors= 27, p=2, metric='euclidean')
knn.fit(x_train,y_train)
#Predicting the Test set results
y_predK = knn.predict(x_test)
#accuracy
print('Accuracy: ',accuracy_score(y_test,y_predK))
#making confsion matrix

km = confusion_matrix(y_test, y_predK)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
mlr=LogisticRegression(max_iter= 180)
mlr.fit(x_train, y_train)
y_predLR=mlr.predict(x_test)
#accuracy
print('Accuracy: ',accuracy_score(y_test,y_predLR))







