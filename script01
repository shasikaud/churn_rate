import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)
X = X[:, 1:] #remove first column to avoid dummy trap ?
#if there are m number of categories, use m-1 in the model, the value left out can be thought of as the reference value and the fit values of the remaining categories represent the change from this reference

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#############################################################################
#creating ANN

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(output_dim = 6, init = 'uniform',activation = 'relu', input_dim = 11))
classifier.add(Dense(output_dim = 6, init = 'uniform',activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform',activation = 'sigmoid')) #softmax if output more than 1 prob

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred)
