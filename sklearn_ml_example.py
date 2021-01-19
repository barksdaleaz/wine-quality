import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.externals import joblib <- DEPRECATED
import joblib
# Load red wine data.
dataset_url = 'winequality-red.csv'
data = pd.read_csv(dataset_url)
# data = pd.read_csv(dataset_url, sep=';')

# print(data.head())
# print(data.shape)

# Split data into training and test sets
Y = data.quality
X = data.drop('quality', axis = 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123, stratify=Y)

scaler = preprocessing.StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
# print(X_train_scaled.mean(axis=0))
#
# print(X_train_scaled.std(axis=0))

X_test_scaled = scaler.transform(X_test)
# print(X_test_scaled.mean(axis=0))
# print(X_test_scaled.std(axis=0))

# Declare data preprocessing steps
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

# Declare hyperparameters to tune
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

# Tune model using cross-validation pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

# Fit and tune model
clf.fit(X_train, Y_train)

# print(clf.best_params_)

# Refit on the entire training set. No additional code needed if clf.refit == true
# print(clf.refit)
#True^

# Evaluate model pipeline on test data
Y_pred = clf.predict(X_test)
print(r2_score(Y_test, Y_pred))

print(mean_squared_error(Y_test, Y_pred))

# Save model for future use
joblib.dump(clf, 'rf_regressor.pkl')

#load model from .pkl file
#clf2 = joblib.load('rf_regressor.pkl')
#Predict data set using loaded model
#clf2.predict(X_test)

# Resources:
# https://elitedatascience.com/python-machine-learning-tutorial-scikit-learn
# https://www.kaggle.com/madhurisivalenka/basic-machine-learning-with-red-wine-quality-data
# https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009
