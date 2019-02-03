#/usr/bin/env python 3.6
# Build a logistic regression classifier

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
from matplotlib import style
style.use('ggplot')
import json
from sklearn.decomposition import PCA
import pylab as pl
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.linear_model import LogisticRegression


with open('assess2_data.json') as data_file:
    data = json.load(data_file)
dataframe = pd.DataFrame(data)

# drop rows with outliners

dataframe = dataframe[dataframe['AVGGIFT'] != -9999.00]
dataframe = dataframe[dataframe['FISTDATE'] != -9999]

# drop some unuseful features and data transfomations
y = dataframe['TARGET_B']
dataframe = dataframe.drop(['NAME','PEPSTRFL','FISTDATE','LASTDATE','TARGET_B'], axis = 1)

dataframe['RFA_2A'] = dataframe['RFA_2A'].apply({'E':0, 'F':1, 'D':2, 'G':3, -9999:0}.get)

print(dataframe.head())
print(y.head())

dummy = input('Press Enter')

categorical_features = ['INCOME','RFA_2A','RFA_2F']
continuous_features = ['AVGGIFT','LASTGIFT','WEALTH_INDEX']

#print(dataframe[continuous_features].describe())

for col in categorical_features:
    dummies = pd.get_dummies(dataframe[col], prefix = col, drop_first = True)
    dataframe = pd.concat([dataframe, dummies], axis = 1)
    dataframe.drop(col, axis = 1, inplace = True)

print('The final data before conducting classification model')
print(dataframe.head())
print(dataframe.columns.values)

# after the dummy variables change, 213 rows !!!!

# scale the continuous features
# mms = MinMaxScaler()
# mms.fit(dataframe)
# data_transformed = mms.transform(dataframe)

# print("The dataframe has become a numpy array with size of ")
# print(data_transformed.shape)

X = dataframe

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)


logmodel = LogisticRegression(random_state = 0)
result = logmodel.fit(X_train, y_train)
#print(result.summary())


print('\n The accuracy for logistic regression classifier on trains set is ')
print(logmodel.score(X_train,y_train))
print('The accuracy for logistic regression classifier on test set is ')
print(logmodel.score(X_test,y_test))

# plotting the confusion matrix

from sklearn.metrics import confusion_matrix 

y_pred = logmodel.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
print('\n The confusion_matrix of logistic regression model is ')
print(confusion_matrix)

# There are 6538 + 18 = 6554 correct predictions

# plotting the roc curve

from sklearn.metrics import roc_curve, roc_auc_score
logit_roc_auc = roc_auc_score(y_test, logmodel.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logmodel.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'b--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()






