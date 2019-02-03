#/usr/bin/env python 3.6

# Use Elbow analysis to find out the optimal number of clusters

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import json
import pylab as pl
from sklearn.preprocessing import MinMaxScaler

with open('assess2_data.json') as data_file:
    data = json.load(data_file)
dataframe = pd.DataFrame(data)

# drop some unuseful features
dataframe = dataframe.drop(['NAME','PEPSTRFL'], axis = 1)

categorical_features = ['INCOME','RFA_2A','RFA_2F', 'TARGET_B']
continuous_features = ['AVGGIFT','LASTGIFT','WEALTH_INDEX']

# Transform the categorical variables into dummy variables

for col in categorical_features:
    dummies = pd.get_dummies(dataframe[col], prefix = col)
    dataframe = pd.concat([dataframe, dummies], axis = 1)
    dataframe.drop(col, axis = 1, inplace = True)

print(dataframe.head())

#scale the continuous features
mms = MinMaxScaler()
mms.fit(dataframe)
data_transformed = mms.transform(dataframe)

print("The dataframe has become a numpy array with size of ")
print(data_transformed.shape)

dummy =input('Press Enter')

Sum_of_squared_distances = []

K = range(1,30)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method for Optimal K')
plt.show()



