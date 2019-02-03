#/usr/bin/env python 3.6

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style
import json

with open('assess2_data.json') as data_file:
    data = json.load(data_file)
dataframe = pd.DataFrame(data)

# make sepatate list for respoonder and non-responder

res = list(dataframe[dataframe['TARGET_B'] == 1]['WEALTH_INDEX'])
nonres = list(dataframe[dataframe['TARGET_B'] == 0]['WEALTH_INDEX'])

# Assign colors for responder and nonresponders

colors = ['#E69F00','#56B4E9']
kinds = [1,0]

# Make histogram using a list of lists
plt.hist([res, nonres], bins = 20, normed = True, color = colors, label = kinds)
plt.legend()
plt.xlabel('Frequency Distribution')
plt.ylabel('Normalized the counts')
plt.show()
