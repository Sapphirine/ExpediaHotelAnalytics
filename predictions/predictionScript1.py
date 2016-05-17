# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import datetime

train1 = pd.read_csv("../input/train.csv", parse_dates=['date_time'], nrows=1000)
train1test = pd.read_csv("../input/train.csv", parse_dates=['date_time'], nrows=2000)
train1test1 = train1test.ix[1000:,:]
train1test1.info()
train1['date_time'] = pd.to_datetime(train1["date_time"])
train1['year'] = train1['date_time'].dt.year
train1['month'] = train1['date_time'].dt.month
train1['day_of_week'] = train1['date_time'].dt.month
train1['day'] = train1['date_time'].dt.day
train1['hour'] = train1['date_time'].dt.hour
train1test1['date_time'] = pd.to_datetime(train1test1["date_time"])
train1test1['year'] = train1test1['date_time'].dt.year
train1test1['month'] = train1test1['date_time'].dt.month
train1test1['day_of_week'] = train1test1['date_time'].dt.month
train1test1['day'] = train1test1['date_time'].dt.day
train1test1['hour'] = train1test1['date_time'].dt.hour
train1.ix[(train1['hour'] >= 10) & (train1['hour'] < 18), 'hour'] = 1
train1.ix[(train1['hour'] >= 18) & (train1['hour'] < 22), 'hour'] = 2
train1.ix[(train1['hour'] >= 22) & (train1['hour'] == 24), 'hour'] = 3
train1.ix[(train1['hour'] >= 1) & (train1['hour'] < 10), 'hour'] = 3
train1test1.ix[(train1test1['hour'] >= 10) & (train1test1['hour'] < 18), 'hour'] = 1
train1test1.ix[(train1test1['hour'] >= 18) & (train1test1['hour'] < 22), 'hour'] = 2
train1test1.ix[(train1test1['hour'] >= 22) & (train1test1['hour'] == 24), 'hour'] = 3
train1test1.ix[(train1test1['hour'] >= 1) & (train1test1['hour'] < 10), 'hour'] = 3
train1 = train1.fillna(-1)
train1test1 = train1test1.fillna(-1)
train3 = train1
train1['is_booking'] = -1
train1test1['is_booking'] = -1
hotelCluster = train1.ix[:,'hotel_cluster']
hotelClustertest = train1test1.ix[:,'hotel_cluster']
train3 = train3.drop('hotel_cluster', axis=1) #df.drop('reports', axis=1)
train1test1 = train1test1.drop('hotel_cluster',axis=1)
train3 = train3.drop('srch_ci', axis=1)
train1test1 = train1test1.drop('srch_ci', axis=1)
train3 = train3.drop('srch_co', axis=1)
train1test1 = train1test1.drop('srch_co', axis=1)
train3 = train3.drop('date_time', axis=1)
train1test1 = train1test1.drop('date_time', axis=1)
train3.info()
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(40, 100), random_state=1)
clf.fit(train3, hotelCluster) 
clf.predict(train1test1)