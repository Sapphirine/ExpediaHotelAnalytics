import numpy as np
import pandas as pd 
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import datetime
import time
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
desti1 = pd.read_csv("../input/destinations.csv")
X = desti1.ix[:,1:150]
y = desti1.ix[:,0:1]
pca = PCA(n_components=5)
X_r = pca.fit(X).transform(X)




X_r1 = pd.DataFrame(X_r)
X_r1["srch_destination_id"] = desti1["srch_destination_id"]
print('explained variance ratio (first 5 components): %s' % str(pca.explained_variance_ratio_))
# We compresses the 149 columns in destinations down to 5 columns, and creates a new DataFrame called X_r1, preserve most of the variance in destinations, to save a lot of runtime for a machine learning algorithm.
train1 = pd.read_csv("../input/train.csv", nrows=1000000)
train1["srch_ci"] = pd.to_datetime(train1["srch_ci"], format='%Y-%m-%d', errors="coerce")
train1["srch_co"] = pd.to_datetime(train1["srch_co"], format='%Y-%m-%d', errors="coerce")
train1["stay_span"] = (train1["srch_co"] - train1["srch_ci"]).astype('timedelta64[D]')
train2 = pd.to_datetime(train1["date_time"])
train2 = pd.DataFrame(train2)
train1['year'] = train2['date_time'].dt.year
train1['month'] = train2['date_time'].dt.month
train1['day_of_week'] = train2['date_time'].dt.dayofweek
train1['day'] = train2['date_time'].dt.day
train1['hour'] = train2['date_time'].dt.hour
train1["date_time"] = pd.to_datetime(train1["date_time"], format='%Y-%m-%d', errors="coerce")
train1["srch_ci"] = pd.to_datetime(train1["srch_ci"], format='%Y-%m-%d', errors="coerce")
train1["search_span"] = (train1["srch_ci"] - train1["date_time"]).astype('timedelta64[D]')
train1.ix[(train1['hour'] >= 10) & (train1['hour'] < 18), 'hour'] = 1
train1.ix[(train1['hour'] >= 18) & (train1['hour'] < 22), 'hour'] = 2
train1.ix[(train1['hour'] >= 22) & (train1['hour'] == 24), 'hour'] = 3
train1.ix[(train1['hour'] >= 1) & (train1['hour'] < 10), 'hour'] = 3
train1 = train1.drop('srch_ci', axis=1)
train1 = train1.drop('srch_co', axis=1)
train1 = train1.drop('date_time', axis=1)
train1 = train1.join(X_r1, on = 'srch_destination_id', how = 'left', rsuffix='dest')
train1 = train1.drop("srch_destination_iddest", axis=1)
train1.fillna(-1, inplace=True)
hotelCluster = train1.ix[:,'hotel_cluster']
hotelCluster = pd.DataFrame(hotelCluster)
train1 = train1.drop('hotel_cluster', axis=1)
#from sklearn import cross_validation
#from sklearn.ensemble import RandomForestClassifier

clf = MLPClassifier(algorithm='adam', alpha=1e-5, hidden_layer_sizes=(40, 100), learning_rate='adaptive', random_state=1)
hotelCluster.info()
hotelCluster = hotelCluster.values.ravel()
clf.fit(train1, hotelCluster)
np.size(hotelCluster)
clf.fit(train1, hotelCluster)
