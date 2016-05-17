import numpy as np
import pandas as pd 
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import datetime
import time
train1 = pd.read_csv("../input/train.csv", nrows=10000)
from sklearn import svm
from sklearn.decomposition import PCA
test1 = pd.read_csv("../input/test.csv", nrows=100)
train1["srch_ci"] = pd.to_datetime(train1["srch_ci"], format='%Y-%m-%d', errors="coerce")
test1["srch_ci"] = pd.to_datetime(test1["srch_ci"], format='%Y-%m-%d', errors="coerce")
train1["srch_co"] = pd.to_datetime(train1["srch_co"], format='%Y-%m-%d', errors="coerce")
test1["srch_co"] = pd.to_datetime(test1["srch_co"], format='%Y-%m-%d', errors="coerce")
train1["stay_span"] = (train1["srch_co"] - train1["srch_ci"]).astype('timedelta64[D]')
test1["stay_span"] = (test1["srch_co"] - test1["srch_ci"]).astype('timedelta64[D]')
train2 = pd.to_datetime(train1["date_time"])
test2 = pd.to_datetime(test1["date_time"])
train2 = pd.DataFrame(train2)
test2 = pd.DataFrame(test2)
train1['year'] = train2['date_time'].dt.year
train1['month'] = train2['date_time'].dt.month
train1['day_of_week'] = train2['date_time'].dt.dayofweek
train1['day'] = train2['date_time'].dt.day
train1['hour'] = train2['date_time'].dt.hour
test1['year'] = test2['date_time'].dt.year
test1['month'] = test2['date_time'].dt.month
test1['day_of_week'] = test2['date_time'].dt.dayofweek
test1['day'] = test2['date_time'].dt.day
test1['hour'] = test2['date_time'].dt.hour
train1["date_time"] = pd.to_datetime(train1["date_time"], format='%Y-%m-%d', errors="coerce")
train1["srch_ci"] = pd.to_datetime(train1["srch_ci"], format='%Y-%m-%d', errors="coerce")
train1["search_span"] = (train1["srch_ci"] - train1["date_time"]).astype('timedelta64[D]')
test1["date_time"] = pd.to_datetime(test1["date_time"], format='%Y-%m-%d', errors="coerce")
test1["srch_ci"] = pd.to_datetime(test1["srch_ci"], format='%Y-%m-%d', errors="coerce")
test1["search_span"] = (test1["srch_ci"] - test1["date_time"]).astype('timedelta64[D]')
train1.ix[(train1['hour'] >= 10) & (train1['hour'] < 18), 'hour'] = 1
train1.ix[(train1['hour'] >= 18) & (train1['hour'] < 22), 'hour'] = 2
train1.ix[(train1['hour'] >= 22) & (train1['hour'] == 24), 'hour'] = 3
train1.ix[(train1['hour'] >= 1) & (train1['hour'] < 10), 'hour'] = 3
test1.ix[(test1['hour'] >= 10) & (test1['hour'] < 18), 'hour'] = 1
test1.ix[(test1['hour'] >= 18) & (test1['hour'] < 22), 'hour'] = 2
test1.ix[(test1['hour'] >= 22) & (test1['hour'] == 24), 'hour'] = 3
test1.ix[(test1['hour'] >= 1) & (test1['hour'] < 10), 'hour'] = 3
train1 = train1.drop('srch_ci', axis=1)
test1 = test1.drop('srch_ci', axis=1)
train1 = train1.drop('srch_co', axis=1)
test1 = test1.drop('srch_co', axis=1)
train1 = train1.drop('date_time', axis=1)
test1 = test1.drop('date_time', axis=1)
desti1 = pd.read_csv("../input/destinations.csv")
Xt = desti1.ix[:,1:150]
yt = desti1.ix[:,0:1]
pca1 = PCA(n_components=3)
X_train = pca1.fit(Xt).transform(Xt)
X_train1 = pd.DataFrame(X_train)
X_train1["srch_destination_id"] = desti1["srch_destination_id"]
X_train1.info()
pca2 = PCA(n_components=4)
X_test = pca2.fit(Xt).transform(Xt)
X_test1 = pd.DataFrame(X_test)
X_test1["srch_destination_id"] = desti1["srch_destination_id"]
X_train1.columns = ['d0','d1','d2','srch_destination_id']

X_test1.columns = ['d0','d1','d3','d4','srch_destination_id']
train1 = train1.join(X_train1, on = 'srch_destination_id', how = 'left', rsuffix='dest')
test1 = test1.join(X_test1, on = 'srch_destination_id', how = 'left', rsuffix='dest')
train1 = train1.drop("srch_destination_iddest", axis=1)
test1 = test1.drop("srch_destination_iddest", axis=1)
train1.fillna(-1, inplace=True)
test1.fillna(-1, inplace=True)
hotelCluster = train1.ix[:,'hotel_cluster']
hotelCluster1 = pd.DataFrame(hotelCluster)

hotelCluster.shape
train1 = train1.drop('hotel_cluster', axis=1)
hc = np.array(hotelCluster1)
hc.shape
print(hotelCluster.shape)
clf = svm.SVC()
clf.fit(train1, hc)
