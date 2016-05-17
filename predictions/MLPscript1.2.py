import numpy as np
import pandas as pd 
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import datetime
import time
from sklearn import svm
from sklearn.neural_network import MLPClassifier
train1 = pd.read_csv("../input/train.csv", nrows=1000000)
test = pd.read_csv('../input/test.csv', dtype={'srch_destination_id':np.int32}, usecols=['srch_destination_id'])
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
hotelCluster = train1.ix[:,'hotel_cluster']
hotelCluster = pd.DataFrame(hotelCluster)
hotelCluster.info()
train1 = train1.drop('hotel_cluster', axis=1)
train1 = train1.drop('site_name', axis=1)
train1 = train1.drop('posa_continent', axis=1)
train1['Individuals'] = train1['srch_adults_cnt']+train1['srch_children_cnt']
usecols=['srch_destination_id','is_booking','srch_destination_type_id','hotel_cluster','user_location_city','day_of_week','hour','orig_destination_distance','srch_rm_cnt', 'hotel_country','Individuals','month']
train2 = train1.ix[:,usecols]
train1 = pd.concat([train1, hotelCluster], axis=1)
train2 = train2.drop('srch_rm_cnt', axis=1)
train2.info()
train3 = train2.groupby(['srch_destination_id', 'hotel_cluster','srch_destination_type_id','day_of_week','hour','orig_destination_distance','hotel_country','Individuals','month'])['is_booking'].agg(['sum','count'])
train3.reset_index(inplace=True)
train3.head()
CLICK_WEIGHT = 0.05
agg = train3.groupby(['srch_destination_id', 'hotel_cluster','srch_destination_type_id','day_of_week','hour','orig_destination_distance','hotel_country','Individuals','month']).sum().reset_index()
agg['count'] -= agg['sum']
agg = agg.rename(columns={'sum':'bookings','count':'clicks'})
agg['relevance'] = agg['bookings'] + CLICK_WEIGHT * agg['clicks']
agg.head()
def most_popular(group, n_max=5):
    relevance = group['relevance'].values
    hotel_cluster = group['hotel_cluster'].values
    most_popular = hotel_cluster[np.argsort(relevance)[::-1]][:n_max]
    return np.array_str(most_popular)[1:-1]
most_pop = agg.groupby(['srch_destination_id']).apply(most_popular)
most_pop.size
most_pop = pd.DataFrame(most_pop).rename(columns={0:'hotel_cluster'})
most_pop.head()
most_pop.info()
test = test.merge(most_pop, how='left',left_on='srch_destination_id',right_index=True)
test.head()
test.hotel_cluster.isnull().sum()
most_pop_all = agg.groupby('hotel_cluster')['relevance'].sum().nlargest(5).index
most_pop_all = np.array_str(most_pop_all)[1:-1]
most_pop_all
test.hotel_cluster.fillna(most_pop_all,inplace=True)
test.hotel_cluster.to_csv('predicted_without_allgroupby.csv',header=True, index_label='id')