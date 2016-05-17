import numpy as np
import pandas as pd 
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import datetime
import time
# Any results you write to the current directory are saved as output.
train1 = pd.read_csv('../input/train.csv', dtype={'is_booking':bool,'srch_destination_id':np.int32, 'hotel_cluster':np.int32, 'srch_children_cnt':np.int32,'srch_adults_cnt':np.int32,'srch_destination_type_id':np.int32, 'hotel_cluster':np.int32,'orig_destination_distance':np.float64},
                    usecols=['date_time','srch_ci','srch_co','srch_destination_id','is_booking','srch_children_cnt','srch_adults_cnt','srch_destination_type_id','hotel_cluster','orig_destination_distance'], chunksize=1000000)
test = pd.read_csv('../input/test.csv', dtype={'srch_destination_id':np.int32}, usecols=['srch_destination_id'])
train = [ ]
for train2 in train1:
    train2["srch_ci"] = pd.to_datetime(train2["srch_ci"], format='%Y-%m-%d', errors="coerce")
    train2["srch_co"] = pd.to_datetime(train2["srch_co"], format='%Y-%m-%d', errors="coerce")
    train2["stay_span"] = (train2["srch_co"] - train2["srch_ci"]).astype('timedelta64[D]')
    #train2 = train2.drop('srch_co', axis=1)
    train2["date_time"] = pd.to_datetime(train2["date_time"], format='%Y-%m-%d', errors="coerce")
    train2["search_span"] = (train2["srch_ci"] - train2["date_time"]).astype('timedelta64[D]')
    #train2 = train2.drop('srch_ci', axis=1)
    train2['year'] = train2['date_time'].dt.year
    train2['month'] = train2['date_time'].dt.month
    train2['day_of_week'] = train2['date_time'].dt.dayofweek
    train2['hour'] = train2['date_time'].dt.hour
    #train2 = train2.drop('date_time', axis=1)
    train2.ix[(train2['hour'] >= 10) & (train2['hour'] < 18), 'hour'] = 1
    train2.ix[(train2['hour'] >= 18) & (train2['hour'] < 22), 'hour'] = 2
    train2.ix[(train2['hour'] >= 22) & (train2['hour'] == 24), 'hour'] = 3
    train2.ix[(train2['hour'] >= 1) & (train2['hour'] < 10), 'hour'] = 3
    train2['Individuals'] = train2['srch_adults_cnt']+train2['srch_children_cnt']
    #train2 = train2.drop('srch_adults_cnt', axis=1)
    #train2 = train2.drop('srch_children_cnt', axis=1)
    #train2 = train2.drop('search_span', axis=1)
    #train2 = train2.drop('user_location_city', axis=1)
    #train2 = train2.drop('hotel_country', axis=1)
    train2 = train2[['orig_destination_distance','srch_destination_id','srch_destination_type_id','is_booking','hotel_cluster','stay_span','year','month','day_of_week','hour', 'Individuals']]
    agg = train2.groupby(['srch_destination_id','srch_destination_type_id','hotel_cluster','day_of_week','hour'])['is_booking'].agg(['sum','count'])
    agg.reset_index(inplace=True)
    train.append(agg)
train = pd.concat(train, axis=0)
train.head()
CLICK_WEIGHT = 0.15
agg = train.groupby(['srch_destination_id','srch_destination_type_id','hotel_cluster','day_of_week','hour']).sum().reset_index()
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
most_pop = pd.DataFrame(most_pop).rename(columns={0:'hotel_cluster'})
most_pop.head()
test = test.merge(most_pop, how='left',left_on='srch_destination_id',right_index=True)
test.head()
test.hotel_cluster.isnull().sum()
most_pop_all = agg.groupby('hotel_cluster')['relevance'].sum().nlargest(5).index
most_pop_all = np.array_str(most_pop_all)[1:-1]
most_pop_all
test.hotel_cluster.fillna(most_pop_all,inplace=True)
test.hotel_cluster.to_csv('predicted_with_pandas.csv',header=True, index_label='id')