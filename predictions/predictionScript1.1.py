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

from collections import defaultdict
from datetime import datetime
usecols=['srch_destination_id','is_booking','hotel_cluster','srch_adults_cnt','srch_children_cnt','srch_rm_cnt','srch_destination_type_id']
li_cols = ['srch_destination_id','srch_adults_cnt','srch_children_cnt','srch_rm_cnt','srch_destination_type_id']
train = pd.read_csv('../input/train.csv', parse_dates=['date_time'], nrows=100000)
test = pd.read_csv('../input/test.csv', parse_dates=['date_time'], nrows=1000000)
hotel_cluster_count = train.groupby(['srch_destination_id','is_booking','hotel_cluster','srch_adults_cnt','srch_children_cnt','srch_rm_cnt','srch_destination_type_id']).is_booking.agg(['sum','count']).reset_index()
hotel_cluster_count_test = test[li_cols]
hotel_cluster_count_test.describe()
hotel_cluster_count['is_booking'] = 0.8456 * hotel_cluster_count['sum'] + (1 - 0.8456) * hotel_cluster_count['count']
hotel_cluster_count.head(30)
def popular_hotels(gp):
    p = gp.values
    # order hotel_clusters by score then reverse it and take the 5 first
    clusters = p[:, 0][p[:,1].argsort()[::-1]][:5].astype(np.int8)
    return np.array_str(clusters)[1:-1]# remove square brackets
dest_top_five = hotel_cluster_count.groupby(['srch_destination_id', 'srch_adults_cnt', 'srch_children_cnt','srch_rm_cnt','srch_destination_type_id'])['hotel_cluster', 'is_booking'].apply(popular_hotels).reset_index()
dest_top_five = pd.DataFrame(dest_top_five).rename(columns={0:'hotel_cluster'})
dest_top_five.head(10)
merge2 = hotel_cluster_count_test.merge(dest_top_five, how = 'left', on = ('srch_destination_id','srch_adults_cnt','srch_children_cnt','srch_rm_cnt','srch_destination_type_id') , suffixes=('_train_top', '_test_merge'))
merge2.head()
most_pop_all = hotel_cluster_count.groupby('hotel_cluster')['is_booking'].sum().nlargest(5).index
most_pop_all = np.array_str(most_pop_all)[1:-1]
merge2['hotel_cluster'].fillna(most_pop_all,inplace=True)
merge2[['srch_destination_id','hotel_cluster']].head()