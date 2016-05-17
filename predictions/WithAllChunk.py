import numpy as np
import pandas as pd 
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import datetime
import time
train1 = pd.read_csv('../input/train.csv', dtype={'is_booking':bool,'srch_destination_id':np.int32, 'hotel_cluster':np.int32, 'srch_children_cnt':np.int32,'srch_adults_cnt':np.int32,'srch_destination_type_id':np.int32, 'hotel_cluster':np.int32,'orig_destination_distance':np.float64},
                    usecols=['date_time','srch_ci','srch_co','srch_destination_id','is_booking','srch_children_cnt','srch_adults_cnt','srch_destination_type_id','hotel_cluster','orig_destination_distance'], chunksize=1000000)
train1
test = pd.read_csv('../input/test.csv', dtype={'srch_destination_id':np.int32}, usecols=['srch_destination_id'])
train = []
for train2 in train1:
    train2["srch_ci"] = pd.to_datetime(train2["srch_ci"], format='%Y-%m-%d', errors="coerce")
    agg = train2.groupby(['srch_destination_id', 'hotel_cluster','srch_ci'])['is_booking'].agg(['sum','count'])
    #agg.reset_index(inplace=True)
    #aggs.append(agg)
print('')
aggs = pd.concat(aggs, axis=0)
aggs.head()
