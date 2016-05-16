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
%matplotlib inline
import numpy as np
import pandas as pd 
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
train1 = pd.read_csv("../input/train.csv", parse_dates=['date_time'], nrows=10000000)
train1["date_time"] = pd.to_datetime(train1["date_time"], format='%Y-%m-%d', errors="coerce")
train1["srch_ci"] = pd.to_datetime(train1["srch_ci"], format='%Y-%m-%d', errors="coerce")
train1["search_span"] = (train1["srch_ci"] - train1["date_time"]).astype('timedelta64[D]')
train_bookings = train1[train1['is_booking'] == 1].drop('is_booking', axis=1)
train_clicks = train1[train1['is_booking'] == 0].drop('is_booking', axis=1)
train_clicks.shape
train1.shape
train_bookings.shape
train_bookingszna = train_bookings.fillna(0)
train_clickszna = train_clicks.fillna(0)
train_bookings_package = train_bookingszna[train_bookingszna['is_package'] == 1]
train_bookingszna.shape
train_bookings_package.shape
train_bookingszna_ss = train_bookings_package['search_span']
train_bookingszna_all = train_bookingszna['search_span']
booked_bookings_all, countall = np.unique(train_bookingszna_all, return_counts=True)
booked_bookings, count = np.unique(train_bookingszna_ss, return_counts=True)
np.mean(count)
np.mean(countall)
np.sort(count)
np.sort(countall)
most_visited2 = booked_bookings[count >= 95]
most_visited3= booked_bookings_all[countall >= 1150]
train_bookings_package1 = train_bookings_package[train_bookings_package['search_span'].isin(most_visited2)]
train_bookings_package1all = train_bookingszna[train_bookingszna['search_span'].isin(most_visited3)]
train_bookings_package1.ix[(train_bookings_package1['search_span'] <= -1), 'search_span'] = 0
train_bookings_package1all.ix[(train_bookings_package1all['search_span'] <= -1), 'search_span'] = 0
f, ax = plt.subplots(figsize=(15, 25))
sns.countplot(y='search_span', data=train_bookings_package1)
sns.plt.title('number of days before check in vs count of instances with packages for 10m bookings')
plt.show()
f, ax = plt.subplots(figsize=(10, 15))
sns.countplot(y='search_span', hue="is_package", data=train_bookings_package1all)
sns.plt.title('with packages or not: number of days before check in vs count of instances for 10m bookings')
plt.show()