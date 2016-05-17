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
train = pd.read_csv("../input/train.csv", parse_dates=['date_time', 'srch_ci', 'srch_co'], nrows=100000)
train1 = pd.read_csv("../input/train.csv", parse_dates=['date_time', 'srch_ci', 'srch_co'], nrows=10000000)
train_bookings = train1[train1['is_booking'] == 1].drop('is_booking', axis=1)
train_clicks = train1[train1['is_booking'] == 0].drop('is_booking', axis=1)
train_bookingszna = train_bookings.fillna(0)
train_clickszna = train_clicks.fillna(0)
train_bookingszna.shape
train_clickszna.shape
sns.countplot(y='hotel_continent', data=train_bookingszna)
sns.plt.title('Continent wise destination distribution of 10m bookings')
plt.show()
sns.countplot(y='hotel_continent', data=train_clickszna)
sns.plt.title('Continent-wise aspired destination distribution of 10m bookings')
plt.show()
booked_hotels_country = pd.read_csv("../input/test.csv", usecols=['hotel_country'])
booked_hotels_country_bookings, count = np.unique(booked_hotels_country, return_counts=True)
booked_hotels_country_bookings.shape
#len(count[count>=10])
count.min()
np.median(count)
count.max()
most_visited1 = booked_hotels_country_bookings[count == 1329976]
print(most_visited1)
len(most_visited)
most_visited = booked_hotels_country_bookings[count >= 562]
most_visited
train_bookingszna = train_bookingszna[train_bookingszna['hotel_country'].isin(most_visited)]
f, ax = plt.subplots(figsize=(15, 25))
sns.countplot(y='hotel_country', data=train_bookingszna)
plt.title('Bookings of hotels per country for countries with booking greater than mean')
plt.show()
sns.pairplot(train_bookingszna[['hotel_country', 'user_location_country']], size=6)
plt.show()
corr = train_bookingszna.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 15))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, vmax=.3, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
plt.show()