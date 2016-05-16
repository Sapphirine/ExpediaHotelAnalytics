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
train_bookings_package = train_bookings[train_bookings['is_package'] == 1]
train_clicks_package = train_clicks[train_clicks['is_package'] == 1]
train_bookings_nonpackage = train_bookings[train_bookings['is_package'] == 0]
train_clicks_nonpackage = train_clicks[train_clicks['is_package'] == 0]
# booked_hotels_country_bookings, count = np.unique(train_clicks_nonpackage, return_counts=True)


sns.set(style="darkgrid")
ax = sns.countplot(x="search_span", hue="is_package", data=train_clicks_nonpackage)
sns.countplot(y='search_span', data=train_clicks_nonpackage)
sns.plt.title('stay_span wise destination distribution of 10m bookings')
plt.show()