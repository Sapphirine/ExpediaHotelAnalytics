# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
%matplotlib inline
import numpy as np
import pandas as pd 
#print(check_output(["ls", "../input"]).decode("utf8"))
import datetime
# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv", parse_dates=['date_time'], nrows=10000000)
train1 = pd.read_csv("../input/train.csv", parse_dates=['date_time'], nrows=1000000)
train1['total'] = train1['srch_adults_cnt']+train1['srch_children_cnt']
import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(train['user_location_country'], label="User country")
sns.distplot(train['hotel_country'], label="Hotel country")
plt.legend()
train1["srch_ci"] = pd.to_datetime(train1["srch_ci"], format='%Y-%m-%d', errors="coerce")
train1["srch_co"] = pd.to_datetime(train1["srch_co"], format='%Y-%m-%d', errors="coerce")
train1["stay_span"] = (train1["srch_co"] - train1["srch_ci"]).astype('timedelta64[D]')
xmin = train1['total'].min()
xmax = train1['total'].max()
ymin = train1['hotel_country'].min()
ymax = train1['hotel_country'].max()
train1.plot.hexbin(x='total', y='hotel_country', C='stay_span', gridsize=30,xscale = 'linear', yscale = 'linear')