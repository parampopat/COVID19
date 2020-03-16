"""
__author__ = "Param Popat"
__version__ = "1.0"
__git__ = "https://github.com/parampopat/"
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
np.seterr(divide='ignore', invalid='ignore')


def normalize(a):
    """
    Calculates Z score
    :param a: Array to be normalized
    :return: Normalized Array
    """
    a = (a - np.mean(a)) / (np.std(a))
    return a


# dataset = pd.read_csv('COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')
dataset = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')
dates = dataset.columns[4:]
group_by_country = dataset.groupby(['Country/Region'])
count_by_dates = {}
count_by_countries = defaultdict(list)
count_by_countries_norm = {}

# Arrange Data by Dates
for i in range(len(dates)):
    count_by_dates[dates[i]] = group_by_country[dates[i]].sum()

# Arrange Data by Country
for key in count_by_dates.keys():
    for country in count_by_dates[key].keys():
        count_by_countries[country].append(count_by_dates[key][country])

# Z-Score Normalize the data
for key in count_by_countries.keys():
    count_by_countries_norm[key] = normalize(count_by_countries[key])

# Calculate Maximum Cross-Correlation and Delays.
fopen = open('all_all.csv', "a")
st = str("Country_1,Country_2,Delay,Pearson")
fopen.write(st)
fopen.close()
for key in count_by_countries_norm.keys():
    for key_1 in count_by_countries_norm.keys():
        if key_1 != key:
            x = []
            rel_corr = np.correlate(count_by_countries_norm[key], count_by_countries_norm[key_1], mode='full')
            for i in range(rel_corr.__len__()):
                x.append(i - (rel_corr.__len__() / 2.0))
            max_index = np.ceil(x[np.argmax(rel_corr)])
            fopen = open('all_all.csv', "a")
            st = str(key.replace(',', '')) + "," + str(key_1.replace(',', '')) + "," + str(max_index) + "," + str(
                rel_corr.max())
            fopen.write("\n" + st)
            fopen.close()

plt.plot(count_by_countries_norm['Italy'], label='Italy')
plt.plot(count_by_countries_norm['Lebanon'], label='Lebanon')
plt.legend()
plt.show()
