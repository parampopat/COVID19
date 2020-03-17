"""
__author__ = "Param Popat"
__version__ = "1.0"
__git__ = "https://github.com/parampopat/"
__data__ = "https://github.com/CSSEGISandData/COVID-19"
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


def write(file, content):
    """
    Writes content to a file
    :param file: file name
    :param content:content to be written onto file
    :return:
    """
    f = open(file, "a")
    f.write(content)
    f.close()


def cross_corr(data, to_write=False, file=None):
    """
    Calculates Maximum Cross-Correlation and Delays.
    :param file: File name
    :param data: Dictionary of Data
    :param to_write: True to write to a csv
    :return:
    """
    if to_write:
        if file is None:
            raise ValueError("Expected input for 'file'")
        st = str("Country_1,Country_2,Delay,Pearson")
        write(file=file, content=st)

    max_indices = defaultdict(list)
    for key in data.keys():
        for key_1 in data.keys():
            if key_1 != key:
                x = []
                rel_corr = np.correlate(data[key], data[key_1], mode='full')
                for i in range(rel_corr.__len__()):
                    x.append(i - (rel_corr.__len__() / 2.0))
                max_index = np.ceil(x[np.argmax(rel_corr)])
                max_indices[key].append({key_1: [max_index, rel_corr.max()]})
                if to_write:
                    st = str(key.replace(',', '')) + "," + str(key_1.replace(',', '')) + "," + str(
                        max_index) + "," + str(
                        rel_corr.max())
                    write(file=file, content="\n" + st)
    return max_indices


dataset_confirmed = pd.read_csv(
    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')
dataset_recovered = pd.read_csv(
    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')
dataset_deaths = pd.read_csv(
    'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')

dataset_used = dataset_confirmed
if np.isnan(dataset_used.iloc[0, -1]):
    dates = dataset_used.columns[4:-1]
else:
    dates = dataset_used.columns[4:]

group_by_country = dataset_confirmed.groupby(['Country/Region'])
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
file = 'all_all.csv'

# Plot Something
corr_data = cross_corr(count_by_countries_norm, to_write=True, file=file)

plt.plot(count_by_countries_norm['Italy'], label='Italy')
plt.plot(count_by_countries_norm['Lebanon'], label='Lebanon')
plt.legend()
plt.show()
