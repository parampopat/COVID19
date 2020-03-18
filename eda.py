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


def write(file, content, mode='a'):
    """
    Writes content to a file
    :param mode: File mode either 'a' or 'w'
    :param file: file name
    :param content:content to be written onto file
    :return:
    """
    f = open(file, mode)
    f.write(content)
    f.close()


def arrange_data(data, dates, group_by='Country/Region'):
    """
    Arranges data into dictionaries.
    :param dates: List of dates over which data is to be truncated
    :param data: Data DataFrame which is to be arranged
    :param group_by: Column header over which data is to be grouped by
    :return:
    """
    count_by_dates = {}
    count_by_countries = defaultdict(list)
    count_by_countries_norm = {}
    group_by_data = data.groupby([group_by])

    # Arrange Data by Dates
    for i in range(len(dates)):
        count_by_dates[dates[i]] = group_by_data[dates[i]].sum()

    # Arrange Data by Country
    for key in count_by_dates.keys():
        for country in count_by_dates[key].keys():
            count_by_countries[country].append(count_by_dates[key][country])

    # Z-Score Normalize the data
    for key in count_by_countries.keys():
        count_by_countries_norm[key] = normalize(count_by_countries[key])

    return group_by_data, count_by_dates, count_by_countries, count_by_countries_norm


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
        write(file=file, content=st, mode='w')

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


def plot_data(labels, data, title='', save=False):
    """
    Plots data
    :param labels: List of data labels to plot
    :param data: Data from which plot is to be generated
    :param title: Title of the chart
    :param save: True to save plot as a png file
    :return:
    """
    for label in labels:
        plt.plot(data[label], label=label)
    plt.title(title)
    plt.xlabel('Days from 1/22/20')
    plt.ylabel('Z-Score Normalized Count')
    plt.legend()
    if save:
        savefile = ""
        for label in labels:
            savefile = savefile + label + '-'
        plt.savefig(savefile + title + '.png', dpi=300)
        plt.close()
    else:
        plt.show()


def analyze(dataset, type, to_plot=False, labels=None, save_plot=True, save_csv=True):
    """

    :param dataset: Dataset DataFrame
    :param type: Type of cases
    :param to_plot: True to plot
    :param labels: Plotting labels
    :param save_plot: True to save plots, False will show the plot
    :param save_csv: True to save correlation data into CSV file
    :return: Correlation Data
    """
    # Sometimes the data source has added an empty column for the current date.
    if np.isnan(dataset.iloc[0, -1]):
        dates = dataset.columns[4:-1]
    else:
        dates = dataset.columns[4:]

    # Group the data by country, this can be changed to region or state as well.
    group_by_country, count_by_dates, count_by_countries, count_by_countries_norm = arrange_data(data=dataset,
                                                                                                 dates=dates)

    # Calculate Maximum Cross-Correlation and Delays.
    file = 'analysis_' + type.lower() + '.csv'
    corr_data = cross_corr(count_by_countries_norm, to_write=save_csv, file=file)

    # Plot Something
    if to_plot:
        if labels is None:
            raise ValueError("Expected input for 'labels'")
        plot_data(labels=labels, data=count_by_countries_norm, title=type, save=save_plot)
    return corr_data


if __name__ == '__main__':
    # Analysis of confirmed cases
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'
    dataset = pd.read_csv(url)
    correlation_confirmed = analyze(dataset=dataset, type=url.split('/')[-1].split('.')[0].split('-')[-1], to_plot=True,
                                    labels=['US', 'Germany'])

    # Analysis of recovered cases
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv'
    dataset = pd.read_csv(url)
    correlation_recovered = analyze(dataset=dataset, type=url.split('/')[-1].split('.')[0].split('-')[-1], to_plot=True,
                                    labels=['US', 'Singapore'])

    # Analysis of death cases
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv'
    dataset = pd.read_csv(url)
    correlation_deaths = analyze(dataset=dataset, type=url.split('/')[-1].split('.')[0].split('-')[-1], to_plot=True,
                                 labels=['US', 'Iran'])
