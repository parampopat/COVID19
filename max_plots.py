"""
__author__ = "Param Popat"
__version__ = "1.0"
__git__ = "https://github.com/parampopat/"
"""

import pandas as pd
import json


def max_analysis(data):
    """
    Gets maximum correlated countries for each country
    :param data: Dataframe of data
    :return: Dataframe of analyzed maxima per each country
    """
    rows = []
    for key in data['Country_1'].unique():
        rows.append(data[data['Country_1'] == key].sort_values(by=['Pearson'], ascending=False).iloc[0, :].values)
    return pd.DataFrame(rows, columns=data.columns)


def plot_max(data, url):
    pass


if __name__ == '__main__':
    f = open('url.json', "r")
    url = json.load(f)

    # Analysis of confirmed cases
    data = pd.read_csv('analysis_time_series_covid19_confirmed_global.csv')
    max_confirmed = max_analysis(data)
    max_confirmed.to_csv('max_confirmed.csv', index=False)

    # Analysis of recovered cases
    data = pd.read_csv('analysis_time_series_covid19_recovered_global.csv')
    max_confirmed = max_analysis(data)
    max_confirmed.to_csv('max_recovered.csv', index=False)

    # Analysis of death cases
    data = pd.read_csv('analysis_time_series_covid19_deaths_global.csv')
    max_confirmed = max_analysis(data)
    max_confirmed.to_csv('max_deaths.csv', index=False)
