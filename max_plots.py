"""
__author__ = "Param Popat"
__version__ = "1.0"
__git__ = "https://github.com/parampopat/"
"""

import pandas as pd


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


if __name__ == '__main__':
    # Analysis of confirmed cases
    data = pd.read_csv('analysis_confirmed.csv')
    max_confirmed = max_analysis(data)
    max_confirmed.to_csv('max_confirmed.csv', index=False)

    # Analysis of recovered cases
    data = pd.read_csv('analysis_recovered.csv')
    max_confirmed = max_analysis(data)
    max_confirmed.to_csv('max_recovered.csv', index=False)

    # Analysis of death cases
    data = pd.read_csv('analysis_deaths.csv')
    max_confirmed = max_analysis(data)
    max_confirmed.to_csv('max_deaths.csv', index=False)