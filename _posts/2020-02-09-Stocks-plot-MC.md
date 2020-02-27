---
title: "Plotting and simulating stocks using Monte Carlo Simulation in Python"
date: 2020-02-09
tags: [monte carlo simulation, stocks, finance]
header:
excerpt: "Monte Carlo Simulation, Stocks, Finance"
mathjax: "true"
---

Most investors are continually attempt to discover a solution to a basic question, _What is the movement of the market in the future_.

Fundamentally speaking, nobody can infallibly claim to accurately answer that question. That is why the majority of investors and analyst go through a really long effort attempting to estimate the future of an assest's price. One of tools to do this is using **Python** languange to assist in analyzing large amount of data of the stock market.

In this article, we will write a script that will allow us to:

 - Get market data with Python and Pandas package
 - Determine the daily return of the stock
 - Plot the daily return with a chart
 - Simulate the movement of the stock price in the future with Monte Carlo Simulation
 
Monte Carlo Simulation is a mathematical technique that generates random variables for modelling. I have briefly written a simple explanation of the simulation [here](https://creixotradeos.github.io/Random-walk-monte-carlo-python/).

## Get market data

First, before attempting to analyze a data, we first must gather the data. In Python, stock price data can be acquired with a relatively easy and automatic method.  For this article, I will use Jakarta Stock Exchance Composite Index with the ticker of '^JKSE' or more commonly known in Indonesia as _IHSG (Indeks Harga Saham Gabungan_. This data can be extracted using Pandas DataReader and for the price source I will use Yahoo Finance Daily Reader.

Let's import the necessary packages for this article:
```python
# Import Libraries
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
%matplotlib inline
import datetime
from statsmodels.graphics.gofplots import qqplot
import scipy.stats as stats
from scipy.stats import norm
import pylab
```

Next, set the starting date and the end date for the data that is going to be retrieved:
```python
#Set start date and end date
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2020, 2, 1)
```
```python
#Get IHSG Data
IHSG = wb.DataReader(['^JKSE'], 'yahoo', start, end)
IHSG
```

Here is the result:

| Attributes | Adj Close   | Close       | High        | Low         | Open        | Volume     |
|------------|-------------|-------------|-------------|-------------|-------------|------------|
| Symbols    | ^JKSE       | ^JKSE       | ^JKSE       | ^JKSE       | ^JKSE       | ^JKSE      |
| Date       |             |             |             |             |             |            |
| 2010-01-04 | 2575.336670 | 2575.413086 | 2576.055908 | 2532.895996 | 2533.947998 | 18339300.0 |
| 2010-01-05 | 2605.199707 | 2605.277100 | 2606.069092 | 2575.616943 | 2575.616943 | 57043800.0 |
| 2010-01-06 | 2603.219727 | 2603.297119 | 2622.115967 | 2587.709961 | 2605.480957 | 51569100.0 |
| 2010-01-07 | 2586.818115 | 2586.895020 | 2611.603027 | 2570.272949 | 2603.500977 | 45510800.0 |
| 2010-01-08 | 2614.292480 | 2614.370117 | 2614.535889 | 2583.846924 | 2586.792969 | 73723500.0 |
| ...        | ...         | ...         | ...         | ...         | ...         | ...        |
| 2020-01-27 | 6133.208008 | 6133.208008 | 6242.176758 | 6130.928223 | 6240.817871 | 43723000.0 |
| 2020-01-28 | 6111.184082 | 6111.184082 | 6111.184082 | 6111.184082 | 6111.184082 | 0.0        |
| 2020-01-29 | 6113.044922 | 6113.044922 | 6152.588867 | 6102.795898 | 6123.095215 | 34605300.0 |
| 2020-01-30 | 6057.596191 | 6057.596191 | 6057.596191 | 6057.596191 | 6057.596191 | 0.0        |
| 2020-01-31 | 5940.047852 | 5940.047852 | 6078.930176 | 5937.021973 | 6076.458984 | 41508700.0 |

As we can see, we have 6 informations of the daily stock price the Adjusted Closing Price, the Closing Price, the High price of the day, the Low price of the day, the Opening Price, and the trading Volume of the day.


<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE5NjQyNDI5NDVdfQ==
-->