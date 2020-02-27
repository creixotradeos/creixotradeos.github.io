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
IHSG['Adj Close'].tail(n=350).plot()
```

Here

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE3MzA3MDM5NTddfQ==
-->