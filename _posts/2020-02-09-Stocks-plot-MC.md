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

## Determine the daily return of the index

Next, we can calculate the daily return of the index. For this article I will use the logarithmic return of the index. 

$$r_t  =  log  ({P_t \over P_{t-1}}) $$

with Pt as the current stock price and Pt-1 is the previous day's stock price.

Logarithmic Return of IHSG:
```python
#Calculate IHSG Logarithmic daily return
IHSG_log_return = np.log(1 + IHSG.iloc[: , 0].pct_change())
IHSG_log_return
```
| Date                                                         	| Return    	|
|--------------------------------------------------------------	|-----------	|
| 2010-01-04                                                   	| NaN       	|
| 2010-01-05                                                   	| 0.011529  	|
| 2010-01-06                                                   	| -0.000760 	|
| 2010-01-07                                                   	| -0.006320 	|
| 2010-01-08                                                   	| 0.010565  	|
| ...                                                          	|           	|
| 2020-01-27                                                   	| -0.017920 	|
| 2020-01-28                                                   	| -0.003597 	|
| 2020-01-29                                                   	| 0.000304  	|
| 2020-01-30                                                   	| -0.009112 	|
| 2020-01-31                                                   	| -0.019596 	|
| Name: (Adj Close, ^JKSE), Length: 2461, dtype: float64</pre> 	|           	|
|                                                              	|           	|
|                                                              	|           	|

## Plotting the return

**Plotting with Python and Matplotlib** is super easy, we only need to select the IHSG_log_return data.
```python
#Ploting the return scatter plot
plt.figure(figsize=(10,6))
plt.scatter(IHSG_log_return.index, IHSG_log_return)
plt.show()
```
The result:
<img src="{{ site.url }}{{ site.baseurl }}/images/Plot and Monte Carlo/Log return scatter plot.png" 
alt="Log return scatter plot">

As we can see, that the majority of the daily return are between -0.025 and +0.025 over the decade.

Now, we can do an attempt to answer the question at the start of this article: to have an estimate of the future stock price.

## Monte Carlo Simulation on Stock Price
A typical way to explain Monte Carlo Simulation is to imagine any type of probability, for example the slot machine image. At any given time, numerous occasions can happen in whenever next step depending on an action.

Technically written, below is the Monte Carlo equation that is going to be used in this article:

$$ P_t = P_{t-1} * e^{(Drift + Random Component)}$$

Since Monte Carlo is formally defined taken from a probability distribution to provide a multi-variable model of present 'what-if' events.
With the equation, it basically contains 3 components:

 - Yesterday's asset price
 - Drift component i.e. the direction of the asset in the past
 - Random component this component is the variables taken from a distribution

For the simplicity sake of this article, we will assume normality of the data for now. We will try to simulate the JKSE stock price for the next 250 days and for each day we will do 100 trials.
Written in Python it will look like this:

```python
np.random.seed(0) # Set seed
u = IHSG_log_return.mean() # mean of log return
var = IHSG_log_return.var() # variance of log return
stdev = IHSG_log_return.std() # stdev of log return
drift = u - (0.5 * var) # calculate drift value

t_intervals = 250 # days of trading
iterations = 100 # how many possibilities are to be calculated
z_value = norm.ppf(np.random.rand(t_intervals, iterations)) # Z value, the random walk element | type: np.ndarray

daily_returns = np.exp(drift + stdev * z_value) # calculate daily return based on random walk | type: np.ndarray
```

```python
S0 = IHSG.iloc[-1:,0:1] # Create variable S0 equal to the last price of IHSG
price_list = np.zeros_like(daily_returns) # Create price_list variable with the same dimension as daily_returns matrix
price_list[0] = S0 # Set the first row to be equal to S0 | type: np.ndarray

# Create a loop in the range (1, t_intervals) that reassigns 
# to the price in time t the product of the price in day (t-1) with the value of the daily returns in t.
for t in range(1, t_intervals):
    price_list[t] = price_list[t-1] * daily_returns[t]
```

Plotting the Monte Carlo Simulation
```python
#Plot the simulation
plt.figure(figsize=(15,10))
plt.plot(price_list)
plt.show()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/Plot and Monte Carlo/Monte Carlo.png" 
alt="Monte Carlo Simulation">

If we look at the result of the simulation, based on 10 years return and from the last retrieved price of JKSE, this composite index can potetially swing from Rp 4,200 to above Rp 8,000 in 250 days. At the latest price of Rp 5,940 and this simulation we can reasonably confident that JKSE index has a better potential of going up in its value.

## Conclusion
Monte Carlo Simulation is yet another tool in helping to make a investment decision. It has an edge in its capacity to take into consideration different qualities and random movement for different possibilities. Of course, this tool simply cannot accurately simulate an extreme events such as a sudden market crash or sudden surge in price in a short time. This is the limitation of this simulation to predict the movement of the stock market. 

In the future, I will try to make a more in-depth analysis on this simulation to help in making a better decision in investing in stock market. 

Thank you for reading this article any feedback would be gladly appreciated!
You can see my full code and further analysis for this article on my [GitHub repo](https://github.com/creixotradeos/Random-Walk-and-Monte-Carlo.git)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE5MzQ5MTg3MjYsLTIwMDAyNTU4ODAsLT
Q5ODIyMjQ5NSwtNTE5Mzg3NjIxLDEwNzQzODk1ODQsODg1MjYy
MTEzLDE4Mzc2MzUxMDRdfQ==
-->