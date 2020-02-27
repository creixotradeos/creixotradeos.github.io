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

First, before attempting to analyze a data, we first must gather the data. In Python, stock price data can be acquired with a relatively easy and automatic method.  For this article

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTExODkyODUxNjldfQ==
-->