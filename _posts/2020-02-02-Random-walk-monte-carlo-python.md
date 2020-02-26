---
title: "Random Walk and Monte Carlo Simulation in Python"
date: 2020-02-02
tags: [random walk, monte carlo simulation]
header:
excerpt: "Random Walk, Monte Carlo Simulation"
mathjax: "true"
---

## Random Walk
A Random Walk refers to any process in which there is no observable pattern or trend; that is, where the movement of an object, or the values taken by a certain variable, are completely random.  A more formal definition can be read on [Wiki page](https://en.wikipedia.org/wiki/Random_walk) .Certain real-life scenarios that could be modeled as random walk could be:
 
 - The movement of animals in the wilds
 - The path traced by a molecule as it moves through a liquid or a gas
 - The price of a stock as it moves up and down
 - The path of a drunkard walking down the street
 - A gambler's luck in any gambling sites

To further illustrate this concept, let's simulate a Random Walk with a simple example : a random walk around the neighborhood.

Suppose that you live in a perfectly arranged neighborhood such as illustrated below with your location pointed red:

<img src="{{ site.url }}{{ site.baseurl }}/images/Random Walk and Monte Carlo/Grid first location.png" 
alt="Your Neighborhood and Your location (pointed red)">

Now, imagine that your original position as the point zero (0, 0) in the standard X-Y chart:

<img src="{{ site.url }}{{ site.baseurl }}/images/Random Walk and Monte Carlo/Grid with XY NESW.png" 
alt="Your choice of direction North (0, +1), East (1, 0), South (0, -1), West (-1, 0)">

Suppose that you will walk randomly along your neighborhood's line and at each street crossing you will randomly choose which direction to walk to. Say that there is four direction to go North, East, South, and West. Written differently in numbers, at each crossing you will randomly choose to go North (0, +1), East (1, 0), South (0, -1), or West (-1, 0). With each decision, you will add these coordinates to your previous position. The end of your coordinate after you decide to finish your walk is the result of you randomly choose which direction to go to. Of course if you redo the experiment several time, you will end up in a different location then before as it is random where you choose your direction.

This illustrates  the concept of random walking in a simple example.

In Python, it is possible to express random walk concept :

First, import all necessary modules :

```python
#Import package
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statistics as st
```
```python
#Write function
def random_walk_2(n):
    """Return coordinate after 'n' block random walks"""
    x, y = 0, 0 # Intial coordinate position
    for i in range(n):
        (dx, dy) = random.choice([(0,1), (0,-1), (1,0), (-1,0)]) # dx difference in x and dy difference in y
        x += dx
        y += dy
    return (x, y)
```
With this function lets see you coordinate if you walk 15 blocks randomly
```python
random_walk_2(15)
```
```python
(-5, -2)
```
This means that with randomly walk 15 blocks you will end up at 5 blocks to the west and 2 blocks to the south, or 7 blocks away from your original position.

<img src="{{ site.url }}{{ site.baseurl }}/images/Random Walk and Monte Carlo/Grid new coordinate.png" 
alt="Your new coordinate (Red point)">

Now, suppose that if you will only go back home by walk if you are 2 blocks away from your house otherwise you will use a transportation. ***What is the longest walk so that on average you will end up 2 blocks or fewer from home?***

You can repeat the experiment above as many times as possible and calculate the average of ending up 2 blocks or fewer. But, with Monte-Carlo Simulation, you can easily simulate this as many time as you want with as little time as possible.

## Monte Carlo Simulation
Oversimplified, Monte Carlo Simulation is a mathematical technique that generates random variables for modelling. For a more formal definition and explanation you can read the [Wiki page](https://en.wikipedia.org/wiki/Monte_Carlo_method).

The random variables that are goingto be generated in this article is the random choices to be taken at each crossing and repeat the whole process for a designated times.

Lets say that the experiments will be conducted 10,000 times and with each experiments you will walk from 1 blocks to 15 blocks. ***What is the longest walk so that on average you will end up 2 blocks or fewer from home?***

Written in Python as follow:

```python
number_of_walks = 10000 # The how many trials taken for the simulation
blocks_limit = 2 # The how far blocks you can walk without transportation
blocks_walked = 15 # The how many blocks taken for the simulation

walk_calc = [] # Make a dataframe to store loop output

for walk_length in range(1, blocks_walked+1): 
    no_transport = 0 # Number of walks 4 or fewer blocks from home - Counter
    for i in range(number_of_walks): # Start of monte carlo simulation/loop
        (x, y) = random_walk_2(walk_length) # The position of the random walk
        distance = abs(x) + abs(y) # The distance of position to home
        if distance <= blocks_limit: # If the distance is less than 4
            no_transport += 1 # Add 1 to the Counter
    no_transport_percentage = float(no_transport) / number_of_walks # The percentage of walks that requires no transport
    # Stores the output in a list
    walk_calc.append({'Walk_length': walk_length, 'Chance within walking distance (%)': no_transport_percentage*100})

# Transform the output into dataframe object
no_transport_walking = pd.DataFrame(walk_calc).copy()
no_transport_walking
```
The following result:

|    Walk_length    |    Chance   within walking distance (%)|
|-------------------|----------------------------------------|
|    1              |    100.00                              |
|    2              |    100.00                              |
|    3              |    56.57                               |
|    4              |    77.45                               |
|    5              |    38.91                               |
|    6              |    61.11                               |
|    7              |    30.51                               |
|    8              |    50.67                               |
|    9              |    24.79                               |
|    10             |    42.77                               |
|    11             |    19.89                               |
|    12             |    38.27                               |
|    13             |    17.36                               |
|    14             |    32.25                               |
|    15             |    15.55                               |

Now to answer the objective, the word 'average' be defined first. Let's go ahead and assume that anything above 50% meets the criteria. So, we will filter the resul table

```python
no_transport_walking[(no_transport_walking['Chance within walking distance (%)'] > 50)]
```
Which results:

| Walk_length | Chance within walking distance |
|-------------|--------------------------------|
| 1           | 100.00                         |
| 2           | 100.00                         |
| 3           | 56.57                          |
| 4           | 77.45                          |
| 6           | 61.11                          |
| 8           | 50.67                          |

From above, we can see that the longest random walk around your neighbourhood that will result, on average, 2 blocks or fewer is 8 walks from your starting location. Furthermore, if we look at the result, an even random walk has more probability of you ended up closer to home compared to an odd random walk.

## Conclusion

Random walk and Monte Carlo Simulation are has many real world application. Especially those that concerns with the randomness nature of its subject. In the future I will try to use the these method to analyse some stocks in Indonesian stock market.

Thank you for reading this article any feedback would be gladly appreciated!
You can see my full code and further analysis for this article on my [GitHub repo](https://github.com/creixotradeos/Random-Walk-and-Monte-Carlo.git)
