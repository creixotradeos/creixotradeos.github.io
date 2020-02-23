# Objective : What is the longest random walk you can take so that on average you will end up 4 blocks or fewer from home?

## First version function - Simple version


```python
#Import package
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
```


```python
#Write function

def random_walk(n):
    """Return coordinates after 'n' block of random walk"""
    x = 0 # Initial position
    y = 0 # Initial position
    for i in range(n):
        step = random.choice(['N','S','E','W'])
        if step == 'N':
            y = y + 1 # Take 1 block step to the north
        elif step == 'S':
            y = y - 1 # Take 1 block step to the south
        elif step == 'E':
            x = x + 1 # Take 1 block step to the east
        else:
            x = x - 1 # Take 1 block step to the west
    return (x, y) # Return the coordinate as a tuple
```

To test the random_walk function : take 25 random walks, each 10 blocks long.
The distance from home is the sum of the x and y coordinate(absolute)


```python
#Create Loop expressing the above statement
for i in range(25): # 25 random walks
    walk = random_walk(10) # insert the function of walking 10 blocks each session
    print(walk, "Distance from home = ", abs(walk[0])+abs(walk[1])) # Sum of the x and y coordinate
```

    (0, -2) Distance from home =  2
    (1, -3) Distance from home =  4
    (-3, -1) Distance from home =  4
    (1, 3) Distance from home =  4
    (4, 0) Distance from home =  4
    (3, 1) Distance from home =  4
    (-7, -1) Distance from home =  8
    (-1, 3) Distance from home =  4
    (2, 0) Distance from home =  2
    (0, 0) Distance from home =  0
    (0, -4) Distance from home =  4
    (2, -2) Distance from home =  4
    (2, -2) Distance from home =  4
    (-5, -5) Distance from home =  10
    (-1, 3) Distance from home =  4
    (1, -1) Distance from home =  2
    (-1, -1) Distance from home =  2
    (-2, 4) Distance from home =  6
    (-1, 5) Distance from home =  6
    (-2, 2) Distance from home =  4
    (1, 1) Distance from home =  2
    (0, -2) Distance from home =  2
    (-1, -1) Distance from home =  2
    (-2, -4) Distance from home =  6
    (-1, -5) Distance from home =  6
    

## Second version function - Compact Version


```python
#Write function
def random_walk_2(n):
    """Return coordinate after 'n' block random walks"""
    x, y = 0, 0 # Intial coordinate position
    for i in range(n):
        (dx, dy) = random.choice([(0,1), (0,-1), (1,0), (-1,0)]) # dx difference in x (dx) and difference in y (dy)
        x += dx
        y += dy
    return (x, y)
```

To test the random_walk function : take 25 random walks, each 10 blocks long.
The distance from home is the sum of the x and y coordinate(absolute)


```python
#Create Loop expressing the above statement
random.seed(0)
for i in range(25): # take 25 random walks
    walk = random_walk_2(10) # 10 blocks each
    print(walk, "Distance from home = ", abs(walk[0])+abs(walk[1])) # the distance from the last position to home
```

    (-2, 0) Distance from home =  2
    (4, 0) Distance from home =  4
    (-1, 1) Distance from home =  2
    (0, 2) Distance from home =  2
    (0, 0) Distance from home =  0
    (2, 0) Distance from home =  2
    (1, -1) Distance from home =  2
    (-1, -1) Distance from home =  2
    (1, 1) Distance from home =  2
    (2, 0) Distance from home =  2
    (0, 4) Distance from home =  4
    (-1, 3) Distance from home =  4
    (1, 1) Distance from home =  2
    (-1, 1) Distance from home =  2
    (1, -1) Distance from home =  2
    (-1, 1) Distance from home =  2
    (4, -2) Distance from home =  6
    (0, -2) Distance from home =  2
    (-6, 2) Distance from home =  8
    (-1, 1) Distance from home =  2
    (-2, 2) Distance from home =  4
    (2, 0) Distance from home =  2
    (-1, 3) Distance from home =  4
    (4, 2) Distance from home =  6
    (-1, -1) Distance from home =  2
    

Using Monte Carlo simulation, try to solve the objective

Conduct thousand of random trials and compute the percentage of random walks that ends in short walk to home

Therefore the longest random walk with highest chances on average (half stdev from the mean) to end up several blocks or fewer from home is:


```python
random.seed(1)
number_of_walks = 20000 # The how many trials/walks taken for the simulation
blocks_limit = 4 # The how far blocks you can walk without transportation
blocks_walked = 30 # The how many blocks taken for the simulation

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
    walk_calc.append({'Walk_length': walk_length, 'Chance within walking distance': no_transport_percentage*100})

# Transform the output into dataframe object
no_transport_walking = pd.DataFrame(walk_calc)

walk_prob = no_transport_walking['Chance within walking distance'] # Store the probability in walk_prob variable
walk_prob_mean, walk_prob_std = walk_prob.mean(), walk_prob.std() # Calculate the mean and std of the output
# Test if the output is normally distributed
k, pvalue = stats.normaltest(no_transport_walking['Chance within walking distance']) 

n_halfstd = (0.33/2) * blocks_walked # How many n probability within 0.5 std
if pvalue > 0.05:
    # Filter walk_prob within 0.5 stdev from the mean
    within_halfstd = no_transport_walking[
        (no_transport_walking['Chance within walking distance'] <= (walk_prob_mean + 0.5*walk_prob_std)) 
        & (no_transport_walking['Chance within walking distance'] >= (walk_prob_mean - 0.5*walk_prob_std)) ]
    print( within_halfstd.sort_values('Walk_length', ascending=False).head(n=1) )
else:
    print("Data are not normally distributed")
    # Filter n_halfstd closest value to the mean
    closest_prob = no_transport_walking.iloc[(no_transport_walking['Chance within walking distance'] 
                                             - walk_prob_mean).abs().argsort()[:int(n_halfstd)]]
    print( closest_prob.sort_values('Walk_length', ascending=False).head(n=1) )
    print("")
```

        Walk_length  Chance within walking distance
    21           22                          50.605
    
