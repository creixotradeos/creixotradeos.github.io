---
title: "Iris Dataset : Pattern Recognition"
date: 2019-01-07
tags: [data analytics, pattern recognition, iris]
header:
excerpt: "Data Analytics, Pattern Recognition"
mathjax: "true"
---


# Iris Dataset : Pattern Recognition

This is my take on the Iris Flower Dataset. Which is a multivariate data set introduced by British statiscian and biologist Ronald Fisher in 1936 paper as an example of linear discriminant analysis.

The data set consists of 50 samples from each of three species of Iris flower (Iris setosa, Iris virginica, and Iris versicolor). Four features were measured: the length and width of the sepals and petals, all measured in centimeters.

Here is my attempt to recognise species of Iris flower based on the measured features using Probability Density Function (PDF) and Cumulative Density Function (CDF). This analysis is done in Python using Jupyter Notebook.

## Import all necessary libraries


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
```

## Load Iris.csv into a Pandas dataframe


```python
iris = pd.read_csv("iris.csv", header=None) #Dataframe upload
iris = iris.rename(columns={0:"Sepal.Length", 1:"Sepal.Width", 2:"Petal.Length", 3:"Petal.Width", 4:"Species"})
iris.head(n=10) #Display first 10 row from the dataset

```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sepal.Length</th>
      <th>Sepal.Width</th>
      <th>Petal.Length</th>
      <th>Petal.Width</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.4</td>
      <td>3.9</td>
      <td>1.7</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4.6</td>
      <td>3.4</td>
      <td>1.4</td>
      <td>0.3</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5.0</td>
      <td>3.4</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4.4</td>
      <td>2.9</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4.9</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.1</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>



## Basic Statistical Analysis - Central Tendency and Spread of Data

###Determine the mean and the median of the features present in the data set

```python
iris.groupby('Species').agg(['mean', 'median'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">Sepal.Length</th>
      <th colspan="2" halign="left">Sepal.Width</th>
      <th colspan="2" halign="left">Petal.Length</th>
      <th colspan="2" halign="left">Petal.Width</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>Species</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Iris-setosa</th>
      <td>5.006</td>
      <td>5.0</td>
      <td>3.418</td>
      <td>3.4</td>
      <td>1.464</td>
      <td>1.50</td>
      <td>0.244</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>Iris-versicolor</th>
      <td>5.936</td>
      <td>5.9</td>
      <td>2.770</td>
      <td>2.8</td>
      <td>4.260</td>
      <td>4.35</td>
      <td>1.326</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>Iris-virginica</th>
      <td>6.588</td>
      <td>6.5</td>
      <td>2.974</td>
      <td>3.0</td>
      <td>5.552</td>
      <td>5.55</td>
      <td>2.026</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>

With the above result, the difference of values between the mean and median of each species' variables are found to be pretty close. This indicates that data is nearly symmetrically distributed with insignificant presence of outliers.

### Determine the standard deviation


```python
iris.groupby('Species').std()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sepal.Length</th>
      <th>Sepal.Width</th>
      <th>Petal.Length</th>
      <th>Petal.Width</th>
    </tr>
    <tr>
      <th>Species</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Iris-setosa</th>
      <td>0.352490</td>
      <td>0.381024</td>
      <td>0.173511</td>
      <td>0.107210</td>
    </tr>
    <tr>
      <th>Iris-versicolor</th>
      <td>0.516171</td>
      <td>0.313798</td>
      <td>0.469911</td>
      <td>0.197753</td>
    </tr>
    <tr>
      <th>Iris-virginica</th>
      <td>0.635880</td>
      <td>0.322497</td>
      <td>0.551895</td>
      <td>0.274650</td>
    </tr>
  </tbody>
</table>
</div>

Standard deviation is an indication of how widely the data is spread around the mean.

## Box Plot and Violin Plot

### Box Plot

Box plot displays a summary of a large amount of data in five numbers: minimum, lower quartile(25th percentile), median(50th percentile), upper quartile(75th percentile), and maximum data values.  

```python
#Box Plot

#Plotting the box plot using Seaborn library

sns.set(style="ticks")
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
sns.boxplot(x='Species', y='Sepal.Length', data=iris)
plt.subplot(2,2,2)
sns.boxplot(x='Species', y='Sepal.Width', data=iris)
plt.subplot(2,2,3)
sns.boxplot(x='Species', y='Petal.Length', data=iris)
plt.subplot(2,2,4)
sns.boxplot(x='Species', y='Petal.Width', data=iris)
plt.show()
```


<img src="{{ site.url }}{{ site.baseurl }}/images/Iris Pattern Recog/Boxplot.png" alt="Boxplot">


The isolated points that can be seen in the box-plots above are the outliers in the data. Since these are very few in number, it wouldn't have any significant impact on the analysis.

### Violin Plot
Violin plot, similar to the boxp plot, shows the distriution of data accross several levels of one (or more) categorical variables(in this case flower species) such that those distributions can be visually compared. However, unlike box plot, violin plot shows the kernel density estimation of the underlying distribution.

```python
#Violin Plot

sns.set(style="whitegrid")
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species', y='Sepal.Length', data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='Species', y='Sepal.Width', data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='Species', y='Petal.Length', data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='Species', y='Petal.Width', data=iris)
plt.show()
```


<img src="{{ site.url }}{{ site.baseurl }}/images/Iris Pattern Recog/Violin plot.png" alt="Violin Plot">


Violin plots typically are more informative as compared to the box plots as violin plots also represent the underlying distribution of the data in addition to the statistical summary.

## Probability Density Function (PDF) & Cumulative Distribution Function (CDF)

Uni-variate as the name suggests is one variable analysis. The objective is to be able to correctly identify the specie of Iris flower given it’s features — sepal length, sepal width, petal length and petal width. Which among the four features is more useful than other variables in order to distinguish between the species of Iris flower ? To answer this, we will plot the probability density function(PDF) with each feature as a variable on X-axis and it’s histogram and corresponding kernel density plot on Y-axis.

Before we begin further analysis, we need to split the Data Frame according to the 3 distinct class-labels — Setosa, Versicolor and Virginica.


```python
iris_setosa = iris[iris["Species"]=="Iris-setosa"]
iris_versicolor = iris[iris["Species"]=="Iris-versicolor"]
iris_virginica = iris[iris["Species"]=="Iris-virginica"]
```

Plotting the Histogram & PDF using Seaborn FacetGrid object


```python
#FacetGrid object visualize
sns.FacetGrid(iris, hue="Species", size=5) \
   .map(sns.distplot, "Sepal.Length") \
       .add_legend(); #Plot 1
sns.FacetGrid(iris, hue="Species", size=5) \
   .map(sns.distplot, "Sepal.Width") \
       .add_legend(); #Plot 2
sns.FacetGrid(iris, hue="Species", size=5) \
   .map(sns.distplot, "Petal.Length") \
       .add_legend(); #Plot 3
sns.FacetGrid(iris, hue="Species", size=5) \
   .map(sns.distplot, "Petal.Width") \
       .add_legend(); #Plot 4
plt.show()
```


<img src="{{ site.url }}{{ site.baseurl }}/images/Iris Pattern Recog/Sepal Length - Fig 1.png" alt="Fig 1">

The density plot above(Plot 1) reveals that there is a significant amount of overlap between the species on sepal length, so it wouldn’t be a good idea to consider sepal length as a distinctive feature in our uni-variate analysis.

<img src="{{ site.url }}{{ site.baseurl }}/images/Iris Pattern Recog/Sepal Width - Fig 2.png" alt="Fig 2">

With sepal width as a classification feature(Plot 2), the overlap is even more than sepal length as seen in Plot 1 above. The spread of the data is also high. So, again we cannot make any comment on the specie of the flower given it’s sepal width only.

<img src="{{ site.url }}{{ site.baseurl }}/images/Iris Pattern Recog/Petal Length - Fig 3.png" alt="Fig 3">

The density plot of petal length alongside(Plot 3) looks promising from the point of view of uni-variate classification. The Setosa species are well separated from Versicolor and Virginica, although there is some overlap between the Versicolor and Virginica, but not as bad as the the above two plots.

<img src="{{ site.url }}{{ site.baseurl }}/images/Iris Pattern Recog/Petal Width - Fig 4.png" alt="Fig 4">

The density plot of petal width alongside(Plot 4) also looks good. There is slight intersection between the Setosa and Versicolor species, while the overlap between the Versicolor and Virginica is somewhat similar to that of petal length(Plot 3).

To summarize, if we have to choose one feature for classification, we will pick petal length (Plot 3) to distinguish among the species. If we have to select two features, then we will choose petal width as the second feature, but then again it would be a wiser to look at pair-plots(bi-variate and multivariate analysis) to determine which two features are most useful in classification.

We have already established above how petal length could stand out as an useful metric to differentiate between the species of Iris flower. From our preliminary investigation, below pseudo-code can be constructed —


```python
sns.FacetGrid(iris, hue="Species", size=5) \
   .map(sns.distplot, "Petal.Length") \
       .add_legend(); #Plot 3
plt.show()
```


<img src="{{ site.url }}{{ site.baseurl }}/images/Iris Pattern Recog/Petal Length - Fig 3.png" alt="Fig 3">



If petal_length < 2.1
then species = ‘Setosa’
else if petal_length > 2.1 and petal_length < 4.8
then species = ‘Versicolor’
else if petal_length > 4.8
then species = ‘Virginica’
*all lengths are in centimeters.

To further find out how much proportion or perentage of Versicolor and Virginica flowers will be incorrectly classified with above analysis - Cumulative distribution plots is used

```python
plt.figure(figsize=(15,10))

counts, bin_edges_setosa = np.histogram(iris_setosa['Petal.Length'], bins=10, density= True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges_setosa[1:], pdf, label='Setosa PDF')
plt.plot(bin_edges_setosa[1:], cdf, label='Setosa CDF')

counts, bin_edges_versicolor = np.histogram(iris_versicolor['Petal.Length'], bins=10, density= True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges_versicolor[1:], pdf, label='Versicolor PDF')
plt.plot(bin_edges_versicolor[1:], cdf, label='Versicolor CDF')

counts, bin_edges_virginica = np.histogram(iris_virginica['Petal.Length'], bins=10, density= True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges_virginica[1:], pdf, label='Virginica PDF')
plt.plot(bin_edges_virginica[1:], cdf, label='Virginica CDF')

plt.legend()
plt.show()
```


<img src="{{ site.url }}{{ site.baseurl }}/images/Iris Pattern Recog/PDF-CDF.png" alt="PDF-CDF">

From the above results, several insights can be taken :


  Percentage of Virginica Petal Length under 5 cm :


```python
vers_above_5 = (max(bin_edges_versicolor[1:])-5) / (max(bin_edges_versicolor[1:]) - min(bin_edges_versicolor[1:]))
vers_above_5 = vers_above_5 * 100
print str(round(vers_above_5, 2)) + "%" 
```

    5.29%
    

  Percentage of Virginica Petal Length under 5 cm :


```python
virg_below_5 = (5-min(bin_edges_virginica[1:])) / (max(bin_edges_virginica[1:])-min(bin_edges_virginica[1:]))
virg_below_5 = virg_below_5 * 100
print str(round(virg_below_5, 2)) + "%"
```

    12.04%
    

From the above CDF plots, it can be seen that 100 % of the Setosa flower species have petal length less than 1.9. Near about 95 % of the Versicolor flowers have petal length less than 5, while about 12% of the Virginica flowers have petal length less than 5. So, we will incorporate our newly found insights into our previously written pseudo-code to construct a simple uni-variate ‘classification model’.

If petal_length < 1.9
then specie = ‘Setosa’
(accuracy = 100%)
else if petal_length > 3.2 and petal_length < 5
then specie = ‘Versicolor’
(accuracy = 94.71%)…
…else if petal_length > 5
then specie = ‘Virginica’
(accuracy = 87.96%)

Thus by using the cumulative distribution plot, we get a better picture and robust understanding of distribution leading to formulation of simple uni-variate classification model.
