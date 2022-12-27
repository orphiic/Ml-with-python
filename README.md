# Machine Learning with Python
Machine learning is a subfield of artificial intelligence that involves the use of algorithms and statistical models to enable computers to learn and make decisions based on data, without being explicitly programmed. Machine learning algorithms can be trained on large datasets to recognize patterns, classify data, and make predictions or decisions. There are many different types of machine learning algorithms, including supervised learning algorithms, which learn from labeled training data, and unsupervised learning algorithms, which learn from unlabeled data. Some popular applications of machine learning include image and speech recognition, natural language processing, and predictive modeling. Machine learning is an active area of research and development, with new techniques and approaches being developed all the time.

## How To Start?
It is essential to study mathematics as well as statistics so that you can calculate significant numbers based on data sets.
Learn how to use various Python modules to get the answers we need
And how to make functions that are able to predict the outcome based on what we have learned.

## Dataset
A data set is any collection of data in the mind of a computer. It can range from an array to an entire database.

An array example:
```json
[99,86,87,88,103]
```
| Bike          | Color         |age         | speed      |    feature        
| ------------- | ------------- |------------- | ------------- |------------- |
| Honda  | Red  |4  | 99  | y |         
| Yamaha  | Blue  |7 | 86  | n  | 
| Apache  | white  |8 | 87  | n  |
| Beneli  | Black  |2 | 88  | y  |
| Hero | red  |2 | 103  | n  |  

We can predict the average by glancing at the array, and we can also identify the greatest and lowest values, yet what else can we do?
And we can see from the information that the most prevalent color is red and the oldest bike is 8 years old, but what if we could forecast if a bike has a feature only by looking at the other value systems?
That is the purpose of Machine Learning! Analyzing data and forecasting outcomes!

## Data Types

To evaluate data, we must first understand the sort of data we are working with.  We may divide data types into 3 groups:
- Numerical
- Categorical
- Ordinal

Numerical data are integers that may be divided into two categories:
- Discrete Data refers to numbers that are restricted to integers. As an example, consider the amount of automobiles going past.
- Continuous Data are numbers with an endless value. As an example, consider the price or size of an object.


Categorical data are values that can't be compared to one another. A color value, for example, or any yes/no responses.

Ordinal data are similar to categorical data in that they may be compared. For example, at college, A is better than B, and so on.

## Mean, Median & Mode
The mean, mode, and median are three popular metrics of central tendency that may be used to describe a dataset in machine learning and statistics.
- Mean - The average value
- Median - The mid point value
- Mode - The most common value
For instance, we recorded the speeds of 13 bikes:
```json
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
```

## Mean
The average value is the mean value.

To determine the mean, add all of the values together and divide the total by the number of values:
```json
(99+86+87+88+111+86+103+87+94+78+77+85+86) / 13 = 89.77
```
There is a way for this in the [NumPy](https://numpy.org/) library.

Use NumPy libray to find average:
```python
import numpy as np

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

average = np.mean(speed)

print(average)
```
## Median
When the values in a dataset are sorted ascendingly, the median represents the middle value. If the dataset has an odd number of items, the middle value is the median. If the number of values is even, the median is the mean of the two middle values. The median is less affected by outliers than the mean.

```python
import numpy as np

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

median = numpy.median(speed)

print(median)
```
## Mode
The most common value in a dataset is called the mode. The value that occurs the most frequently across the dataset is this one.

The [SciPy](https://scipy.org/) library has a method to find mode.
Use the SciPy `mode()` method to find mode.
```python
from scipy import stat

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

mode = stats.mode(speed)

print(mode)
```


             
          
