# Machine Learning with Python
Machine learning is a subfield of artificial intelligence that involves the use of algorithms and statistical models to enable computers to learn and make decisions based on data, without being explicitly programmed. Machine learning algorithms can be trained on large datasets to recognize patterns, classify data, and make predictions or decisions. There are many different types of machine learning algorithms, including supervised learning algorithms, which learn from labeled training data, and unsupervised learning algorithms, which learn from unlabeled data. Some popular applications of machine learning include image and speech recognition, natural language processing, and predictive modeling. Machine learning is an active area of research and development, with new techniques and approaches being developed all the time.

## How To Start?
It is essential to study mathematics as well as statistics so that you can calculate significant numbers based on data sets.
Learn how to use various Python modules to get the answers we need
And how to make functions that are able to predict the outcome based on what we have learned.

## Dataset
A data set is any collection of data in the mind of a computer. It can range from an array to an entire database.

An array example:
```
[99,86,87,88,103]
```
![image](https://user-images.githubusercontent.com/67673221/209763352-bc7d0c98-05f4-40dd-b734-93d5141a37a9.png)  

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
```
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
```

## Mean
The average value is the mean value.

To determine the mean, add all of the values together and divide the total by the number of values:
```
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
# Standard Deviation
A collection of data's spread or dispersion is measured by standard deviation. It is a means to express how much variance or dispersion there is in a group of data.
You must first determine the mean (average) of a collection of data before you can determine the standard deviation. Afterward, you deduct the mean from each data point and square the outcome (to make it positive). These squared differences are then added, and the result is divided by the total number of data points. To obtain the standard deviation, you take the square root of this result.

You must first determine the mean (average) of a collection of data before you can determine the standard deviation. Afterward, you deduct the mean from each data point and square the outcome (to make it positive). These squared differences are then added, and the result is divided by the total number of data points. To obtain the standard deviation, you take the square root of this result.

The majority of the data are likely close to the mean (average) value if the standard deviation is low.

When the standard deviation is large, the values are more evenly distributed.

Example: This time, we recorded the speeds of seven vehicles:
```
speed = [86,87,88,86,87,85,86]
```
This is the standard deviation:
```
0.9
```
Hence, the majority of the values fall within a 0.9 standard deviation range of the average value, which is 86.4.

Let's try the same thing with a larger range of numbers:
```
speed = [32,111,138,28,59,77,97]
```
This is the standard deviation:
```
37.85
```
Consequently, the majority of values fall between 37.85 and 77.4, which is the mean value.

As you can see, a greater standard deviation denotes a broader range of values.

The standard deviation may be determined using the NumPy module's formula:

Find the standard deviation using the NumPy `std()` method:
```python
import numpy as np 

speed = [86,87,88,86,87,85,86]

std = np.std(speed)

print(std)
```
Output:
`0.9035079029052513`

# Percentiles
A percentile is a measure that indicates the value below which a given percentage of observations in a group of observations fall. For example, the 50th percentile is the value below which 50% of the observations fall. Percentiles are commonly used to understand the distribution of a set of data.

To calculate percentiles, you first need to order the data from smallest to largest. Then, you can use the following formula to find the value at a given percentile:

-$$ Value = (p/100) \times N $$

Where:

- $Value$ is the value at the desired percentile
- $p$ is the percentile (expressed as a decimal)
- $N$ is the total number of values in the data set
For example, consider a data set with the following values: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}. To find the 50th percentile (also known as the median), we first order the values from smallest to largest: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}. The 50th percentile is the value below which 50% of the observations fall. There are 10 values in the data set, so we can use the following formula to find the value at the 50th percentile:

- $$ Value = (50/100) \times 10 = 5 $$

The 50th percentile (or median) is 5. This means that 50% of the values in the data set are less than or equal to 5, and 50% are greater than 5.

Percentiles are often used to understand the distribution of a data set and to compare values within a data set. For example, you might use percentiles to understand how a student's test score compares to the scores of other students in the same class.

Let's use an example where we have a list of all the residents on a street, organized by age.
```
ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
```
What percentage is the 75%? Since the answer is 43, 75% of the population falls inside this age range.

The specified percentile may be found using a method in the NumPy module:

```
# use NumPy `percentile()` method to find the percentiles:

import numpy as np

ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]

percentiles = np.percentile(ages, 75)

print(percentiles)
```
Output: `43.0`

# Data distribution
Earlier in this course, we used very little data in our examples in order to better comprehend the various concepts.

The data sets in the actual world are significantly larger, yet collecting real world data can be challenging, at least in the beginning stages of a project.

## Are Big Data Sets Available?
We utilize the Python package NumPy, which has a number of ways to build random data sets, to generate large data sets for testing.

```python

# Create an array containing 250 random floats between 0  and 5:

import numpy as np

x = numpy.random.uniform(0.0, 5.0, 250)

print(x)
```
Output: `[4.70107872 0.24787675 0.2195555  3.06204253 4.77830208 4.3809458 
 4.57322999 4.36807984 1.67478644 3.68797008 3.99521373 3.20963264 
 0.94637018 1.51078597 0.13048172 4.37043882 4.91572869 4.9944569 
 2.32662378 3.00030176 3.67345324 1.86468628 3.27187565 3.75195031 
 4.67731372 4.97360003 4.51354256 4.16388364 2.07021774 0.62298694 
 2.47066525 4.0448314  3.42833033 1.33428236 1.73459005 3.21883822 
 3.74238411 0.70161377 4.10076553 2.71635406 4.25199618 1.71047347 
 0.56080893 1.52101476 3.268613   0.44348611 4.75300544 1.93018848 
 4.02334818 4.2936394  3.86262357 4.65161361 3.18855835 1.7270845 
 3.00572657 4.33593454 0.72605499 2.18322726 0.45156024 1.19257312 
 4.06588255 3.55786778 1.7739621  3.46521942 3.31536931 1.91225274 
 3.03644112 4.40795135 3.41339553 1.58513041 3.04306268 4.17387258 
 3.87451494 1.48655976 0.81035771 2.79862049 1.93737551 2.77350631 
 0.02961769 0.76614433 0.30462889 1.96064363 3.61984286 0.18390026 
 2.12864059 1.24368084 3.12624663 0.87324141 2.49182244 0.73894522 
 3.28624627 0.54325432 1.14019016 0.19408833 0.27587769 2.76731562 
 4.24239649 1.62670982 3.48601752 2.34165538 4.33079158 3.32373447 
 3.95982219 2.59970366 0.55374039 1.53498152 1.84149167 1.7103894 
 0.07011496 2.95268057 3.19355581 3.44488969 4.62279333 3.96444207 
 4.08320222 2.45938531 4.40211166 3.08226174 2.7413684  0.65249907 
 3.88554841 0.54426484 3.1453845  1.11706224 2.71756474 3.54715157 
 0.26560431 1.80808047 0.80432088 3.69086109 0.9538585  2.37378315 
 0.13003854 0.176845   0.15389107 3.69736303 0.64101966 4.06556222 
 1.8471563  2.71152545 0.8789574  1.9509209  3.85439304 3.00958342 
 0.82933679 2.68746095 2.63675    3.31097973 2.64676176 1.24868746 
 3.57576447 2.29019517 4.57538641 1.09943657 0.35057439 0.32663094 
 3.41203514 3.07271481 1.78668303 4.9319088  4.44909133 2.0572905 
 2.7863201  1.82312893 3.50405799 2.06544361 4.22933649 0.59760683 
 3.4048237  0.56655243 3.26653888 2.4388069  3.62042053 0.28400035 
 1.23169562 1.74053627 1.86822616 3.40158413 1.51949388 3.96485747 
 1.91560046 2.03155272 1.07299191 4.44833549 4.49162578 1.68513371 
 3.7402374  2.42013058 1.2598136  4.5316802  1.50186405 1.23001043 
 4.14823233 2.89923774 1.61491513 4.694959   2.62637724 4.15848057 
 2.29320707 0.19248169 2.92876948 0.66504363 2.96658345 2.61105925 
 2.10844525 1.9866828  4.33847884 2.1361846  3.51735449 0.3373951 
 2.51504308 4.52648976 0.11202356 3.20143812 3.46439049 2.42704263 
 4.49869534 2.28628955 4.64371298 1.09661546 2.69151894 3.93238154 
 2.56424271 2.14934753 0.84539193 4.32824987 2.04898603 1.41383722 
 3.0403419  1.51738193 4.93516012 4.49610783 1.63867322 1.40193809 
 0.74791302 2.89177327 2.71531315 1.96517301 1.83925883 4.90219347 
 4.23705842 1.77975734 3.25498864 4.15504765 0.02360116 4.88180179 
 2.65075243 3.26660281 2.65401096 1.13530684]
 `

 ## Histogram 
 We can create a histogram using the data we obtained to visually represent the data set.

To create a histogram, we'll utilize the Python program [Matplotlib](https://matplotlib.org/).
```python
# To draw a histogram:

import numpy as np
import matplotlib.pyplot as plot 

x =  np.random.uniform(0.0, 5.0, 250)

plot.hist(x, 5)
plot.show()
```
Output:
# Standard Deviation
A collection of data's spread or dispersion is measured by standard deviation. It is a means to express how much variance or dispersion there is in a group of data.
You must first determine the mean (average) of a collection of data before you can determine the standard deviation. Afterward, you deduct the mean from each data point and square the outcome (to make it positive). These squared differences are then added, and the result is divided by the total number of data points. To obtain the standard deviation, you take the square root of this result.

You must first determine the mean (average) of a collection of data before you can determine the standard deviation. Afterward, you deduct the mean from each data point and square the outcome (to make it positive). These squared differences are then added, and the result is divided by the total number of data points. To obtain the standard deviation, you take the square root of this result.

The majority of the data are likely close to the mean (average) value if the standard deviation is low.

When the standard deviation is large, the values are more evenly distributed.

Example: This time, we recorded the speeds of seven vehicles:
```
speed = [86,87,88,86,87,85,86]
```
This is the standard deviation:
```
0.9
```
Hence, the majority of the values fall within a 0.9 standard deviation range of the average value, which is 86.4.

Let's try the same thing with a larger range of numbers:
```
speed = [32,111,138,28,59,77,97]
```
This is the standard deviation:
```
37.85
```
Consequently, the majority of values fall between 37.85 and 77.4, which is the mean value.

As you can see, a greater standard deviation denotes a broader range of values.

The standard deviation may be determined using the NumPy module's formula:

Find the standard deviation using the NumPy `std()` method:
```python
import numpy as np 

speed = [86,87,88,86,87,85,86]

std = np.std(speed)

print(std)
```
Output:
`0.9035079029052513`

# Percentiles
A percentile is a measure that indicates the value below which a given percentage of observations in a group of observations fall. For example, the 50th percentile is the value below which 50% of the observations fall. Percentiles are commonly used to understand the distribution of a set of data.

To calculate percentiles, you first need to order the data from smallest to largest. Then, you can use the following formula to find the value at a given percentile:

-$$ Value = (p/100) \times N $$

Where:

- $Value$ is the value at the desired percentile
- $p$ is the percentile (expressed as a decimal)
- $N$ is the total number of values in the data set
For example, consider a data set with the following values: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}. To find the 50th percentile (also known as the median), we first order the values from smallest to largest: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}. The 50th percentile is the value below which 50% of the observations fall. There are 10 values in the data set, so we can use the following formula to find the value at the 50th percentile:

- $$ Value = (50/100) \times 10 = 5 $$

The 50th percentile (or median) is 5. This means that 50% of the values in the data set are less than or equal to 5, and 50% are greater than 5.

Percentiles are often used to understand the distribution of a data set and to compare values within a data set. For example, you might use percentiles to understand how a student's test score compares to the scores of other students in the same class.

Let's use an example where we have a list of all the residents on a street, organized by age.
```
ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
```
What percentage is the 75%? Since the answer is 43, 75% of the population falls inside this age range.

The specified percentile may be found using a method in the NumPy module:

```
# use NumPy `percentile()` method to find the percentiles:

import numpy as np

ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]

percentiles = np.percentile(ages, 75)

print(percentiles)
```
Output: `43.0`

# Data distribution
Earlier in this course, we used very little data in our examples in order to better comprehend the various concepts.

The data sets in the actual world are significantly larger, yet collecting real world data can be challenging, at least in the beginning stages of a project.

## Are Big Data Sets Available?
We utilize the Python package NumPy, which has a number of ways to build random data sets, to generate large data sets for testing.

```python

# Create an array containing 250 random floats between 0  and 5:

import numpy as np

x = numpy.random.uniform(0.0, 5.0, 250)

print(x)
```
Output: `[4.70107872 0.24787675 0.2195555  3.06204253 4.77830208 4.3809458 
 4.57322999 4.36807984 1.67478644 3.68797008 3.99521373 3.20963264 
 0.94637018 1.51078597 0.13048172 4.37043882 4.91572869 4.9944569 
 2.32662378 3.00030176 3.67345324 1.86468628 3.27187565 3.75195031 
 4.67731372 4.97360003 4.51354256 4.16388364 2.07021774 0.62298694 
 2.47066525 4.0448314  3.42833033 1.33428236 1.73459005 3.21883822 
 3.74238411 0.70161377 4.10076553 2.71635406 4.25199618 1.71047347 
 0.56080893 1.52101476 3.268613   0.44348611 4.75300544 1.93018848 
 4.02334818 4.2936394  3.86262357 4.65161361 3.18855835 1.7270845 
 3.00572657 4.33593454 0.72605499 2.18322726 0.45156024 1.19257312 
 4.06588255 3.55786778 1.7739621  3.46521942 3.31536931 1.91225274 
 3.03644112 4.40795135 3.41339553 1.58513041 3.04306268 4.17387258 
 3.87451494 1.48655976 0.81035771 2.79862049 1.93737551 2.77350631 
 0.02961769 0.76614433 0.30462889 1.96064363 3.61984286 0.18390026 
 2.12864059 1.24368084 3.12624663 0.87324141 2.49182244 0.73894522 
 3.28624627 0.54325432 1.14019016 0.19408833 0.27587769 2.76731562 
 4.24239649 1.62670982 3.48601752 2.34165538 4.33079158 3.32373447 
 3.95982219 2.59970366 0.55374039 1.53498152 1.84149167 1.7103894 
 0.07011496 2.95268057 3.19355581 3.44488969 4.62279333 3.96444207 
 4.08320222 2.45938531 4.40211166 3.08226174 2.7413684  0.65249907 
 3.88554841 0.54426484 3.1453845  1.11706224 2.71756474 3.54715157 
 0.26560431 1.80808047 0.80432088 3.69086109 0.9538585  2.37378315 
 0.13003854 0.176845   0.15389107 3.69736303 0.64101966 4.06556222 
 1.8471563  2.71152545 0.8789574  1.9509209  3.85439304 3.00958342 
 0.82933679 2.68746095 2.63675    3.31097973 2.64676176 1.24868746 
 3.57576447 2.29019517 4.57538641 1.09943657 0.35057439 0.32663094 
 3.41203514 3.07271481 1.78668303 4.9319088  4.44909133 2.0572905 
 2.7863201  1.82312893 3.50405799 2.06544361 4.22933649 0.59760683 
 3.4048237  0.56655243 3.26653888 2.4388069  3.62042053 0.28400035 
 1.23169562 1.74053627 1.86822616 3.40158413 1.51949388 3.96485747 
 1.91560046 2.03155272 1.07299191 4.44833549 4.49162578 1.68513371 
 3.7402374  2.42013058 1.2598136  4.5316802  1.50186405 1.23001043 
 4.14823233 2.89923774 1.61491513 4.694959   2.62637724 4.15848057 
 2.29320707 0.19248169 2.92876948 0.66504363 2.96658345 2.61105925 
 2.10844525 1.9866828  4.33847884 2.1361846  3.51735449 0.3373951 
 2.51504308 4.52648976 0.11202356 3.20143812 3.46439049 2.42704263 
 4.49869534 2.28628955 4.64371298 1.09661546 2.69151894 3.93238154 
 2.56424271 2.14934753 0.84539193 4.32824987 2.04898603 1.41383722 
 3.0403419  1.51738193 4.93516012 4.49610783 1.63867322 1.40193809 
 0.74791302 2.89177327 2.71531315 1.96517301 1.83925883 4.90219347 
 4.23705842 1.77975734 3.25498864 4.15504765 0.02360116 4.88180179 
 2.65075243 3.26660281 2.65401096 1.13530684]
 `

 ## Histogram 
 We can create a histogram using the data we obtained to visually represent the data set.

To create a histogram, we'll utilize the Python program [Matplotlib](https://matplotlib.org/).
```python
# To draw a histogram:

import numpy as np
import matplotlib.pyplot as plot 

x =  np.random.uniform(0.0, 5.0, 250)

plot.hist(x, 5)
plot.show()
```
Output:

![image](https://user-images.githubusercontent.com/67673221/209814353-b9d85c31-a559-49eb-9dbc-ae968a83ba94.png)





             
          
