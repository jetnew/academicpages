---
title: 'Anomaly Detection of Time Series Data'
date: 2019-06-06
permalink: /posts/2019/06/anomaly-detection-of-time-series-data/
tags:
  - anomaly-detection
  - time-series
  - machine-learning
---

A note on anomaly detection techniques, evaluation and application on time series data.

# Overview

Definition - Anomaly Detection
------
Anomaly detection (also outlier detection) is the identification of rare items, events or observations which raise suspicions by differing significantly from the majority of the data. – Wikipedia

Definition - Anomaly
------
An anomaly is the deviation in a quantity from its expected value, e.g., the difference between a measurement and a mean or a model prediction. – Wikipedia

Statistical Methods
------
* Holt-Winters (Triple Exponential Smoothing)
* ARIMA (Auto-Regressive Integrated Moving Average)
* Histogram-Based Outlier Detection (HBOS)

Conventional statistical methods are generally more interpretable and sometimes more useful than machine learning-based methods, depending on the specified problem.

Machine Learning Methods
------
* Supervised (e.g. Decision Tree, SVM, LSTM Forecasting)
* Unsupervised (e.g. K-Means, Hierarchical Clustering, DBSCAN)
* Self-Supervised (e.g. LSTM Autoencoder)

Machine learning methods can model more complex data and hence able to detect more complex anomalies than conventional statistical methods.

Data Representation
------
* Point
* Rolling Window (or trajectory matrix)
* Time Series Features (transformations, decompositions and statistical measurements)

Other Techniques
------
* Synthetic Anomaly Generation (e.g. GANs)
* Note-Worthy Libraries (tsfresh, fbprophet)

# Statistical Methods

Holt-Winters (Triple Exponential Smoothing)
------
Holt-Winters is a forecasting technique for seasonal (i.e. cyclical) time series data, based on previous timestamps.

Holt-Winters models a time series in 3 ways – average, trend and seasonality. An average is a value referenced upon, a trend is a general increase/decrease over time and a seasonality is a cyclical repeating pattern over a period.

Equation:
ŷ x = α⋅yx + (1−α)⋅ŷ x−1

The value forecast at t=x is a factor of the value at t=x, along with a discounted value of the value forecast at t=x-1. (1-a) is recursively multiplied every timestamp back, resulting in an exponential computation.

```
from statsmodels.tsa.holtwinters import ExponentialSmoothing

fit = ExponentialSmoothing(data, seasonal_periods=periodicity, trend='add', seasonal='add').fit(use_boxcox=True)
fit.fittedvalues.plot(color='blue')
fit.forecast(5).plot(color='green')
plt.show()
```

