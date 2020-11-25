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

ARIMA (Auto-Regressive Integrated Moving Average)
------
ARIMA is a statistical model for time series data, capturing 3 key aspects of the temporal information — Auto-Regression(AR), Integration(I) and Moving Average(MA).

* Auto-Regression — Observations are regressed on its own lagged (i.e., prior) values.
* Integrated — Data values are replaced by the difference between values.
* Moving Average — Regression errors are dependent on lagged observations.

```
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

p = 5  # lag
d = 1  # difference order
q = 0  # size of moving average window

train, test = train_test_split(X, test_size=0.20, shuffle=False)
history = train.tolist()
predictions = []

for t in range(len(test)):
	model = ARIMA(history, order=(p,d,q))
	fit = model.fit(disp=False)
	pred = fit.forecast()[0]
  
	predictions.append(pred)
	history.append(test[t])
  
print('MSE: %.3f' % mean_squared_error(test, predictions))

plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
```

Histogram-Based Outlier Detection
------

Histogram-Based Outlier Score (HBOS) is a O(n) linear time unsupervised algorithm that is faster than multivariate approaches at the cost of less precision. It can detect global outliers well but performs poorly on local outlier problems.

```
from kenchi.outlier_detection.statistical import HBOS

hbos = HBOS(novelty=True).fit(X)
y_pred = hbos.predict(X)
```

# Machine Learning Methods

Decision Tree — Supervised
------
Decision Trees are used for anomaly detection to learn rules from data. However, with few labelled data, aside from the class imbalance problem, inferences from rules may not make sense, as leaf nodes may end up with very few observations, e.g. 2 positives and 0 negatives.

```

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
```

Support Vector Machines (SVM) — Supervised
------
SVMs first maps input vectors into a higher-dimensional feature space, then obtains the optimal separating hyper-plane in the feature space. The decision boundary is determined by support vectors rather than the whole training sample, and thus is extremely robust to outliers.

```

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)

clf = SVC(gamma='auto')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
```

LSTM Forecasting — Supervised
------
LSTM Forecasting is a supervised method that, given a time series sequence as input, predicts the value at the next timestamp. It trains on normal data only, and the prediction error is used as the anomaly score.
```
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

timesteps = window_size-1
n_features = 1

model = Sequential()
model.add(LSTM(16, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
model.add(LSTM(16, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=30, batch_size=32)
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
```

K-Means Clustering — Unsupervised
------
K-Means Clustering is generally not useful in anomaly detection due to its sensitivity to outliers. Centroids cannot be updated if a set of objects close to it is empty.
```
from sklearn.cluster import KMeans

clusters = 3
y_pred = KMeans(n_clusters=clusters).fit_predict(X)

plt.scatter(X[:,0], X[:,1], c=y_pred)
plt.show()
```

Hierarchical Clustering — Unsupervised
------
Hierarchical Clustering, unlike K-Means, does not require specification of the number of clusters at initialisation. It creates a dendrogram and clusters can be separately selected thereafter. Scipy’s Hierarchical Clustering is recommended over Scikit-Learn’s because of its customisability, ability to select number of clusters after the clustering, and dendrogram plots.
```

from sklearn.cluster import AgglomerativeClustering

clusters = 3
y_pred = AgglomerativeClustering(n_clusters=clusters).fit_predict(X)


from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

clusters=5
cls = linkage(X, method='ward')
y_pred = fcluster(cls, t=clusters, criterion='maxclust')

dendrogram(cls)
plt.show()
```

LSTM Autoencoder — Self-Supervised
------
LSTM Autoencoder is a self-supervised method that, given a time series sequence as input, predicts the same input sequence as its output. With this approach, it learns a representation of normal sequences and the prediction error can be interpreted as the anomaly score.
```
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.models import Sequential

class LSTM_Autoencoder:
  def __init__(self, optimizer='adam', loss='mse'):
    self.optimizer = optimizer
    self.loss = loss
    self.n_features = 1
    
  def build_model(self):
    timesteps = self.timesteps
    n_features = self.n_features
    model = Sequential()
    
    # Encoder
    model.add(LSTM(timesteps, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
    model.add(LSTM(16, activation='relu', return_sequences=True))
    model.add(LSTM(1, activation='relu'))
    model.add(RepeatVector(timesteps))
    
    # Decoder
    model.add(LSTM(timesteps, activation='relu', return_sequences=True))
    model.add(LSTM(16, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    
    model.compile(optimizer=self.optimizer, loss=self.loss)
    model.summary()
    self.model = model
    
  def fit(self, X, epochs=3, batch_size=32):
    self.timesteps = X.shape[1]
    self.build_model()
    
    input_X = np.expand_dims(X, axis=2)
    self.model.fit(input_X, input_X, epochs=epochs, batch_size=batch_size)
    
  def predict(self, X):
    input_X = np.expand_dims(X, axis=2)
    output_X = self.model.predict(input_X)
    reconstruction = np.squeeze(output_X)
    return np.linalg.norm(X - reconstruction, axis=-1)
  
  def plot(self, scores, timeseries, threshold=0.95):
    sorted_scores = sorted(scores)
    threshold_score = sorted_scores[round(len(scores) * threshold)]
    
    plt.title("Reconstruction Error")
    plt.plot(scores)
    plt.plot([threshold_score]*len(scores), c='r')
    plt.show()
    
    anomalous = np.where(scores > threshold_score)
    normal = np.where(scores <= threshold_score)
    
    plt.title("Anomalies")
    plt.scatter(normal, timeseries[normal][:,-1], s=3)
    plt.scatter(anomalous, timeseries[anomalous][:,-1], s=5, c='r')
    plt.show()
    
lstm_autoencoder = LSTM_Autoencoder(optimizer='adam', loss='mse')
lstm_autoencoder.fit(normal_timeseries, epochs=3, batch_size=32)
scores = lstm_autoencoder.predict(test_timeseries)
lstm_autoencoder.plot(scores, test_timeseries, threshold=0.95)
```

Data Representation
------
* Point
  * Given a only a point, contextual anomalies cannot be detected due to the lack of temporal information.
* Rolling Window
  * A rolling window (representing a point) contains temporal information from a few time steps back, allowing the possibility of detecting contextual anomalies. This is sufficient for LSTM-based models.
  ```
  from skimage.util import view_as_windows

  window_size = 5
  timeseries = np.array([1,2,3,4,5,6,7,8,9,10])

  trajectory_matrix = view_as_windows(timeseries, window_shape=window_size)
  ```
* Time Series Features
  * Signal transformations/decompositions such as Fast Fourier Transform (FFT), Continuous Wavelet Transform (CWT) and Singular Spectrum Analysis (SSA), as well as statistical measurements such as Max/Min, Mean, No. of Peaks, can surface temporal information crucial to detecting anomalies.

Other Techniques
------
Synthetic Anomaly Generation
* Importance
  * Synthetic anomaly generation is important due to the usual case of lack of labelled anomalous data. Without a good set of known anomalies (in variety and volume), evaluation of anomaly detection models cannot be reliable.
* Rule-based Generator
  * Based on domain expertise, rule-based generation of anomalous data can cover the scope/variety of most possible types of anomalies.
* Generative Adversarial Network
  * Given few data, GANs can generate anomalous data, albeit difficult, due to mode collapse, training instability and noisy generated signals.
  
Note-Worthy Libraries
------
* tsfresh - Automatic Time Series Feature Extraction ([Open-source](https://github.com/blue-yonder/tsfresh))
* fbprophet - Facebook’s Time Series Forecasting Model ([Open-source](https://facebook.github.io/prophet/))

# References
1. [Exponential smoothing — StatsModels](https://www.statsmodels.org/dev/examples/notebooks/generated/exponential_smoothing.html)
2. [How to Create an ARIMA Model for Time Series Forecasting in Python](https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/)
3. [Histogram-based Outlier Score (HBOS): A fast Unsupervised Anomaly Detection Algorithm](https://www.researchgate.net/publication/231614824_Histogram-based_Outlier_Score_HBOS_A_fast_Unsupervised_Anomaly_Detection_Algorithm)
4. [How to Develop LSTM Models for Time Series Forecasting](https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/)
5. [LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection](https://arxiv.org/abs/1607.00148)
