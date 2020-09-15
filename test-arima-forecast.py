import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pyodbc
from decimal import Decimal
import datetime 
import pandas as pd
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from greend_niom import DataSet 
from statsmodels.tsa.ar_model import AR
from math import sqrt
from pandas import concat
import math
from sklearn.impute import SimpleImputer



dataset = DataSet('building4', 'B4')

data = dataset.fetch_aggregated(datetime.datetime(2014, 2, 3, 10, 0, 0), datetime.datetime(2014, 2, 3, 18, 0, 0))

data_before_gap = dataset.fetch_aggregated(datetime.datetime(2014, 2, 3, 10, 0, 0), datetime.datetime(2014, 2, 3, 14, 7, 0))

data_after_gap = dataset.fetch_aggregated(datetime.datetime(2014, 2, 3, 14, 24, 0), datetime.datetime(2014, 2, 3, 18, 0, 0))

meter1_before_gap = data_before_gap[:, 11].astype('float64')

meter1_after_gap = data_after_gap[:, 11].astype('float64')

mater1_gap = np.full(18, None)

meter1_fill = np.concatenate((meter1_before_gap, mater1_gap, meter1_after_gap), axis=0)

imputer = SimpleImputer ()
transformed_X = imputer.fit_transform([meter1_fill])

print(meter1_fill)
print(transformed_X)

meter1 = data[:, 11].astype('float64')

#missing_time_range = pd.date_range("14:07", "14:23", freq="1min").time


plt.plot(meter1, color='blue')
plt.show()

train, test = list(meter1[0:len(meter1)]), list(meter1[len(meter1)-18:])


predictions = list()

model = AR(train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)

for t in range(len(test)):
    model = ARIMA(train, order=(10,1,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    train.append(yhat)
    print('predicted=%f, expected=%f' % (yhat, obs))

# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()