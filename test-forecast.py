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
from greend_niom import Niom 
from statsmodels.tsa.ar_model import AR
from math import sqrt
from pandas import concat
import math



values = DataFrame([20, 23, 85, 12, 66, 11, 33, 45, 44, 62])
dataframe = pd.concat([values.shift(1), values], axis=1)
dataframe.columns = ['t', 't+1']

X = dataframe.values
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

predictions = [x for x in test_X]
# skill of persistence model
rmse = np.sqrt(mean_squared_error(test_y, predictions))
print('Test RMSE: %.3f' % rmse)
# calculate residuals
residuals = [test_y[i]-predictions[i] for i in range(len(predictions))]
residuals = DataFrame(residuals)


#pd.plotting.autocorrelation_plot(values)
#plt.show()

niom = Niom('building4', 'B4')

data = niom.run(datetime.date(2014, 2, 3), datetime.date(2014, 2, 3))

meter1 = DataFrame(data[:, 15].astype('float64'))

#plt.plot(meter1, color='blue', linewidth=0.2)

pd_time = pd.to_datetime(data[:, 7])
x_time = DataFrame(pd_time)




dataframes = concat([x_time, meter1], axis = 1)
dataframes.columns = ['time', 'meter']
dataframes.set_index('time')
p_data = dataframes.resample('10min').mean()

p_data.fillna(method='ffill') 
print(p_data)
plt.plot(p_data.values, color='red', linewidth=0.2)
#plt.show()

clean = DataFrame(p_data.values[:,0])
clean.fillna(method='ffill')
X = list(i if not math.isnan(i) else 0 for i in p_data.values[:,0])


train, test = X[0:len(X)-1], X[len(X)-1:]
# train autoregression
model = AR(train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# make predictions
start = len(train)
end = len(train)+len(test)-1
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

for i in range(len(predictions)):
    print('predicted=%f, expected=%f' % (predictions[i], test[i]))

train_last = train[-30:]

rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot results
#plt.plot(test)
#plt.plot(predictions, color='red')
#plt.show()

model = ARIMA(train, order=(20,0,1))
model_fit = model.fit()
output = model_fit.forecast(steps=1)[0]

plt.plot(test)
plt.plot(output, color='red')
plt.show()

#d = output

#print(output)