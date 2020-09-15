# evaluate a persistence model
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.ar_model import AR

# load dataset
series = [2, 5, 6, 8, 10, 9, 8, 11, 14, 6, 3, 5, 8, 12, 13, 9, 5, 6, 8, 10, 9, 8, 11, 14, 6, 3, 5, 8, 12, 13, 9]
# create lagged dataset
values = DataFrame([2, 5, 6, 8, 10, 9, 8, 11, 14, 6])
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t', 't+1']
# split into train and test sets
X = series

train, test = X[1:len(X)-5], X[len(X)-5:]
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

rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
