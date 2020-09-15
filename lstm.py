
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


rawData = [10, 20, 30, 40, 50, 60, 70, 80, 90]
X, y = split_sequence(rawData, 3)
print(X)

model = Sequential()
model.add(LSTM(units=32, activation='relu', return_sequences=True, input_shape=(3, 1)))
model.add(LSTM(units=32, activation='relu', return_sequences=True))
model.add(LSTM(units=32, activation='relu', return_sequences=True))
model.add(LSTM(units=32, activation='relu', return_sequences=True))
model.add(LSTM(units=32, activation='relu', return_sequences=True))
model.add(LSTM(units=32, activation='relu', return_sequences=True))
model.add(LSTM(units=32, activation='relu', return_sequences=True))
model.add(LSTM(units=32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()


print(X.shape[0])
X = X.reshape((X.shape[0], X.shape[1], 1))

print(X)

model.fit(X, y, epochs=100, verbose=0)

plot_model(model, to_file='hallo.png')

x_input = np.array([70, 80, 90]).reshape((1, 3, 1))
print(x_input)

yhat = model.predict(x_input)
print("Prediction")
print(yhat)


