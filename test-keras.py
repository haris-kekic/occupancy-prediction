
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np


model = Sequential()
model.add(Dense(3, input_dim=4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

X = np.array([[0.5, 0.6, 0.9, 0.4], 
               [0.2, 0.3, 0.4, 0.6],
                [0.5, 0.6, 0.9, 0.4], 
               [0.2, 0.3, 0.4, 0.6],
               [0.5, 0.6, 0.9, 0.4], 
               [0.2, 0.3, 0.4, 0.6],
               [0.5, 0.6, 0.9, 0.4], 
               [0.2, 0.3, 0.4, 0.6],
               [0.5, 0.6, 0.9, 0.4], 
               [0.2, 0.3, 0.4, 0.6],
               [0.5, 0.6, 0.9, 0.4], 
               [0.2, 0.3, 0.4, 0.6],
               [0.5, 0.6, 0.9, 0.4], 
               [0.2, 0.3, 0.4, 0.6],
               [0.5, 0.6, 0.9, 0.4], 
               [0.2, 0.3, 0.4, 0.6],
               [0.5, 0.6, 0.9, 0.4], 
               [0.2, 0.3, 0.4, 0.6],
               [0.5, 0.6, 0.9, 0.4], 
               [0.2, 0.3, 0.4, 0.6],
               [0.5, 0.6, 0.9, 0.4], 
               [0.2, 0.3, 0.4, 0.6],
               [0.5, 0.6, 0.9, 0.4], 
               [0.2, 0.3, 0.4, 0.6],
               [0.5, 0.6, 0.9, 0.4], 
               [0.2, 0.3, 0.4, 0.6],
               [0.5, 0.6, 0.9, 0.4], 
               [0.2, 0.3, 0.4, 0.6],
               [0.5, 0.6, 0.9, 0.4], 
               [0.2, 0.3, 0.4, 0.6],
               [0.5, 0.6, 0.9, 0.4], 
               [0.2, 0.3, 0.4, 0.6]
                ])
y = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9])

model.fit(X, y, epochs=150, batch_size=1)

pred = model.predict(np.array([[0.5, 0.6, 0.9, 0.4], [0.2, 0.3, 0.4, 0.6]]))

print(pred)

