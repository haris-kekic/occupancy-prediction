import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.merge import concatenate
from keras import losses
from keras import metrics
from keras import optimizers
from keras.utils.vis_utils import plot_model
from keras import backend
import matplotlib.pyplot as plt
import data_prepare as dp
import mlp 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import tensorflow as tf


# FORWARD CHAINING !!!!!!! CROSS VALIDATION

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]

# root mean squared error or rmse
def measure_loss(truth, predicted):
    return log_loss(truth, predicted)

def measure_binary_accuracy(truth, predicted):
    # Dies Funktion bildet die Binary Accuracy vom Keras
    # Sie berechnet die Anzahl der übereinstimmenden Elemente in den beiden Arrays und teilt durch die Gesamtanzahl
    # count(groun_truth == predicted) / total
    predicted = np.array(predicted).round() # Alle Werte y_pred >= 0.5 werden als 1 sonst 0 ausgewertet
    return accuracy_score(truth, predicted)

# fit a model
def model_fit(X_train, y_train, config):
    # unpack config
    n_input, n_nodes, n_epochs, n_batch = config


    train_feature_dataset = mlp.feature_flatten(X_train)
    features = X_train.shape[2]
    outputs = y_train.shape[1]
    # define model
    model = mlp.build_model(n_input, features, output_neurons=outputs, hidden_layers=[100])
    model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=['accuracy'])
    # fit
    model.fit(train_feature_dataset, y_train, batch_size=96, epochs=1)
    return model

# forecast with a pre-fit model
def model_predict(model, history, config):
    # unpack config
    n_input, _, _, _ = config
    # prepare data
    x_input = np.array(history[-1::])
    feature_dataset = mlp.feature_flatten(x_input)
    # forecast
    y_pred = model.predict(feature_dataset, verbose=0)
    return y_pred[0]


# walk-forward validation for univariate data
def walk_forward_validation(input_data, target_data, n_test, cfg):
    predictions = []

    X, y = dp.transform_samples(input_data, target_data, cfg[0])

    # split dataset
    X_train, X_test = train_test_split(X, n_test)
    y_train, y_test = train_test_split(y, n_test)

    X_train, X_test = dp.min_max_scale(X_train, X_test)

    # fit model
    model = model_fit(X_train, y_train, cfg)

    history = [x for x in X_train]

    for i in range(len(X_test)):

        y_pred = model_predict(model, history, cfg)

        predictions.append(y_pred)

        history.append(X_test[i])

    loss = measure_loss(y_test, predictions)
    acc = measure_binary_accuracy(y_test, predictions)
    print(f'Loss: {loss} Accuracy: {acc}')
    return [loss, acc]

# repeat evaluation of a config
def repeat_evaluate(input_data, target_data, config, n_test, n_repeats=30):
    # fit and evaluate the model n times
    scores = [walk_forward_validation(input_data, target_data, n_test, config) for _ in range(n_repeats)]
    return scores
# summarize model performance

def summarize_scores(scores):
    # print a summary
    scores_m, score_std = np.mean(scores[:0]), np.std(scores[:1])
    print('%s: %.3f RMSE (+/- %.3f)' % (scores_m, score_std))
    # box and whisker plot
    plt.boxplot(scores)
    plt.show()

df_dataset = dp.read_dataset('building4')
input_data = df_dataset[['MinuteTimeStamp', 'Total']].values
target_dat = df_dataset[['OCCUPIED']].values

time_steps = 96

# data split
n_test = 19
# define config
config = [time_steps, 500, 100, 100]
# grid search
scores = repeat_evaluate(input_data, target_dat, config, n_test)
# summarize scores
summarize_scores(scores)