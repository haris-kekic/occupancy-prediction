# multivariate data preparation
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import glob
from pandas import DataFrame
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.merge import concatenate
from keras import losses
from keras import metrics
from keras import optimizers
import matplotlib.pyplot as plt

# Answer to your question: Do Normalization after splitting into train and test/validation. The reason is to avoid any data leakage.

# Data Leakage:

# Data leakage is when information from outside the training dataset is used to create the model. This additional information can allow the model to learn or know something that it otherwise would not know and in turn invalidate the estimated performance of the mode being constructed.

# You can read about it here : https://machinelearningmastery.com/data-leakage-machine-learning/

# split a multivariate sequence into samples
def split_sequences(input, output, n_steps):
    X, y = list(), list()
    for i in range(len(input)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(input):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = input[i:end_ix], output[end_ix - 1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def min_max_scale(train, val, test):

    for feature in range(1, (train.shape[2] + 1)):
        train_min = np.min(train[:,:,:feature]) 
        train_max = np.max(train[:,:,:feature]) 
        train[:,:,:feature] = (train[:,:,:feature] - train_min) / (train_max - train_min) 
        # Für das Scaling von Validation und Test werden 
        # die errechneten Min/Max Parameter der Trainingsmenge angewandt
        val[:,:,:feature] = (val[:,:,:feature] - train_min) / (train_max - train_min) 
        test[:,:,:feature] = (test[:,:,:feature] - train_min) / (train_max - train_min) 


csv_files = glob.glob('data/building4/phase7/ext/*.csv')
df_dataset = DataFrame()
for file in csv_files:
    df_file = pd.read_csv(file)
    df_dataset = df_dataset.append(df_file)

input_dataset = df_dataset[['MinuteTimeStamp', 'Total']].values
output_dataset = df_dataset[['OCCUPIED']].values

time_steps = 96

X, y = split_sequences(input_dataset, output_dataset, time_steps)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle=False)


min_max_scale(X_train, X_val, X_test)


visible1 = Input(shape=(time_steps,))
dense1 = Dense(100, activation='relu')(visible1)
dense1_1 = Dense(50, activation='relu')(dense1)
visible2 = Input(shape=(time_steps,))
dense2 = Dense(100, activation='relu')(visible2)
dense2_1 = Dense(50, activation='relu')(dense2)

# merge input models
merge = concatenate([dense1_1, dense2_1])
output = Dense(1, activation='sigmoid')(merge)
model = Model(inputs=[visible1, visible2], outputs=output)
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])


X_train_feat1 = X_train[:, :, 0]
X_train_feat2 = X_train[:, :, 1]

X_val_feat1 = X_val[:, :, 0]
X_val_feat2 = X_val[:, :, 1]

X_test_feat1 = X_test[:, :, 0]
X_test_feat2 = X_test[:, :, 1]

d = [X_train_feat1, X_train_feat2]

history = model.fit([X_train_feat1, X_train_feat2], y_train, validation_data=([X_val_feat1, X_val_feat2] , y_val), epochs=100)
test_score = model.evaluate([X_test_feat1, X_test_feat2], y_test, batch_size=96)


print('test loss, test acc:', test_score)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)

i_val_loss = val_loss.index(np.min(val_loss))
i_train_loss = loss.index(np.min(loss))

i_val_acc= val_acc.index(np.max(val_acc))
i_train_acc = acc.index(np.max(acc))

min_val_loss_epoch = epochs[i_val_loss]
max_val_acc_epoch = epochs[i_val_acc]

min_train_loss_epoch = epochs[i_train_loss]
max_train_acc_epoch = epochs[i_train_acc]

print('Train Loss Epoch: ', min_train_loss_epoch)
print('Train Accurancy Epoch: ', max_train_acc_epoch)

print('Validation Loss Epoch: ', min_val_loss_epoch)
print('Validation Accurancy Epoch: ', max_val_acc_epoch)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'b', label='Training loss', color='blue')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss', color='red')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


