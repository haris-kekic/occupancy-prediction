# -----------------------------------------------------------
# Zur Erstellung eines Multi-Headed Multi-Layer Perceotrons
# in Keras, basierend auf dem Datensatz und Parametern
#
# (C) 2020 Haris Kekic, Fernuniversität in Hagen
# -----------------------------------------------------------


import numpy as np
import pandas as pd
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, LSTM
from keras.layers.merge import concatenate
from keras import losses, metrics, optimizers, regularizers
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import data_prepare as dp
from pandas import DataFrame
import os 
from contextlib import redirect_stdout
from train_callbacks import TestCallback, LearningRateScheduler,lr_schedule


class RNN_LSTM:

    # config (Anzahl der Units pro LSTM im Stack, regularisierung, dropout)
    def __init__(self, config=([32], False, False)): 
        ''' Initialisiert das Objekt mit der Konfiguration für das MLP Netzwerk

            :config: Ein tuple mit folgenden 3 Elementen
                    [0]: Liste der Units für die LSTMs auf einem Stack
                    [1]: Regularisierung verwenden JA/NEIN
                    [2]: Dropout Layer verwenden JA/NEIN
        '''
        self.config = config
        
    def initialize(self, train_set, val_set, test_set=None):
        ''' Initialisiert das Netzwerk mit den Datensätzen
            die für das Trainieren, Validierung und Testen verwendet werden soll

            :train_set: Supervised Trainingsmenge
            :val_set: Supervised Validierungsmenge
            :test_set: Supervised Testmenge

        '''

        self.train_set = train_set
        self.val_set = val_set

        if (test_set is not None):
            self.test_set = test_set

        # Aus den Dimensionen der Trainingsmenge werden auch entsprechend
        # Parameter für das Netzwerk abgeleitet

        # Die Anzahl der Legs, wird für die Anzahl 
        # der Input-Neuronen für die Input Layer verwendet
        self.time_steps = self.train_set.X.shape[1] 

        # Anhand der Anzahl von Featuren werden im Input-Layer
        # sogenannte Input-Header erstellt
        self.feature_count = self.train_set.X.shape[2]

        # Aus den Vorhersageschritten wird die Anzahl der 
        # Output-Neuronen abgeleitet
        self.forcast_steps = self.train_set.y.shape[1]

        if (self.time_steps == 0):
            raise Exception('Anzahl an Zeitschritte muss größer als 0 sein!')

        if (self.feature_count == 0):
            raise Exception('Anzahl an Features muss größer als 0 sein!')

        if (self.forcast_steps == 0):
            raise Exception('Anzahl an Vorhersageschritte muss größer als 0 sein!')
        
        # Bereitet die Input-Mengen für die Eingabe in das Netzwerk vor
        # Aus einem Beispiel (Sample) mit 3 Timesteps und 3 Feature:
        # [[ 10 20 30]
        #  [ 40 50 60]
        #  [ 70 80 90]]
        # werden drei Inputs für die Input-Header generiert, denn
        # die Header stehen für Feature in einem Sample
        # [10 40 70] Header 1
        # [20 50 60] Header 2
        # [30 60 90] Header 3
        # D.h. die Header des MLP bekommen gleichzeitig
        # diese Drei 1-D Vektoren auf einmal, denn
        # sie sind ein Sample. Wobei in einem Header
        # dann entsprechend 3 Input Neuronen für den Zeitschritt
        # bzw. Timestep stehen
        # self.train_set.X = dp.feature_flatten(self.train_set.X)
        # self.val_set.X = dp.feature_flatten(self.val_set.X)
        # if (test_set is not None):
        #     self.test_set.X = dp.feature_flatten(self.test_set.X)

        self.__build(*self.config)

    def __build(self, stacks, regularization=False, dropout=False):
        ''' Baut dynamisch mittels Keras-API das Netzwerk anhand der Parameter zusammen

            :stacks: Eine Liste, die die Units für die LSTM, die gestapelt werden enthält.
                     Je nach Anzahl der Elemente der Liste werden entsprechend LSTM aufeinander gestapalt
            :regularization: Verwende Regularisierung
            :dropout: Verwende Dropout Layer
        '''

        self.model = Sequential()

        kernel_regulizer = regularizers.l2() if regularization else None
        recurrent_regulizer = regularizers.l2() if regularization else None

        for i in range(0, len(stacks)):
            stack_unit = stacks[i]
            return_sequences = False

            if (i < len(stacks) - 1):
                return_sequences = True
      
            if (i == 0):
                self.model.add(LSTM(units=stack_unit, activation='relu', return_sequences=return_sequences, kernel_regularizer=kernel_regulizer, recurrent_regularizer=recurrent_regulizer, input_shape=(self.time_steps, self.feature_count)))
            else:
                self.model.add(LSTM(units=stack_unit, activation='relu', kernel_regularizer=kernel_regulizer, recurrent_regularizer=recurrent_regulizer, return_sequences=return_sequences))

            if (dropout):
                 self.model.add(Dropout(0.5))
        
        self.model.add(Dense(self.forcast_steps, activation='sigmoid'))
        self.model.compile(optimizer=optimizers.RMSprop(lr=0.01), loss=losses.binary_crossentropy, metrics=['accuracy'])

    def train_evaluate(self, epochs=100, batch_size=64):
        ''' Startet den Training und Valiiderungsprozess für das Modell und
            die übergebenen Trainings und Validierungsmengen mit Keras.
            Am Ende wird noch das Model mit der Testmenge evaluiert.

            Achtung: Derzeit wird die Funktion nicht verwendet!

            :epochs: Anzahl der Epochen für das Training
            :batch_size: Die Größe des Batches nach dem der Fehler und Accuracy berechnet werden soll während des Trainings

        '''
        # Trainieren
        train_history = self.train(epochs, batch_size)
        # Evaluiere mit der Testmenge
        test_score = self.model.evaluate(self.test_set.X, self.test_set.y, batch_size=batch_size) 
        return train_history, test_score
    


    def train_validate_evaluate(self, epochs=100, batch_size=64):
        ''' Startet den Training und Valiiderungsprozess für das Modell und
            die übergebenen Trainings und Validierungsmengen mit Keras.

            Am Ende wird noch das Model mit der Testmenge evaluiert.

            Achtung: Derzeit wird die Funktion nicht verwendet!

            :epochs: Anzahl der Epochen für das Training
            :batch_size: Die Größe des Batches nach dem der Fehler und Accuracy berechnet werden soll während des Trainings

        '''
        if (self.model is None):
            raise Exception('Model wurde noch nicht erstellt!')
        
        # Definiere die Keras-Callbacks, die während des Trainigs ausgelöst werden.
        # Test-Callback ist eine Keras-Custom-Callback, der in diesem Fall
        # nach jeder Epoche die Evalueriung auf der Testmenge durchführt
        # KEIN GUTER ANSATZ, aber ich lasse es mal so
        testCallBack = TestCallback(self.test_set.X, self.test_set.y, batch_size)
        # Callback Passt die Trainigsrate an
        learnRateCallback = LearningRateScheduler(lr_schedule, verbose=1) 
        train_history = self.model.fit(self.train_set.X, 
                                    self.train_set.y, 
                                    validation_data=(self.val_set.X, self.val_set.y), 
                                    batch_size=batch_size, 
                                    epochs=epochs, 
                                    callbacks=[testCallBack, learnRateCallback])
        return train_history, testCallBack.history

    def train_validate(self, epochs=100, batch_size=64):
        ''' Startet den Training und Valiiderungsprozess für das Modell und
            die übergebenen Trainings und Validierungsmengen mit Keras.

            :epochs: Anzahl der Epochen für das Training
            :batch_size: Die Größe des Batches nach dem der Fehler und Accuracy berechnet werden soll während des Trainings

        '''
        if (self.model is None):
            raise Exception('Model wurde noch nicht erstellt!')
        
        # Callback Passt die Trainigsrate an
        learnRateCallback = LearningRateScheduler(lr_schedule, verbose=1)

        # Startet das Training und Validierung des Modells mit den übergebenen Mengen
        train_history = self.model.fit(self.train_set.X, 
                                    self.train_set.y, 
                                    validation_data=(self.val_set.X, self.val_set.y), 
                                    batch_size=batch_size, 
                                    epochs=epochs, 
                                    shuffle=False,
                                    callbacks=[learnRateCallback])
        return train_history

    def train(self, epochs=100, batch_size=64):
        ''' Startet das Training für das Modell

            :epochs: Anzahl der Epochen für das Training
            :batch_size: Die Größe des Batches nach dem der Fehler und Accuracy berechnet werden soll während des Trainings

        '''
        if (self.model is None):
            raise Exception('Model wurde noch nicht erstellt!')
        
        # Callback Passt die Trainigsrate an
        learnRateCallback = LearningRateScheduler(lr_schedule, verbose=1)

        # Startet das Training und Validierung des Modells mit den übergebenen Mengen
        train_history = self.model.fit(self.train_set.X, 
                                    self.train_set.y, 
                                    batch_size=batch_size, 
                                    epochs=epochs, 
                                    callbacks=[learnRateCallback])
        return train_history

    def visualize_model(self, path, name):
        ''' Visualisiert das erstellte Modell und speichert die Datei auf dem
            übergebenen Pfad. Zusätzlich wird noch eine txt Datei mit der
            textuellen Architekturbeschreibung abgespeichert.

            :path: Pfad auf welchem die Visualisierung abgespeichert werden soll
            :name: Dateiname für die Visualisierung

        '''

        if (self.model is None):
            raise Exception('Model wurde noch nicht erstellt!')
        
        # Nicht nur die Visualisierung, aber auch die Beschreibung der Architektur
        # wird in einer Text-Datei mit abgespeichert
        print(self.model.summary())
        with open(f'{path}/{name}_summary.txt', 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

        # Speichere die Visualisierung des Modells
        plot_model(self.model, to_file=f'{path}/{name}.png', show_shapes=True, show_layer_names=True)

