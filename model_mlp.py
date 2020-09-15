# -----------------------------------------------------------
# Zur Erstellung eines Multi-Headed Multi-Layer Perceotrons
# in Keras, basierend auf dem Datensatz und Parametern
#
# (C) 2020 Haris Kekic, Fernuniversität in Hagen
# -----------------------------------------------------------


import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers.merge import concatenate
from keras import losses, metrics, optimizers, regularizers
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import data_prepare as dp
from pandas import DataFrame
import os 
from contextlib import redirect_stdout
from train_callbacks import TestCallback, LearningRateScheduler,lr_schedule


class MHMLP:

    # config (Anzahl der Neuronen pro Hiddenschicht, regularisierung, dropout)
    def __init__(self, config=([100], False, False)): 
        ''' Initialisiert das Objekt mit der Konfiguration für das MLP Netzwerk

            :config: Ein tuple mit folgenden 3 Elementen
                    [0]: Liste der Neuronen für die Hidden-Schichten
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
        self.train_set.X = dp.feature_flatten(self.train_set.X)

        if (self.val_set is not None):
            self.val_set.X = dp.feature_flatten(self.val_set.X)

        if (test_set is not None):
            self.test_set.X = dp.feature_flatten(self.test_set.X)

        self.__build(*self.config)

    def __build(self, hidden_layer_neurons, regularization=False, dropout=False):
        ''' Baut dynamisch mittels Keras-API das Netzwerk anhand der Parameter zusammen

            :hidden_layer_neurons: Ein Array, der sowohl die Anzahl der Hidden-Schichten als auch Neuronen pro Schict bestimmt
            :regularization: Verwende Regularisierung
            :dropout: Verwende Dropout Layer
        '''
        input_layers = list()
        last_hidden_layers = list()

        kernel_regulizer = regularizers.l2() if regularization else None
        # Schleife über die Anzahl der Feature um entsprechend 
        # die Input-Header des MLP Netzwerkes zu erstellen
        for _ in range(0, self.feature_count):
            # Erstelle mit Keras-API den Input Layer
            # wobei die Anzahl der Input-Neuronen der Anzahl
            # der Zeitschritte (Legs) entspricht
            input = Input(shape=(self.time_steps,))
            input_layers.append(input)

            # Gehe durch den Array um die entsprechenden Hidden-Layer
            # mit der übergebenen Anzahl an Neuronen zu erstellen.
            # Da die äußere Schleife durch die Features iteriert
            # und die Input-Header für das Netzwerk erstellt,
            # so wird für auf jeden Header dann entsprechend die
            # Hidden-Schichten aufgelegt. Und für jeden "Pfad" im Netzwerk
            # wird die gleiche Anzahl der Hidden-Schichten und Neuronen angelegt
            for i, neurons in enumerate(hidden_layer_neurons):
                if (i == 0):
                    # Falls das die erste Hiddenschicht ist, dann wird sie an das Input rangehängt
                    dense = Dense(neurons, activation='relu', kernel_regularizer=kernel_regulizer)(input) # An Input-Schicht ranhängen
                    # Sollte Dropout verwendet werden, dann füge der Hidenschicht 
                    # auch eine Dropout-Schicht mit Qoefizienten 0.5
                    if (dropout):
                        dense = Dropout(0.5)(dense)
                else:
                    # Falls das nicht die erste Hiddenschicht ist, 
                    # dann wird sie an die letzte Hiddenschicht rangehängt
                    dense = Dense(neurons, activation='relu', kernel_regularizer=kernel_regulizer)(dense) # An die letzte Hidden-Schicht ranhängen

                    # Sollte Dropout verwendet werden, dann füge der Hidenschicht 
                    # auch eine Dropout-Schicht mit Qoefizienten 0.5
                    if (dropout):
                        dense = Dropout(0.5)(dense)
                
                if (i == len(hidden_layer_neurons) - 1):
                    # Wenn wir bei der letzten HIdden-Schicht angelangt sind
                    # dann füge sie der Liste hinzu um später die Pfade
                    # im Netzwerk mergen zu können
                    last_hidden_layers.append(dense)
        
        # Merge die Pfade im Netzwerk, die von den Input-Headern der Feature ausgehen, zusammen
        merge = concatenate(last_hidden_layers) if len(last_hidden_layers) > 1 else last_hidden_layers[0]
        # Füge dann noch den finalen  Output-Layer
        output = Dense(self.forcast_steps, activation='sigmoid')(merge)

        # Erstelle das Keras-Modell und Compiliere mit den Parametern
        self.model = Model(inputs=input_layers, outputs=output)
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

