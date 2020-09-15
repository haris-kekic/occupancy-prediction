# -----------------------------------------------------------
# Klasse, die es ermöglicht Modelle mit einem Datensatz zu
# tainieren und evaluieren. Dabei wird Forward-Chaining
# k-Fold Cross Validation für Zeitreihen verwendet
#
# (C) 2020 Haris Kekic, Fernuniversität in Hagen
# -----------------------------------------------------------

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
import matplotlib.pyplot as plt
import data_prepare as dp
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from pandas import DataFrame
from pandas import Series
from artifacts import TimeSeries, smooth_line

class ModelValidator:
    
    def __init__(self, description, model, df_dataset, df_eval = None):
        ''' Konstruktur um das Objekt zu initialisieren

            :description: Einfache Beschreibung des Vorgangs
            :model: Das Model, das trainiert und validiert werden soll
            :df_dataset: Datensatz der zum Trainieren und Validieren des Modells herangezogen werden soll

        '''
        self.model = model
        self.description = description
        self.df_dataset = dp.min_max_scale(df_dataset)
        self.df_scores = DataFrame()
        self.df_agg_scores = DataFrame()

        if (df_eval is not None):
            self.df_dataset_eval = dp.min_max_scale(df_eval)


    def validate(self, time_steps=30, forcast_steps=5, epochs=100, splits=5): #WHen splits = 1, dann ist es holdout
        ''' Startet den Trainings und Validierungsprozess des Modells mit übergebenen Parametern

            :time_steps: Anzahl der Schritte bzw. Legs die für die Vorhersage genommen werden sollen.
                         Wenn z.B. 20 d.h. das 20 Einträge genommen werden um Werte danach vorherzusagen.
            :forcast_steps: Anzahl der Schrite in Zukunft, die vorhergesagt werden sollen
            :epochs: Anzahl der Epochen die für das Training verwendet werden
            :splits: Anzahl der k-Fold Splits für den Datensatz
        '''

        if self.model is None:
            raise Exception('Kein Modell zum Validieren definiert!')
        
        # Der Datensatz wird entsprechend in ein X und y Mengen geteilt und entsprechend den Vorgaben (Zeitschritte/Leg) 
        # werden sie in Mengen die für das Netzwerk und supervised learning gedacht sind, umgewandelt
        timeseries = TimeSeries(*(dp.trans_to_timeseries_samples(self.df_dataset, time_steps, forcast_steps))) # Tuple (X, y) auspacken

        # Die supervised Menge wird hier durch in die Trainings und Test bzw. Validierungsmenge gesplittet
        # Was hier interessant ist, dass TimeSeriesSplit eigentlich mit einer kleineren Trainigsmenge
        # startet und diese nach jeder Iteration immer wieder erweitert bis dann am Ende die übergebene
        # Anzahl an Splits erreicht wurde z.B. 4
        kfold = TimeSeriesSplit(n_splits=splits)
        fold = 1 # Zähler
        # Nested Cross Validation für Zeitreihen mit forward chaining
        for train_index, val_index in kfold.split(timeseries.X, timeseries.y):
    
            print(train_index)
            print(val_index)
            # Teile die Mengen auf Training und Validierungsmenge
            X_train, y_train = timeseries.X[train_index], timeseries.y[train_index]
            X_val, y_val = timeseries.X[val_index], timeseries.y[val_index]

            # Initialisiere und baue auf das Model anhand der Daten
            self.model.initialize(TimeSeries(X_train, y_train), TimeSeries(X_val, y_val))

            # Starte das Training und Validierung der Mengen
            train_history = self.model.train_validate(epochs=epochs)

            # Die Ergebnisse werden nun in ein pandas Frame umgewandelt um
            # eine leichtere Handhabung zu haben
            self.__train_validate_scores_to_dataframe(train_history, fold)

            # erhöhe den Zähler
            fold = fold + 1

        return self.df_scores, self.df_agg_scores

    def evaluate(self, time_steps=30, forcast_steps=5, epochs=100, splits=5): #WHen splits = 1, dann ist es holdout
        ''' Startet den Trainings und Validierungsprozess des Modells mit übergebenen Parametern

            :time_steps: Anzahl der Schritte bzw. Legs die für die Vorhersage genommen werden sollen.
                         Wenn z.B. 20 d.h. das 20 Einträge genommen werden um Werte danach vorherzusagen.
            :forcast_steps: Anzahl der Schrite in Zukunft, die vorhergesagt werden sollen
            :epochs: Anzahl der Epochen die für das Training verwendet werden
            :splits: Anzahl der k-Fold Splits für den Datensatz
        '''

        if self.model is None:
            raise Exception('Kein Modell zum Validieren definiert!')
        
        # Der Datensatz wird entsprechend in ein X und y Mengen geteilt und entsprechend den Vorgaben (Zeitschritte/Leg) 
        # werden sie in Mengen die für das Netzwerk und supervised learning gedacht sind, umgewandelt
        timeseries = TimeSeries(*(dp.trans_to_timeseries_samples(self.df_dataset, time_steps, forcast_steps))) # Tuple (X, y) auspacken
        timeseries_test = TimeSeries(*(dp.trans_to_timeseries_samples(self.df_dataset_eval, time_steps, forcast_steps))) # Tuple (X, y) auspacken

        
        X_train, y_train = timeseries.X, timeseries.y
        X_test, y_test = timeseries_test.X, timeseries_test.y

        # Initialisiere und baue auf das Model anhand der Daten
        self.model.initialize(TimeSeries(X_train, y_train), None, TimeSeries(X_test, y_test))

        # Starte das Training und Validierung der Mengen
        train_history, test_score = self.model.train_evaluate(epochs=epochs)

        return test_score

    def __train_validate_scores_to_dataframe(self, train_history, fold):
        ''' Wandelt die Ergebnisse einer Validierung aus einem 
            k-Fold Durchgang in ein entsprechendes pandas DataFrame 
            zur leichteren Handhabung

            :train_history: Trainings und Validierungshistorie
            :fold: Nummer des aktuellen Durchlaufs

        '''

        epoch_train_loss = np.array(train_history.history['loss'])
        epoch_train_acc = np.array(train_history.history['accuracy'])
        epoch_val_loss = np.array(train_history.history['val_loss'])
        epoch_val_acc = np.array(train_history.history['val_accuracy'])
        epoch_smooth_val_loss = smooth_line(epoch_val_loss) # Entferne die ersten 5 Punkte
        epoch_smooth_val_acc = smooth_line(epoch_val_acc) # Entferne die ersten 5 Punkte
        
        # Erstellt zunächst eine Zeile für den aktuellen Durchlauf
        df_fold_row = DataFrame({ 'k_Fold': fold, 
                                    'Epoch': train_history.epoch, 
                                    'EpochTrainLoss': epoch_train_loss, 
                                    'EpochTrainAcc': epoch_train_acc, 
                                    'EpochValLoss': epoch_val_loss, 
                                    'EpochSmoothValLoss': epoch_smooth_val_loss, 
                                    'EpochValAcc': epoch_val_acc, 
                                    'EpochSmoothValAcc': epoch_smooth_val_acc
                                })

        # fügt die Zeile in den endültigen Scores Frame, der auch unter 
        # den Namen score_full.csv in der Results Datei abgespeichert wird
        self.df_scores = self.df_scores.append(df_fold_row, ignore_index=True)

        # Erstellt für Folds aggregierte Datei. D.h. es wird ein Mittelwert für jede Epoche
        # aus jedem einzelnen Folds berechnet und aggregiert
        self.df_agg_scores = self.df_scores[self.df_scores.columns.difference(['k_Fold'])].groupby(['Epoch']).mean()

    def __train_test_scores_to_dataframe(self, train_history, test_history, test_score, fold=1):
        ''' Die Methode hat die gleiche Funktion wie die obere. Der Unterschied ist die 
            Handhabung einer zusätzlichen evaluierten Testmenge.
            Diese Methode wird derzeit nicht verwendet, aber soll für Zukunft hier bleiben
            falls wieder Training-Validierung-Test Ansatz verwendet werden soll.

        '''
        epoch_train_loss = np.array(train_history.history['loss'])
        epoch_train_acc = np.array(train_history.history['accuracy'])
        epoch_val_loss = np.array(train_history.history['val_loss'])
        epoch_val_acc = np.array(train_history.history['val_accuracy'])
        epoch_smooth_val_loss = smooth_line(epoch_val_loss)
        epoch_smooth_val_acc = smooth_line(epoch_val_acc)
        epoch_test_loss = test_history['test_loss']
        epoch_test_acc = test_history['test_accuracy']
        epoch_smooth_test_loss = smooth_line(epoch_test_loss)
        epoch_smooth_test_acc = smooth_line(epoch_test_acc)
        
        final_test_loss = np.array(test_score[0]) # broadcast
        final_test_acc = np.array(test_score[1]) # broadcast
            
        df_fold_row = DataFrame({ 'k_Fold': fold, 
                                    'Epoch': train_history.epoch, 
                                    'EpochTrainLoss': epoch_train_loss, 
                                    'EpochTrainAcc': epoch_train_acc, 
                                    'EpochValLoss': epoch_val_loss, 
                                    'EpochSmoothValLoss': epoch_smooth_val_loss, 
                                    'EpochValAcc': epoch_val_acc, 
                                    'EpochSmoothValAcc': epoch_smooth_val_acc, 
                                    'EpochTestLoss': epoch_test_loss , 
                                    'EpochTestAcc': epoch_test_acc, 
                                    'EpochSmoothTestLoss': epoch_smooth_test_loss , 
                                    'EpochSmoothTestAcc': epoch_smooth_test_acc, 
                                    'FinalTestLoss': final_test_loss , 
                                    'FinalTestAcc': final_test_acc})
        self.df_scores = self.df_scores.append(df_fold_row, ignore_index=True)
        self.df_agg_scores = self.df_scores.groupby(['Epoch']).mean() #.apply(func_weighted_mean) # Aggregated scores


