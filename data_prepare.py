# -----------------------------------------------------------
# Enthält Funktionen zur Datenaufbereitung
#
# (C) 2020 Haris Kekic, Fernuniversität in Hagen
# -----------------------------------------------------------


import numpy as np
import glob
from pandas import DataFrame
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def read_dataset(dataset, set='std', phase='phase7'):
    ''' Fasst alle Daten aus den CSV Dateien der übergebenen Phase zusammen in ein Pandas-DataFrame

        :dataset: Datensatz für welchen die Daten geladen werden sollen
        :phase: Die Phase für welchen die Daten geladen werden sollen
    '''

    csv_data_path = f'data/{dataset}/{phase}/{set}/*.csv'
    # Lade alle CSV Dateien mit dem Pfad
    csv_files = glob.glob(csv_data_path)
    df_dataset = DataFrame()
    for file in csv_files:
        # Lese die CSV Dateien ein
        df_file = pd.read_csv(file)
        # Füge sie dem Ergebsniframe zu
        df_dataset = df_dataset.append(df_file)

    return df_dataset

def read_datasets(datasets, set='std', phase='phase7'):
    ''' Fasst alle Daten aus den CSV Dateien der übergebenen Phase zusammen in ein Pandas-DataFrame

        :dataset: Datensatz für welchen die Daten geladen werden sollen
        :phase: Die Phase für welchen die Daten geladen werden sollen
    '''
    df_datasets = DataFrame()
    for dataset in datasets:
        csv_data_path = f'data/{dataset}/{phase}/{set}/*.csv'
        df_dataset = read_dataset(dataset, set, phase)
        df_datasets = df_datasets.append(df_dataset)

    return df_datasets

def read_dataset_fragment(dataset, quota, direction = 'from_begin', set='std', phase='phase7'):
    ''' Fasst alle Daten aus den CSV Dateien der übergebenen Phase zusammen in ein Pandas-DataFrame

        :quota: Prozentsatz an Daten die aus dem Datensatz rausgelesen werden sollen
        :direction: Von welcher Seite der Datensatz rausgelesen werden soll, 'from_begin' - von Anfang, 'from_end" - vom Ende
        :dataset: Datensatz für welchen die Daten geladen werden sollen
        :phase: Die Phase für welchen die Daten geladen werden sollen
    '''

    csv_data_path = f'data/{dataset}/{phase}/{set}/*.csv'
    # Lade alle CSV Dateien mit dem Pfad
    csv_files = glob.glob(csv_data_path)
    df_dataset = DataFrame()
    for file in csv_files:
        # Lese die CSV Dateien ein
        df_file = pd.read_csv(file)
        # Füge sie dem Ergebsniframe zu
        df_dataset = df_dataset.append(df_file)

    row_count = df_dataset.shape[0]
    fragment_count = round(row_count * quota)
    if (direction == 'from_begin'):
        df_dataset = df_dataset.head(fragment_count)
    
    if (direction == 'from_end'):
        df_dataset = df_dataset.tail(fragment_count)

    return df_dataset

def read_datasets_fragment(datasets, quota, direction = 'from_begin', set='std', phase='phase7'):
    ''' Fasst alle Daten aus den CSV Dateien der übergebenen Phase zusammen in ein Pandas-DataFrame

        :quota: Prozentsatz an Daten die aus dem Datensatz rausgelesen werden sollen
        :direction: Von welcher Seite der Datensatz rausgelesen werden soll, 'from_begin' - von Anfang, 'from_end" - vom Ende
        :dataset: Datensatz für welchen die Daten geladen werden sollen
        :phase: Die Phase für welchen die Daten geladen werden sollen
    '''
    df_datasets = DataFrame()
    for dataset in datasets:
        csv_data_path = f'data/{dataset}/{phase}/{set}/*.csv'
        df_dataset = read_dataset_fragment(dataset, quota, direction, set, phase)
        df_datasets = df_datasets.append(df_dataset)

    return df_datasets

def read_dataset_for_daterange(dataset, from_date, to_date, phase='phase2'):
    ''' Fasst alle Daten aus den CSV Dateien der übergebenen Phase zusammen in ein Pandas-DataFrame

        :dataset: Datensatz für welchen die Daten geladen werden sollen
        :phase: Die Phase für welchen die Daten geladen werden sollen
    '''

    csv_data_path = f'data/{dataset}/{phase}/*.csv'
    # Lade alle CSV Dateien mit dem Pfad
    csv_files = glob.glob(csv_data_path)
    df_dataset = DataFrame()
    for file in csv_files:
        # Lese die CSV Dateien ein
        df_file = pd.read_csv(file)
        # Füge sie dem Ergebsniframe zu
        df_dataset = df_dataset.append(df_file)

    return df_dataset[(df_dataset['LocalDate'] >= from_date) & (df_dataset['LocalDate'] <= to_date)]


def split_train_val_test(X, y, shuffle=False):
    ''' Splittet die Menge auf Training, Validieriung und Test nach dem Verhältnis 0.6, 0.2, 0.2

        :X: Die Input-Menge
        :y: Die Output-Menge
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle=False)

    return X_train, X_val, X_test, y_train, y_val, y_test

def split_train_test(X, y, shuffle=False):
    ''' Splittet die Menge auf Training und Test nach dem Verhältnis 0.6, 0.2, 0.2

        :X: Die Input-Menge
        :y: Die Output-Menge
    '''
    return train_test_split(X, y, test_size=0.2, shuffle=shuffle)
    

def trans_to_timeseries_samples(dataset, back_steps, forward_steps=1):
    ''' Erstelle aus dem input und output ensprechende 
        Beispiele (Samples) für Modelle wobei jedem
        Input-Beispiel der entsprechende Output zugewiesen wird

        Das Ergebnis des Inputs ist 3D Vektor:
        (samples, timesteps, features)

         :input: Die Eingabemenge (Feature)
         :output: Die erwartete Ausgabe (Klasse, Prognose)
         :back_steps: Anzahl der Zeitschritte in die die Eingabemenge
                      zusammengefasst werden soll. Für Zeitreihen, werden Features 
                      in Spalten und die Zeitschritte in Zeilen aufgeteilt.
                      Beispiel:
                      2 Sample 4-Zeitschritte 3-Feature und 
                        INPUT                   OUTPUT
                     [ 
                        [ [0.5, 0.2, 0.1]          [1]
                          [0.2, 0.4, 0.1]
                          [0.3, 0.1, 0.3]
                          [0.5, 0.2, 0.3] ]  

                        [ [0.5, 0.2, 0.1]          [0]
                          [0.2, 0.4, 0.1]
                          [0.3, 0.1, 0.3]
                          [0.5, 0.2, 0.3] ] 

                     ]
    '''
    inputs, targets = dataset[:,:-1], dataset[:,-1]

    X, y = list(), list()
    for i in range(len(inputs)):
        # Finde das Ende des Samples
        end_ix = i + back_steps
        out_end_ix = end_ix + forward_steps - 1
        # Überprüfe ob wir an das Ende gekommen sind
        if (end_ix > len(inputs)) or (out_end_ix > len(targets)):
            break
        # Fasse Input und entsprechenden Output zusammen
        seq_x, seq_y = inputs[i:end_ix], targets[end_ix-1:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def min_max_scale(dataset):
    ''' Skalieren der Mengen auf Werte von 0-1. Formel X = (X - min(X)) / (max(X) - min(X))
            Wobei die Errechneten Min und Max der Trainingsmenge,
            auf die Validierungs und Testmenge übertragen wird.
            Somit verhindert man Data-Leakage, bzw. das Informationen außerhalb
            der Trainingsmenge in diese reinfließen. Würden wir jetzt Min und Max
            auf allen drei Mengen zusammen errechnen, so würden dann auch Informationen
            in die Trainingsmenge reinflißen und das Modell würde damit trainiert werden.
            https://machinelearningmastery.com/data-leakage-machine-learning/

            :train: Trainingsmenge
            :test: Testmenge
            :val: Validierungsmenge, die nicht übergeben werden muss, wenn wir nur mit train und test arbeiten
    '''

    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(dataset)


def feature_flatten(dataset):
    ''' Nimmt ein 3 Dimensionales Array (samples, timesteps, feature) entgegen 
        und liefert ein 2 Dimensionales wobei die 'feature' Dimension aufgerollt wird.

        Ein Beispiel für ein Sample:
        # [[ 10 20 30]
        #  [ 40 50 60]
        #  [ 70 80 90]]
        # werden drei Inputs für die Input-Header generiert, denn
        # die Header stehen für Feature in einem Sample
        # [[10 40 70] Header 1
        # [20 50 60] Header 2
        # [30 60 90]] Header 3

    '''
    feature_count = dataset.shape[2]
    feature_sets = list()
    for feat in range(0, feature_count):
        set = dataset[:, :, feat]
        feature_sets.append(set)

    return feature_sets

