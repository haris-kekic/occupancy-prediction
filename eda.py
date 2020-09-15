import data_prepare as dp
import datetime
import os 
from contextlib import redirect_stdout
import pandas as pd
from visualizer import Visualizer
from visual_data import VisualData
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np 

class EDA:

    def load_datasets(self):
        ''' Lädt die Haushaltsdatensätze für die in der Masterarbeit vorgegebenen Datumbereiche
        '''
        self.df_b0 = dp.read_dataset_for_daterange('building0', '2014-06-18', '2014-10-12')
        self.df_b1 = dp.read_dataset_for_daterange('building1', '2014-03-13', '2015-06-29')
        self.df_b2 = dp.read_dataset_for_daterange('building2', '2014-02-20', '2015-06-29')
        self.df_b3 = dp.read_dataset_for_daterange('building3', '2014-04-08', '2015-03-13')
        self.df_b4 = dp.read_dataset_for_daterange('building4', '2014-02-01', '2014-10-31')
        self.df_b5 = dp.read_dataset_for_daterange('building5', '2014-03-08', '2015-03-21')
        self.df_b7 = dp.read_dataset_for_daterange('building7', '2014-07-15', '2014-11-02')

        self.__extracted_properties()

    def __extracted_properties(self):
        ''' Extrahiert die statistischen Eigenschaften der geladedenen Haushaltdatensätze.
            Es werden statistische Eigenschaften für den ganzen Tag extrahiert, aber auch
            nur für die Nächte für die Eigenschaften des Hintergrundverbrauchs.
        '''
        
        if (self.df_b1 is None or self.df_b2 is None or self.df_b3 is None or self.df_b4 is None or self.df_b5 is None):
            raise NameError(f'Mindestens ein Datensätz ist nicht initialisiert!')

        night_from_time = '01:00'
        night_to_time = '05:00'

        self.df_b0_night = self.df_b0[(self.df_b0['LocalTime'] >= night_from_time) &  (self.df_b0['LocalTime'] <= night_to_time)]
        self.df_b1_night = self.df_b1[(self.df_b1['LocalTime'] >= night_from_time) &  (self.df_b1['LocalTime'] <= night_to_time)]
        self.df_b2_night = self.df_b2[(self.df_b2['LocalTime'] >= night_from_time) &  (self.df_b2['LocalTime'] <= night_to_time)]
        self.df_b3_night = self.df_b3[(self.df_b3['LocalTime'] >= night_from_time) &  (self.df_b3['LocalTime'] <= night_to_time)]
        self.df_b4_night = self.df_b4[(self.df_b4['LocalTime'] >= night_from_time) &  (self.df_b4['LocalTime'] <= night_to_time)]
        self.df_b5_night = self.df_b5[(self.df_b5['LocalTime'] >= night_from_time) &  (self.df_b5['LocalTime'] <= night_to_time)]
        self.df_b7_night = self.df_b7[(self.df_b7['LocalTime'] >= night_from_time) &  (self.df_b7['LocalTime'] <= night_to_time)]

        self.df_prop_b0 = self.df_b0.describe(include=['float'])
        self.df_prop_b1 = self.df_b1.describe(include=['float'])
        self.df_prop_b2 = self.df_b2.describe(include=['float'])
        self.df_prop_b3 = self.df_b3.describe(include=['float'])
        self.df_prop_b4 = self.df_b4.describe(include=['float'])
        self.df_prop_b5 = self.df_b5.describe(include=['float'])
        self.df_prop_b7 = self.df_b7.describe(include=['float'])

        self.df_prop_b0_night = self.df_b0_night.describe(include=['float'])
        self.df_prop_b1_night = self.df_b1_night.describe(include=['float'])
        self.df_prop_b2_night = self.df_b2_night.describe(include=['float'])
        self.df_prop_b3_night = self.df_b3_night.describe(include=['float'])
        self.df_prop_b4_night = self.df_b4_night.describe(include=['float'])
        self.df_prop_b5_night = self.df_b5_night.describe(include=['float'])
        self.df_prop_b7_night = self.df_b7_night.describe(include=['float'])


    def save(self):

        if (self.df_b1 is None or self.df_b2 is None or self.df_b3 is None or self.df_b4 is None or self.df_b5 is None):
            raise NameError(f'Eigenschaften von mindestens einen Datensatz wurden nicht ext!')
        
        self.df_prop_b0.to_csv('data/building0/eda/properties.csv')
        self.df_prop_b1.to_csv('data/building1/eda/properties.csv')
        self.df_prop_b2.to_csv('data/building2/eda/properties.csv')
        self.df_prop_b3.to_csv('data/building3/eda/properties.csv')
        self.df_prop_b4.to_csv('data/building4/eda/properties.csv')
        self.df_prop_b5.to_csv('data/building5/eda/properties.csv')
        self.df_prop_b7.to_csv('data/building7/eda/properties.csv')

        self.df_prop_b0_night.to_csv('data/building0/eda/properties_night.csv')
        self.df_prop_b1_night.to_csv('data/building1/eda/properties_night.csv')
        self.df_prop_b2_night.to_csv('data/building2/eda/properties_night.csv')
        self.df_prop_b3_night.to_csv('data/building3/eda/properties_night.csv')
        self.df_prop_b4_night.to_csv('data/building4/eda/properties_night.csv')
        self.df_prop_b5_night.to_csv('data/building5/eda/properties_night.csv')
        self.df_prop_b7_night.to_csv('data/building7/eda/properties_night.csv')


eda = EDA()
eda.load_datasets()
eda.save()
