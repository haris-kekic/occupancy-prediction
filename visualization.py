# -----------------------------------------------------------
# Sktript zur Datentranvisualisierung
#
# (C) 2020 Haris Kekic, Fernuniversität in Hagen
# -----------------------------------------------------------

import datetime 
import pandas as pd
from visualizer import Visualizer
from visual_data import VisualData
from eda import EDA



def visualize_dataset_metrics(dataset, start_date, end_date):
    ''' Visualsiert die Metriken für alle Phasen für die das möglich ist

        :start_date: Anfangsdatum, von welchem ausgehend die Visualsierungen durchgeführt werden
        :end_date: Enddatum, bis zu welchem die Visualsierungen durchgeführt werden
    '''
    visualizer = Visualizer(dataset)
    visualizer.visualize_dataset_metrics(start_date, end_date, 'phase6', 'std', sample_size_minutes=1)
    visualizer.visualize_dataset_metrics(start_date, end_date, 'phase7', 'std',sample_size_minutes=15)
    visualizer.visualize_dataset_metrics(start_date, end_date, 'phase6', 'ext',sample_size_minutes=1)
    visualizer.visualize_dataset_metrics(start_date, end_date, 'phase7', 'ext',sample_size_minutes=15)

def visualize_dataset_raw(dataset, start_date, end_date):
    ''' Visualsiert den Stromverbrauch aus Rohdaten

        :start_date: Anfangsdatum, von welchem ausgehend die Visualsierungen durchgeführt werden
        :end_date: Enddatum, bis zu welchem die Visualsierungen durchgeführt werden
    '''
    visualizer = Visualizer(dataset)
    visualizer.visualize_dataset_raw(start_date, end_date)

def visualize_frequencies(dataset, dataset_name):

    result = VisualData(dataset_name, 'Stromverbrauch_Bereich', None, None, dataset)

    visualizer = Visualizer(dataset_name)
    visualizer.visualize_frequencies(result, 'eda')

def visualize_properties(dataset, dataset_name):
    
    result = VisualData(dataset_name, 'Eigenschaften', None, None, dataset)

    visualizer = Visualizer(dataset_name)
    visualizer.visualize_properties(result, 'eda')




# Starte die Visualiserung der Metriken für alle Phasen für die das möglich ist
# visualize_dataset_metrics('building4', datetime.datetime(2014, 2, 8), datetime.datetime(2014, 2, 8))

# visualize_dataset_raw('building0', datetime.datetime(2014, 6, 18), datetime.datetime(2014, 6, 18))

# VISUALISIERE DIE LÜCKEN

# dataframe = pd.read_csv('data/building1/phase4/2014-04-04.csv', index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
# result = VisualData('building1', 'intr', datetime.datetime.now(), datetime.datetime.now(), dataframe, timestamp='2014-04-04')
# visualizer = Visualizer('building1')
# visualizer.visualize_gap_inerpolation(result, folder='gap_fills')

# dataframe = pd.read_csv('data/building2/phase4/2014-03-13.csv', index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
# result = VisualData('building2', 'intr', datetime.datetime.now(), datetime.datetime.now(), dataframe, timestamp='2014-03-13')
# visualizer = Visualizer('building2')
# visualizer.visualize_gap_inerpolation(result, folder='gap_fills')

# dataframe = pd.read_csv('data/building2/phase4/2014-04-09.csv', index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
# result = VisualData('building2', 'intr', datetime.datetime.now(), datetime.datetime.now(), dataframe, timestamp='2014-04-09')
# visualizer = Visualizer('building2')
# visualizer.visualize_gap_inerpolation(result, folder='gap_fills')

# dataframe = pd.read_csv('data/building2/phase4/2014-04-13.csv', index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
# result = VisualData('building2', 'intr', datetime.datetime.now(), datetime.datetime.now(), dataframe, timestamp='2014-04-13')
# visualizer = Visualizer('building2')
# visualizer.visualize_gap_inerpolation(result, folder='gap_fills')

# dataframe = pd.read_csv('data/building2/phase4/2014-04-25.csv', index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
# result = VisualData('building2', 'intr', datetime.datetime.now(), datetime.datetime.now(), dataframe, timestamp='2014-04-25')
# visualizer = Visualizer('building2')
# visualizer.visualize_gap_inerpolation(result, folder='gap_fills')

# dataframe = pd.read_csv('data/building2/phase4/2014-05-11.csv', index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
# result = VisualData('building2', 'intr', datetime.datetime.now(), datetime.datetime.now(), dataframe, timestamp='2014-05-11')
# visualizer = Visualizer('building2')
# visualizer.visualize_gap_inerpolation(result, folder='gap_fills')

# dataframe = pd.read_csv('data/building2/phase4/2014-06-06.csv', index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
# result = VisualData('building2', 'intr', datetime.datetime.now(), datetime.datetime.now(), dataframe, timestamp='2014-06-06')
# visualizer = Visualizer('building2')
# visualizer.visualize_gap_inerpolation(result, folder='gap_fills')


# dataframe = pd.read_csv('data/building2/phase4/2014-06-12.csv', index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
# result = VisualData('building2', 'intr', datetime.datetime.now(), datetime.datetime.now(), dataframe, timestamp='2014-06-12')
# visualizer = Visualizer('building2')
# visualizer.visualize_gap_inerpolation(result, folder='gap_fills')

# dataframe = pd.read_csv('data/building2/phase4/2014-07-05.csv', index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
# result = VisualData('building2', 'intr', datetime.datetime.now(), datetime.datetime.now(), dataframe, timestamp='2014-07-05')
# visualizer = Visualizer('building2')
# visualizer.visualize_gap_inerpolation(result, folder='gap_fills')

# dataframe = pd.read_csv('data/building2/phase4/2014-09-04.csv', index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
# result = VisualData('building2', 'intr', datetime.datetime.now(), datetime.datetime.now(), dataframe, timestamp='2014-09-04')
# visualizer = Visualizer('building2')
# visualizer.visualize_gap_inerpolation(result, folder='gap_fills')

# dataframe = pd.read_csv('data/building2/phase4/2014-10-20.csv', index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
# result = VisualData('building2', 'intr', datetime.datetime.now(), datetime.datetime.now(), dataframe, timestamp='2014-10-20')
# visualizer = Visualizer('building2')
# visualizer.visualize_gap_inerpolation(result, folder='gap_fills')

# dataframe = pd.read_csv('data/building2/phase4/2014-10-22.csv', index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
# result = VisualData('building2', 'intr', datetime.datetime.now(), datetime.datetime.now(), dataframe, timestamp='2014-10-22')
# visualizer = Visualizer('building2')
# visualizer.visualize_gap_inerpolation(result, folder='gap_fills')

# dataframe = pd.read_csv('data/building2/phase4/2014-11-08.csv', index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
# result = VisualData('building2', 'intr', datetime.datetime.now(), datetime.datetime.now(), dataframe, timestamp='2014-11-08')
# visualizer = Visualizer('building2')
# visualizer.visualize_gap_inerpolation(result, folder='gap_fills')

# dataframe = pd.read_csv('data/building2/phase4/2014-11-29.csv', index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
# result = VisualData('building2', 'intr', datetime.datetime.now(), datetime.datetime.now(), dataframe, timestamp='2014-11-29')
# visualizer = Visualizer('building2')
# visualizer.visualize_gap_inerpolation(result, folder='gap_fills')

# dataframe = pd.read_csv('data/building2/phase4/2014-12-16.csv', index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
# result = VisualData('building2', 'intr', datetime.datetime.now(), datetime.datetime.now(), dataframe, timestamp='2014-12-16')
# visualizer = Visualizer('building2')
# visualizer.visualize_gap_inerpolation(result, folder='gap_fills')

# dataframe = pd.read_csv('data/building2/phase4/2015-01-04.csv', index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
# result = VisualData('building2', 'intr', datetime.datetime.now(), datetime.datetime.now(), dataframe, timestamp='2015-01-04')
# visualizer = Visualizer('building2')
# visualizer.visualize_gap_inerpolation(result, folder='gap_fills')

# dataframe = pd.read_csv('data/building2/phase4/2015-03-10.csv', index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
# result = VisualData('building2', 'intr', datetime.datetime.now(), datetime.datetime.now(), dataframe, timestamp='2015-03-10')
# visualizer = Visualizer('building2')
# visualizer.visualize_gap_inerpolation(result, folder='gap_fills')

# dataframe = pd.read_csv('data/building2/phase4/2015-03-28.csv', index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
# result = VisualData('building2', 'intr', datetime.datetime.now(), datetime.datetime.now(), dataframe, timestamp='2015-03-28')
# visualizer = Visualizer('building2')
# visualizer.visualize_gap_inerpolation(result, folder='gap_fills')

# dataframe = pd.read_csv('data/building2/phase4/2015-05-15.csv', index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
# result = VisualData('building2', 'intr', datetime.datetime.now(), datetime.datetime.now(), dataframe, timestamp='2015-05-15')
# visualizer = Visualizer('building2')
# visualizer.visualize_gap_inerpolation(result, folder='gap_fills')


# dataframe = pd.read_csv('data/building2/phase4/2015-05-22.csv', index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
# result = VisualData('building2', 'intr', datetime.datetime.now(), datetime.datetime.now(), dataframe, timestamp='2015-05-22')
# visualizer = Visualizer('building2')
# visualizer.visualize_gap_inerpolation(result, folder='gap_fills')


# dataframe = pd.read_csv('data/building2/phase4/2015-06-27.csv', index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
# result = VisualData('building2', 'intr', datetime.datetime.now(), datetime.datetime.now(), dataframe, timestamp='2015-06-09')
# visualizer = Visualizer('building2')
# visualizer.visualize_gap_inerpolation(result, folder='gap_fills')


# df_building5_gaps = pd.read_csv('data_trans_opt/building5/gap_std_fill.csv', index_col="Date",  parse_dates=['Date'])

# for index, row in df_building5_gaps.iterrows():
#     current_date_str = index.strftime('%Y-%m-%d')
#     dataframe = pd.read_csv(f'data/building5/phase4/{current_date_str}.csv', index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
#     result = VisualData('building5', 'intr', datetime.datetime.now(), datetime.datetime.now(), dataframe, timestamp=current_date_str)
#     visualizer = Visualizer('building5')
#     visualizer.visualize_gap_inerpolation(result, folder='gap_fills')


# df_building7_gaps = pd.read_csv('data_trans_opt/building7/gap_std_fill.csv', index_col="Date",  parse_dates=['Date'])

# for index, row in df_building7_gaps.iterrows():
#     current_date_str = index.strftime('%Y-%m-%d')
#     dataframe = pd.read_csv(f'data/building7/phase4/{current_date_str}.csv', index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
#     result = VisualData('building7', 'intr', datetime.datetime.now(), datetime.datetime.now(), dataframe, timestamp=current_date_str)
#     visualizer = Visualizer('building7')
#     visualizer.visualize_gap_inerpolation(result, folder='gap_fills')




eda = EDA()
eda.load_datasets()

visualize_properties(eda.df_b0, 'building0')
visualize_properties(eda.df_b1, 'building1')
visualize_properties(eda.df_b2, 'building2')
visualize_properties(eda.df_b3, 'building3')
visualize_properties(eda.df_b4, 'building4')
visualize_properties(eda.df_b5, 'building5')
visualize_properties(eda.df_b7, 'building7')

visualize_frequencies(eda.df_b0, 'building0')
visualize_frequencies(eda.df_b1, 'building1')
visualize_frequencies(eda.df_b2, 'building2')
visualize_frequencies(eda.df_b3, 'building3')
visualize_frequencies(eda.df_b4, 'building4')
visualize_frequencies(eda.df_b5, 'building5')
visualize_frequencies(eda.df_b7, 'building7')

# visualize_dataset_raw('building1', datetime.datetime(2014, 3, 13), datetime.datetime(2015, 6, 29))
# visualize_dataset_raw('building2', datetime.datetime(2014, 2, 16), datetime.datetime(2015, 6, 29))
# visualize_dataset_raw('building3', datetime.datetime(2014, 4, 8), datetime.datetime(2015, 4, 13))
# visualize_dataset_raw('building4', datetime.datetime(2014, 2, 1), datetime.datetime(2014, 10, 30))
# visualize_dataset_raw('building5', datetime.datetime(2014, 3, 8), datetime.datetime(2015, 3, 21))

#visualize_dataset_metrics('building4', datetime.datetime(2014, 2, 1), datetime.datetime(2014, 10, 31))

#visualize_dataset_metrics('building1', datetime.datetime(2014, 3, 13), datetime.datetime(2015, 6, 29))

# visualize_dataset_metrics('building0', datetime.datetime(2014, 6, 18), datetime.datetime(2014, 10, 12))

# visualize_dataset_metrics('building2', datetime.datetime(2015, 1, 27), datetime.datetime(2015, 1, 27))

