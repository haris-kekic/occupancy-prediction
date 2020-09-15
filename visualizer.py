# -----------------------------------------------------------
# Enthält Klasse zum Visualisieren/Plotten unterschiedlicher
# Diagramme aus den Daten die gewonnen werden
# (C) 2020 Haris Kekic, Fernuniversität in Hagen
# -----------------------------------------------------------

import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from decimal import Decimal
import datetime 
import pandas as pd
from visual_data import VisualData
from date_helper import get_weekday
from os import path
from matplotlib.lines import Line2D

class Visualizer:

    def __init__(self, dataset):
        self.dataset = dataset
        self.csv_data_path_base = f'data/{self.dataset}/'

    def visualize_dataset_metrics(self, start_date, end_date, phase='phase7', algo='std', sample_size_minutes=1):
        date_range = pd.date_range(start_date, end_date)

        if ((phase != 'phase6') and (phase != 'phase7')):
            print(f'Metriken aus Daten in Phase {phase} sind nicht visualizierbar!' )
            return

        csv_path_this_phase = f'{self.csv_data_path_base}/{phase}/{algo}'	

        for current_date in date_range:
            current_date_str = current_date.strftime('%Y-%m-%d')
            filename_cur_date = f'{csv_path_this_phase}/{current_date_str}.csv'

            if (not path.exists(filename_cur_date)):
                print('File existiert nicht und kann nicht visualisiert werden')
                continue

            weekday = get_weekday(current_date)
            title = f'{current_date_str} ({weekday})'
            dataframe = pd.read_csv(filename_cur_date, index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
            time_axis_step = 60 / sample_size_minutes
            print(f'Visualisiere: {filename_cur_date}') 
            visual_data = VisualData(self.dataset, title, datetime.datetime.now(), datetime.datetime.now(), dataframe, time_axis_step)
            self.visualize_metrics(visual_data, f'{algo}_{sample_size_minutes}min')
    
    def visualize_metrics(self, visual_data, folder):
        ''' Plottet das Diagramm mit den NIOM Artefakten, sprich den Diagrammen zum Einzelstromverbrauch, Gesamtstromverbrauch, Leg-Mittelwert, Leg-Stddv, Leg-MinMax
            :visual: Das DataResult mit Metadaten und den DataFrame das geplottet werden soll
            :folder: Ordner in dem die Diagrambilder abgespeichert werden sollen
        '''
       
        # pd.set_option('display.max_rows', 500)
        # pd.set_option('display.max_columns', 500)
        # pd.set_option('display.width', 1000)

        # Erstelle den Subplot für die Artefakten und Metriken des NIOM
        fig, ax = plt.subplots(5, 1)

        # Nehme das DataFrame zum Plotten
        dataframe = visual_data.data

        # Schrittgröße für Zeitachse
        time_axis_step = visual_data.time_axis_step

        # 1. Subplot: Smart Meter Einzelverbrauch
        # Zuerst instanziere die Axis mit der 'Anwesend' Achse
        color = 'green'
        ax[0].set_title(f'{visual_data.dataset_name} / {visual_data.title} / Zähler-Einzelverbrauch')
        ax[0].set_ylabel('Anwesend', color=color)
        ax[0].plot(dataframe['OCCUPIED'], color=color, linewidth=0.2, drawstyle='steps')
        ax[0].tick_params(axis='y', labelcolor=color)
        ax[0].set_yticks([0, 1])
        ax[0].fill_between(dataframe.index, dataframe['OCCUPIED'], step="pre", alpha=0.2)

        ax_02 = ax[0].twinx()  # Intanziere die geteilte Achse für den Einzelstromverbrauch der Meter

        color = 'purple'
        ax_02.set_xlabel('Uhrzeit')
        ax_02.set_ylabel('Verbrauch (Smart Meter)', color=color)
        ax_02.plot(dataframe['Meter1'], color='darkorange', linewidth=1, label='Meter 1')
        ax_02.plot(dataframe['Meter2'], color='gray', linewidth=1, label='Meter 2')
        ax_02.plot(dataframe['Meter3'], color='darkgreen', linewidth=1, label='Meter 3')
        ax_02.plot(dataframe['Meter4'], color='blue', linewidth=1, label='Meter 4')
        ax_02.plot(dataframe['Meter5'], color='red', linewidth=1, label='Meter 5')
        ax_02.plot(dataframe['Meter6'], color='darkgray', linewidth=1, label='Meter 6')
        ax_02.plot(dataframe['Meter7'], color='cyan', linewidth=1, label='Meter 7')
        ax_02.plot(dataframe['Meter8'], color='yellow', linewidth=1, label='Meter 8')
        ax_02.plot(dataframe['Meter9'], color='violet', linewidth=1, label='Meter 9')
        ax_02.tick_params(axis='y', labelcolor=color)
        ax_02.set_xticklabels([dt_index.strftime('%H') for dt_index, row in dataframe[::time_axis_step].iterrows()], rotation='vertical')

        plt.legend(loc='upper left')
        plt.xticks(dataframe.index[::time_axis_step])

        print([dt_index.strftime('%H') for dt_index, row in dataframe[::time_axis_step].iterrows()])
        print(dataframe.index[::time_axis_step])

        # 2. Subplot: Gesamtverbrauch
        # Zuerst instanziere die Axis mit der 'Anwesend' Achse
        color = 'green'
        ax[1].set_title(f'{visual_data.dataset_name} / {visual_data.title} / Gesamtverbrauch')
        ax[1].set_ylabel('Anwesend', color=color)
        ax[1].plot(dataframe['OCCUPIED'], color=color, linewidth=0.2, drawstyle='steps')
        ax[1].tick_params(axis='y', labelcolor=color)
        ax[1].set_yticks([0, 1])
        ax[1].fill_between(dataframe.index, dataframe['OCCUPIED'], step="pre", alpha=0.2)

        ax_12 = ax[1].twinx()  # Intanziere die geteilte Achse für den Einzelstromverbrauch der Meter

        color = 'black'
        ax_12.set_xlabel('Uhrzeit')
        ax_12.set_ylabel('Gesamtverbrauch', color=color)
        ax_12.plot(dataframe['Total'], color=color, linewidth=1)
        ax_12.tick_params(axis='y', labelcolor=color)
        ax_12.set_xticklabels([dt_index.strftime('%H') for dt_index, row in dataframe[::time_axis_step].iterrows()], rotation='vertical')

        plt.xticks(dataframe.index[::time_axis_step])

        # 3. Subplot: Durchschnittverbrauch
        # Zuerst instanziere die Axis mit der 'Anwesend' Achse
        color = 'green'
        ax[2].set_title(f'{visual_data.dataset_name} / {visual_data.title} / Durchschnittsverbrauch')
        ax[2].set_ylabel('Anwesend', color=color)
        ax[2].plot(dataframe['OCCUPIED'], color=color, linewidth=0.2, drawstyle='steps')
        ax[2].tick_params(axis='y', labelcolor=color)
        ax[2].set_yticks([0, 1])
        ax[2].fill_between(dataframe.index, dataframe['OCCUPIED'], step="pre", alpha=0.2)

        ax_22 = ax[2].twinx() # Intanziere die geteilte Achse für den Einzelstromverbrauch der Meter

        color = 'blue'
        ax_22.set_xlabel('Uhrzeit')
        ax_22.set_ylabel('Durchschnittsverbrauch (KW * 15min)', color=color)
        ax_22.plot(dataframe['AVG_PrevMinutes'], color=color, linewidth=1)
        ax_22.plot(dataframe['AVG_Threshold'], color='black', linewidth=1, linestyle='--')
        ax_22.tick_params(axis='y', labelcolor=color)
        ax_22.set_xticklabels([dt_index.strftime('%H') for dt_index, row in dataframe[::time_axis_step].iterrows()], rotation='vertical')

        plt.xticks(dataframe.index[::time_axis_step])

        # 4. Subplot: Standarddeviation
        # Zuerst instanziere die Axis mit der 'Anwesend' Achse
        color = 'darkgreen'
        ax[3].set_title(f'{visual_data.dataset_name} / {visual_data.title} / Standarddeviation')
        ax[3].set_ylabel('Anwesend', color=color)
        ax[3].plot(dataframe['OCCUPIED'], color=color, linewidth=0.2, drawstyle='steps')
        ax[3].tick_params(axis='y', labelcolor=color)
        ax[3].set_yticks([0, 1])
        ax[3].fill_between(dataframe.index, dataframe['OCCUPIED'], step="pre", alpha=0.2)

        ax_32 = ax[3].twinx()  # Intanziere die geteilte Achse für den Einzelstromverbrauch der Meter

        color = 'green'
        ax_32.set_xlabel('Uhrzeit')
        ax_32.set_ylabel('Stddev (STDEV * 15min)', color=color)
        ax_32.plot(dataframe['STDDEV_PrevMinutes'], color=color, linewidth=1)
        ax_32.plot(dataframe['STDDEV_Threshold'], color='black', linewidth=1, linestyle='--')
        ax_32.tick_params(axis='y', labelcolor=color)
        ax_32.set_xticklabels([dt_index.strftime('%H') for dt_index, row in dataframe[::time_axis_step].iterrows()], rotation='vertical')

        plt.xticks(dataframe.index[::time_axis_step])

        # 5. Subplot: Min-Max-Range
        # Zuerst instanziere die Axis mit der 'Anwesend' Achse
        color = 'green'
        ax[4].set_title(f'{visual_data.dataset_name} / {visual_data.title} / Min-Max Range')
        ax[4].set_ylabel('Anwesend', color=color)
        ax[4].plot(dataframe['OCCUPIED'], color=color, linewidth=0.2, drawstyle='steps')
        ax[4].tick_params(axis='y', labelcolor=color)
        ax[4].set_yticks([0, 1])
        ax[4].fill_between(dataframe.index, dataframe['OCCUPIED'], step="pre", alpha=0.2)

        ax_42 = ax[4].twinx() # Intanziere die geteilte Achse für den Einzelstromverbrauch der Meter

        color = 'red'
        ax_42.set_xlabel('Uhrzeit')
        ax_42.set_ylabel('Min-Max Range (RANGE * 15min)', color=color)
        ax_42.plot(dataframe['RANGE_PrevMinutes'], color=color, linewidth=1)
        ax_42.plot(dataframe['RANGE_Threshold'], color='black', linewidth=1, linestyle='--')
        ax_42.tick_params(axis='y', labelcolor=color)
        ax_42.set_xticklabels([dt_index.strftime('%H') for dt_index, row in dataframe[::time_axis_step].iterrows()], rotation='vertical')

        plt.xticks(dataframe.index[::time_axis_step])

        # Adjustieren der Anzeige und Abspeichern
        plt.subplots_adjust(bottom=0.15)
        fig.set_size_inches(15, 20, forward=True)
        #plt.show(block=True)
        save_path = f'visuals/{visual_data.dataset_name}/{folder}/{visual_data.dataset_name}_{visual_data.title}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close('all')

    
    def visualize_gap_inerpolation(self, visual_data, folder, col_prefix='Imputed_'):
        ''' Generiert Diagrame für die einzlnen Meter mit der Anzeige für die in Lücke interpolierten (eingefügten) Werte
            :visual_data: Das DataResult mit Metadaten und den DataFrame das geplottet werden soll
            :gap_start: Startzeit der Lücke in der Zeitreihe
            :gap_end: Endzeit der Lücke in der Zeitreihe
            :folder: Ordner in dem die Diagrambilder abgespeichert werden sollen
            :col_prefix: Prefix der Meter-Spalte die geplottet werden soll
        '''

        # Hole den DataFrame mit den Werten
        dataframe = visual_data.data

        # Metadaten für die Anzeige
        datestamp = visual_data.timestamp
        
        # Plotte die einzelnen Meter mit spezieller Anzeige für die Lücken
        # Also haben wir wieder eine geteilte X-Achse, wobei die Intervalle rundherum und das 
        # Lückenintervall angezeigt werden
        # Anschließend werden die Diagramme einzeln wieder abgespeichert.

        # Wir müssen hier x='time' nicht spezifizieren, da 'time' ein Index und keine Column ist
        meter_label = 'Meter1'
        ax = dataframe.plot.line(y=meter_label, color='blue', linewidth=0.8)
        dataframe.plot.line(ax = ax, y=col_prefix + meter_label, color='red', linewidth=1, xticks=dataframe.index[::15])
        ax.set_xticklabels([dt_index.strftime('%H:%M') for dt_index, row in dataframe[::15].iterrows()], rotation='vertical')
        save_path = f'visuals/{visual_data.dataset_name}/{folder}/b4_imp_{datestamp}_{meter_label}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close('all')


        meter_label = 'Meter2'
        ax = dataframe.plot.line(y=meter_label, color='blue', linewidth=0.8)
        dataframe.plot.line(ax = ax, y=col_prefix + meter_label, color='red', linewidth=1, xticks=dataframe.index[::15])
        ax.set_xticklabels([dt_index.strftime('%H:%M') for dt_index, row in dataframe[::15].iterrows()], rotation='vertical')
        save_path = f'visuals/{visual_data.dataset_name}/{folder}/b4_imp_{datestamp}_{meter_label}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close('all')


        meter_label = 'Meter3'
        ax = dataframe.plot.line(y=meter_label, color='blue', linewidth=0.8)
        dataframe.plot.line(ax = ax, y=col_prefix + meter_label, color='red', linewidth=1, xticks=dataframe.index[::15])
        ax.set_xticklabels([dt_index.strftime('%H:%M') for dt_index, row in dataframe[::15].iterrows()], rotation='vertical')
        save_path = f'visuals/{visual_data.dataset_name}/{folder}/b4_imp_{datestamp}_{meter_label}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close('all')


        meter_label = 'Meter4'
        ax = dataframe.plot.line(y=meter_label, color='blue', linewidth=0.8)
        dataframe.plot.line(ax = ax, y=col_prefix + meter_label, color='red', linewidth=1, xticks=dataframe.index[::15])
        ax.set_xticklabels([dt_index.strftime('%H:%M') for dt_index, row in dataframe[::15].iterrows()], rotation='vertical')
        save_path = f'visuals/{visual_data.dataset_name}/{folder}/b4_imp_{datestamp}_{meter_label}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close('all')


        meter_label = 'Meter5'
        ax = dataframe.plot.line(y=meter_label, color='blue', linewidth=0.8)
        dataframe.plot.line(ax = ax, y=col_prefix + meter_label, color='red', linewidth=1, xticks=dataframe.index[::15])
        ax.set_xticklabels([dt_index.strftime('%H:%M') for dt_index, row in dataframe[::15].iterrows()], rotation='vertical')
        save_path = f'visuals/{visual_data.dataset_name}/{folder}/b4_imp_{datestamp}_{meter_label}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close('all')


        meter_label = 'Meter6'
        ax = dataframe.plot.line(y=meter_label, color='blue', linewidth=0.8)
        dataframe.plot.line(ax = ax, y=col_prefix + meter_label, color='red', linewidth=1, xticks=dataframe.index[::15])
        ax.set_xticklabels([dt_index.strftime('%H:%M') for dt_index, row in dataframe[::15].iterrows()], rotation='vertical')
        save_path = f'visuals/{visual_data.dataset_name}/{folder}/b4_imp_{datestamp}_{meter_label}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close('all')



        meter_label = 'Meter7'
        ax = dataframe.plot.line(y=meter_label, color='blue', linewidth=0.8)
        dataframe.plot.line(ax = ax, y=col_prefix + meter_label, color='red', linewidth=1, xticks=dataframe.index[::15])
        ax.set_xticklabels([dt_index.strftime('%H:%M') for dt_index, row in dataframe[::15].iterrows()], rotation='vertical')
        save_path = f'visuals/{visual_data.dataset_name}/{folder}/b4_imp_{datestamp}_{meter_label}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close('all')


        meter_label = 'Meter8'
        ax = dataframe.plot.line(y=meter_label, color='blue', linewidth=0.8)
        dataframe.plot.line(ax = ax, y=col_prefix + meter_label, color='red', linewidth=1, xticks=dataframe.index[::15])
        ax.set_xticklabels([dt_index.strftime('%H:%M') for dt_index, row in dataframe[::15].iterrows()], rotation='vertical')
        save_path = f'visuals/{visual_data.dataset_name}/{folder}/b4_imp_{datestamp}_{meter_label}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close('all')


        meter_label = 'Meter9'
        ax = dataframe.plot.line(y=meter_label, color='blue', linewidth=0.8)
        dataframe.plot.line(ax = ax, y=col_prefix + meter_label, color='red', linewidth=1, xticks=dataframe.index[::15])
        ax.set_xticklabels([dt_index.strftime('%H:%M') for dt_index, row in dataframe[::15].iterrows()], rotation='vertical')
        save_path = f'visuals/{visual_data.dataset_name}/{folder}/b4_imp_{datestamp}_{meter_label}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close('all')


    def visualize_model_scores(self, df_score, df_best_loss_score, experiment, path):
        
        fig, ax = plt.subplots(2, 1)

        ax[0].plot(df_score['EpochTrainLoss'], color='blue', label='Training Loss', linewidth=1)
        ax[0].plot(df_score['EpochValLoss'], color='lightsalmon', label='Validation Loss', linewidth=1)
        ax[0].plot(df_score['EpochSmoothValLoss'], color='red', label='Smoothed Validation Loss', linewidth=0.7)
        ax[0].set_ylabel('Loss')
        ax[0].set_xlabel('Epochs')
        ax[0].set_title(f'Fehler/Loss [Experiment {experiment}]')
        ax[0].legend()

        ax[1].plot(df_score['EpochTrainAcc'], color='blue', label='Training Accuracy', linewidth=1)
        ax[1].plot(df_score['EpochValAcc'], color='lightsalmon', label='Validation Accuracy', linewidth=1)
        ax[1].plot(df_score['EpochSmoothValAcc'], color='red', label='Smoothed Validation Accuracy', linewidth=0.7)
        ax[1].set_ylabel('Accuracy')
        ax[1].set_xlabel('Epochs')
        ax[1].set_title(f'Trefferquote/Accuracy [Experiment {experiment}]')
        ax[1].legend()

        plt.subplots_adjust(bottom=0.15)
        fig.set_size_inches(12, 10, forward=True)
        save_path = f'{path}/results_{experiment}/train_val_scores.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)


        # ax = df_score.plot.line(y="EpochTestLoss", color='pink', label="Test Loss", linewidth=0.7)
        # df_score.plot.line(ax = ax, y="EpochSmoothTestLoss", color='darkmagenta', label="Smoothed Test Loss", linewidth=1)
        # plt.title(f'Testfehler (Experiment {experiment})')
        # plt.xlabel('Epochen')
        # plt.ylabel('Accuracy')
        # plt.legend()
        # save_path = f'{path}/results_{experiment}/test_loss.png'
        # plt.savefig(save_path, bbox_inches='tight', dpi=300)

        # ax = df_score.plot.line(y="EpochTestAcc", color='pink', label="Test Accurancy", linewidth=0.7)
        # df_score.plot.line(ax = ax, y="EpochSmoothTestAcc", color='darkmagenta', label="Smoothed Test Accuracy", linewidth=1)
        # plt.title(f'Test Trefferquote (Experiment {experiment})')
        # plt.xlabel('Epochen')
        # plt.ylabel('Accuracy')
        # plt.legend()
        # save_path = f'{path}/results_{experiment}/test_acc.png'
        # plt.savefig(save_path, bbox_inches='tight', dpi=300)


        plt.close('all')

    def visualize_dataset_raw(self, start_date, end_date, phase='phase2', sample_size_minutes=1):
        date_range = pd.date_range(start_date, end_date)

        csv_path_this_phase = f'{self.csv_data_path_base}/{phase}'	

        for current_date in date_range:
            current_date_str = current_date.strftime('%Y-%m-%d')
            filename_cur_date = f'{csv_path_this_phase}/{current_date_str}.csv'

            if (not path.exists(filename_cur_date)):
                print('File existiert nicht und kann nicht visualisiert werden')
                continue

            weekday = get_weekday(current_date)
            title = f'{current_date_str} ({weekday})'
            dataframe = pd.read_csv(filename_cur_date, index_col='LocalDateTime', parse_dates=['LocalDateTime', 'UtcDateTime'])
            time_axis_step = 60 / sample_size_minutes
            print(f'Visualisiere: {filename_cur_date}') 
            visual_data = VisualData(self.dataset, title, current_date, current_date, dataframe, time_axis_step)
            self.visualize_raw_raw(visual_data, f'raw')
    
    def visualize_raw_raw(self, visual_data, folder):

        # Erstelle den Subplot für die Artefakten und Metriken des NIOM
        fig, ax = plt.subplots(2, 1)

        # Nehme das DataFrame zum Plotten
        dataframe = visual_data.data

        # Schrittgröße für Zeitachse
        #time_axis_step = visual_data.time_axis_step

        print(visual_data.start_date)
        start = visual_data.start_date
        end = visual_data.start_date + datetime.timedelta(hours=23)

        hour_range = pd.date_range(start = start, end = end, freq = "H")
        hours_str = [hr.strftime('%H') for hr in hour_range]

        print(hour_range)
        print(hours_str)

        # 1. Subplot: Smart Meter Einzelverbrauch
        color = 'purple'
        ax[0].set_title(f'{visual_data.dataset_name} / {visual_data.title} / Smart Meter - Verbrauch')
        ax[0].set_xlabel('Uhrzeit')
        ax[0].set_ylabel('Verbrauch (Smart Meter)', color=color)
        ax[0].plot(dataframe['Meter1'], color='darkorange', linewidth=1, label='Meter 1')
        ax[0].plot(dataframe['Meter2'], color='gray', linewidth=1, label='Meter 2')
        ax[0].plot(dataframe['Meter3'], color='darkgreen', linewidth=1, label='Meter 3')
        ax[0].plot(dataframe['Meter4'], color='blue', linewidth=1, label='Meter 4')
        ax[0].plot(dataframe['Meter5'], color='red', linewidth=1, label='Meter 5')
        ax[0].plot(dataframe['Meter6'], color='darkgray', linewidth=1, label='Meter 6')
        ax[0].plot(dataframe['Meter7'], color='cyan', linewidth=1, label='Meter 7')
        ax[0].plot(dataframe['Meter8'], color='yellow', linewidth=1, label='Meter 8')
        ax[0].plot(dataframe['Meter9'], color='violet', linewidth=1, label='Meter 9')
        ax[0].tick_params(axis='y', labelcolor=color)
        ax[0].set_xticks(hour_range)
        ax[0].set_xticklabels(hours_str, rotation='vertical')
        ax[0].legend(loc='upper left')

        # 2. Subplot: Gesamtverbrauch
        color = 'black'
        ax[1].set_title(f'{visual_data.dataset_name} / {visual_data.title} / Gesamtverbrauch')
        ax[1].set_xlabel('Uhrzeit')
        ax[1].set_ylabel('Gesamtverbrauch', color=color)
        ax[1].plot(dataframe['Total'], color=color, linewidth=1)
        ax[1].tick_params(axis='y', labelcolor=color)
        ax[1].set_xticks(hour_range)
        ax[1].set_xticklabels(hours_str, rotation='vertical')

        # Adjustieren der Anzeige und Abspeichern
        plt.subplots_adjust(bottom=0.15)
        fig.set_size_inches(20, 15, forward=True)
        #plt.show(block=True)
        save_path = f'visuals/{visual_data.dataset_name}/{folder}/{visual_data.dataset_name}_{visual_data.title}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close('all')

    def visualize_frequencies(self, visual_data, folder):

        # Erstelle den Subplot für die Artefakten und Metriken des NIOM
        fig, ax = plt.subplots(10, 1)

        # Nehme das DataFrame zum Plotten
        dataframe = visual_data.data

        sm_bins = np.arange(0.0, 4.0, 0.10)
        total_bins = np.arange(0.0, 8.0, 0.10)

        color='darkorange'
        ax[0].set_title(f'{visual_data.dataset_name} / {visual_data.title}')
        ax[0].set_ylabel('Meter 1', color=color)
        ax[0].hist(dataframe['Meter1'], color=color, bins=sm_bins)

        color='gray'
        ax[1].set_ylabel('Meter 2', color=color)
        ax[1].hist(dataframe['Meter2'], color=color, bins=sm_bins)

        color='darkgreen'
        ax[2].set_ylabel('Meter 3', color=color)
        ax[2].hist(dataframe['Meter3'], color=color, bins=sm_bins)

        color='blue'
        ax[3].set_ylabel('Meter 4', color=color)
        ax[3].hist(dataframe['Meter4'], color=color, bins=sm_bins)

        color='red'
        ax[4].set_ylabel('Meter 5', color=color)
        ax[4].hist(dataframe['Meter5'], color=color, bins=sm_bins)

        color='darkgray'
        ax[5].set_ylabel('Meter 6', color=color)
        ax[5].hist(dataframe['Meter6'], color=color, bins=sm_bins)

        color='cyan'
        ax[6].set_ylabel('Meter 7', color=color)
        ax[6].hist(dataframe['Meter7'], color=color, bins=sm_bins)

        color='yellow'
        ax[7].set_ylabel('Meter 8', color=color)
        ax[7].hist(dataframe['Meter8'], color=color, bins=sm_bins)

        color='violet'
        ax[8].set_ylabel('Meter 9', color=color)
        ax[8].hist(dataframe['Meter9'], color=color, bins=sm_bins)

        color='black'
        ax[9].set_ylabel('Total', color=color)
        ax[9].hist(dataframe['Total'], color=color, bins=total_bins)

        plt.subplots_adjust(bottom=0.15)
        fig.set_size_inches(8, 20, forward=True)

        save_path = f'visuals/{visual_data.dataset_name}/{folder}/{visual_data.dataset_name}_{visual_data.title}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

        plt.close('all')

    def visualize_properties(self, visual_data, folder):

        # Nehme das DataFrame zum Plotten
        dataframe = visual_data.data
        medianprops = dict(linestyle='-', linewidth=2, color='green')
        meanprops = dict(linestyle='-', linewidth=2, color='blue')
        custom_lines = [Line2D([0], [0], color='blue', lw=2),
                        Line2D([0], [0], color='green', lw=2),
                        Line2D([0], [0], color='red', lw=2)]

        columns = ['Meter1', 'Meter2', 'Meter3', 'Meter4', 'Meter5', 'Meter6', 'Meter7', 'Meter8', 'Meter9', 'Total']
        dataframe_properties = dataframe.describe(include=['float'])
        std = dataframe_properties.loc[['std' ]][columns].values[0]
        
        std = np.insert(std, 0, None)
        std_cols = np.insert(columns, 0, '')

        ax = dataframe.boxplot(column=columns, grid=False, showfliers=False, showmeans=True, meanline=True, meanprops=meanprops, medianprops=medianprops)
        ax.scatter(std_cols, std, marker='_', c='red', label='Stadardabweichung')
        ax.legend(custom_lines, ['Mittelwert', 'Median', 'Standardabweichung'], loc='upper left')
        plt.title(f'{visual_data.dataset_name} / {visual_data.title}')
        fig = plt.gcf()
        fig.set_size_inches(9, 6)

        save_path = f'visuals/{visual_data.dataset_name}/{folder}/{visual_data.dataset_name}_{visual_data.title}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

        plt.close('all')



