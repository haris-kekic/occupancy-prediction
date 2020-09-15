# -----------------------------------------------------------
# Klasse implementiert den erweiterten NIOM zur Bestimmung  
# der Anwesenheit von Personen im Haushalt
#
# (C) 2020 Haris Kekic, Fernuniversität in Hagen
# -----------------------------------------------------------

import numpy as np 
from decimal import Decimal
import datetime 
import pandas as pd
from pandas import DataFrame


class OccupancyEstimation:

    def __init__(self, date_current_day, df_current_day, date_previous_day = None,  df_previous_day = None):
        ''' Konstruktor - initialisiert das Objekt

            :date_current_day: Datum des aktuellen Tages, auf welchem der Algorithmus angewandt wird
            :df_current_day: Pandas-DataFrame mit den Verbrauchsdaten für den aktuellen Tag
            :date_previous_day: Datum des vorigen Tages, auf welchem der Algorithmus angewandt wird
            :df_previous_day: Pandas-DataFrame mit den Verbrauchsdaten für den vorigen Tag
        '''
        # Initialisiere Klassen-Properties
        self.date_current_day = date_current_day
        self.df_current_day = df_current_day.copy() # Kopie erstellen

        # Hier ist anzumerken, dass der vorherige Tag nicht umbedingt angegeben wedrden muss.
        # Das ist dann der Fall, wenn eine größere Lücke in den Daten gegeben ist und somit
        # der vorherige Tag nicht vorhanden ist oder es ist der erste Tag im Datensatz
        self.date_previous_day = date_previous_day if df_previous_day is not None else None
        self.df_previous_day = df_previous_day.copy() if df_previous_day is not None else None

        # Dem DataFrame des aktuellen Tages werden neue Spalten (Features) hinzugefügt
        # die im Zuge des Algorithmus berechnet und befüllt werden
        self.df_current_day['DayPart'] = None
        self.df_current_day['ClusterSize'] = None
        self.df_current_day['ClusterTimeFrom'] = None
        self.df_current_day['ClusterTimeTo'] = None
        self.df_current_day['AVG_ThresholdFactor'] = 1
        self.df_current_day['STDDEV_ThresholdFactor'] = 1
        self.df_current_day['RANGE_ThresholdFactor'] = 1
        self.df_current_day['AVG_PrevMinutes'] = None
        self.df_current_day['STDDEV_PrevMinutes'] = None
        self.df_current_day['RANGE_PrevMinutes'] = None
        self.df_current_day['AVG_Threshold'] = None
        self.df_current_day['STDDEV_Threshold'] = None
        self.df_current_day['RANGE_Threshold'] = None
        self.df_current_day['AVG_OCCUPATION_EVENT'] = 0
        self.df_current_day['STDDEV_OCCUPATION_EVENT'] = 0
        self.df_current_day['RANGE_OCCUPATION_EVENT'] = 0
        self.df_current_day['OCCUPATION_EVENT'] = 0
        self.df_current_day['OCCUPIED'] = 0

    def run(self, day_part_lengths, cluster_sizes, threshold_time_from, threshold_time_to, avg_threshold_factors, stddev_threshold_factors, range_threshold_factors, avg_time_window, stddev_time_window, range_time_window):
        ''' Startet den Algorighmus mit gegebenen Parametern.

            :day_part_lengths: Liste, die die Dauer der einzelnen Tagesabschnitte in Minuten enthält
            :cluster_sizes: Liste, die die Größe der Cluster in Minuten enthält für einzelne Tagesabschnitte zur Berechnung der Anwesenheitsereignisse 
            :threshold_time_from: Beginnzeit ab der die Berechnung des initialen Schwellwertes für die Metriken berechnet werden soll
            :threshold_time_to: Endzeit ab der die Berechnung des initialen Schwellwertes für die Metriken berechnet werden soll
            :avg_threshold_factors: Liste, die die Faktoren für einzelne Tagesabschnitte, die mit dem initialen Schwellwert der AVG-Metrik multipliziert werden
            :stddev_threshold_factors: Liste, die die Faktoren für einzelne Tagesabschnitte, die mit dem initialen Schwellwert der STDDEV-Metrik multipliziert werden
            :range_threshold_factors: Liste, die die Faktoren für einzelne Tagesabschnitte, die mit dem initialen Schwellwert der RANGE-Metrik multipliziert werden
            :avg_time_window: Größe des Zeitfensters zur Berechnung der AVG-Metrik
            :stddev_time_window: Größe des Zeitfensters zur Berechnung der STDDEV-Metrik
            :range_time_window: Größe des Zeitfensters zur Berechnung der RANGE-Metrik
        '''
        # Der Algorithmus wird in 5. aufeinander folgenden Schritten ausgeführt
        # 1. Schritt: Initialisierung der Parameter, die in Folge im Algorithmus zur Berechnung der Anwesenheit verwendet werden
        self.df_current_day = self.__setup(day_part_lengths, cluster_sizes, avg_threshold_factors, stddev_threshold_factors, range_threshold_factors)

        # 2. Metriken werden für die angegebenen Zeitfenstergrößen berechnet
        self.df_current_day = self.__calc_metrics(avg_time_window, stddev_time_window, range_time_window)

        # 3. Die Schwellwerte für Metriken werden berechnet (sowohl der initiale Schwellwert, 
        # als auch der Endgültige für die einzelnen Tagesabschnitte nach Multiplikation mit dem Faktor)
        self.df_current_day = self.__calc_thresholds(threshold_time_from, threshold_time_to, avg_threshold_factors, stddev_threshold_factors, range_threshold_factors)

        # 4. Berechnung der Anwesenheitsereignisse für und über einzelne Metriken
        self.df_current_day = self.__calc_occupation_events()

        # 5. Endgültige Berechnung der Anwesenheit auf Basis der Anwesenheitsereignisse
        self.df_current_day = self.__calc_occupation_clustered()

        # Kopie des Ergebnisses wird zurückgeliefert
        return self.df_current_day.copy()
    
    def __setup(self, day_part_lengths, cluster_sizes, avg_threshold_factors, stddev_threshold_factors, range_threshold_factors):
        ''' Setzt die Parameter für den Algorithmus bzw. fügt sie zu dem DataFrame des aktuellen Tages als Spalten, für spätere Berechnung

            :day_part_lengths: Liste, die die Dauer der einzelnen Tagesabschnitte in Minuten enthält
            :cluster_sizes: Liste, die die Größe der Cluster in Minuten enthält für einzelne Tagesabschnitte zur Berechnung der Anwesenheitsereignisse 
            :avg_threshold_factors: Liste, die die Faktoren für einzelne Tagesabschnitte, die mit dem initialen Schwellwert der AVG-Metrik multipliziert werden
            :stddev_threshold_factors: Liste, die die Faktoren für einzelne Tagesabschnitte, die mit dem initialen Schwellwert der STDDEV-Metrik multipliziert werden
            :range_threshold_factors: Liste, die die Faktoren für einzelne Tagesabschnitte, die mit dem initialen Schwellwert der RANGE-Metrik multipliziert werden
        '''

        # Hier initialisieren wir die Startzeit des Tages: also 00:00 Uhr
        # Die Startzeit wird für die Berechnung der Tagesabschnitte herangezogen
        daypart_start_time = self.date_current_day
        # Umwandeln in einen String
        data_current_str = self.date_current_day.strftime('%Y-%m-%d')

        # Überprüfen ob die Summe der Minuten = 1440 ist, was einem Tag entspricht
        if (sum(day_part_lengths) != 1440):
            raise NameError(f'Die Summe der Minuten in dayparts-Parameter muss 1440 betragen! Datei: {data_current_str}.csv')
        
        # Die Tagesabschnitte werden durchlaufen
        for daypart in range(len(day_part_lengths)):

            # Die Endzeit des Tagesabschnites wird berechnet (StartZeit + Dauer)
            daypart_end_time = daypart_start_time + datetime.timedelta(minutes=day_part_lengths[daypart])

            # Wenn der letzte Tagesabschnitt addiert wird, dann kommt man auf 00:00 des nächsten Tages
            # aus dem Grund ziehen wir hier eine Sekunde ab um auf 23:59:59 des aktuellen Tages wieder zu kommen
            if (daypart == len(day_part_lengths) - 1):
                daypart_end_time = daypart_end_time - datetime.timedelta(seconds=1)

            # In Strings umwandeln
            from_minutes_str = daypart_start_time.strftime('%H:%M')
            to_minutes_str = daypart_end_time.strftime('%H:%M')

            # Der Tagesabschnitt wird aus dem DataFrame extrahiert und die Parameter
            # für den Tagesabschnitt über den wir gerade iterieren werden im Frame gesetzt
            self.df_current_day.loc[(self.df_current_day['LocalTime'] >= from_minutes_str) & (self.df_current_day['LocalTime'] <= to_minutes_str), 
                        ['DayPart', 'ClusterSize', 'AVG_ThresholdFactor', 'STDDEV_ThresholdFactor', 'RANGE_ThresholdFactor']] = [daypart, cluster_sizes[daypart], avg_threshold_factors[daypart], stddev_threshold_factors[daypart], range_threshold_factors[daypart]]

            # Berechne die Cluster-Bereiche für den aktuellen Tagesabschnitt
            # Die Cluster-Bereiche 'time_from' - 'time_to' sind Hilfsspalten
            # Sie setzen für jeden Zeiteintrag im Frame den Clusterbereich in dem
            # der Eintrag liegt. Z.B. wenn die Cluster-Größe 10 ist, dann wird
            # der Eintrag zu Minute 14:22 in den Cluster 14:20-14:30 fallen.
            # Das wird später gebraucht, wenn wir die Anwesenheitsevents über Cluster verteilen.
            # Wenn wir z.B. zwei Ereignisse innerhalb eines Clusters haben, wird Anwesenheit für
            # den gesamten Cluster gesetzt (siehe NIOM)

            # Zuerst erzeugen wir die Cluster anhand der Cluster-Größe für den gesamten Tag
            # Bei Cluster-Größe 10, wird hier eine Liste erzeugt z.b. [00:00, 00:10, 00:20 ...]
            # Die letzte Zeit ist die Endzeit des Tagesabschnittes.
            # Wir berechnen also für jeden Tagesabschnitt die Cluster-Bereiche einzeln
            # da die Cluster-Größe pro Abschnitt sich unterscheiden kann laut dem erweiterten Algorithmus
            cluster_time_range = pd.date_range(from_minutes_str, to_minutes_str, freq=f"{cluster_sizes[daypart]}min")

            # Wir erzeugen einen neuen Frame in dem die Zeiten der Cluster gespeichert werden
            df_cluster_times = DataFrame(columns=['time_from', 'time_to'])

            # Die Cluster-Zeiten durchlaufen
            for time_i in range(len(cluster_time_range)):
                # Die Zeiten werden aus der Bereichsliste entnommen, 
                # wobei die Grenzen der Cluster in der Liste aufeinander folgen
                start_time = cluster_time_range[time_i]
                if (time_i < (len(cluster_time_range) - 1)):
                    end_time = cluster_time_range[time_i + 1]
                else:
                    end_time = datetime.time(23, 59, 59)

                # In Strings umwandeln, und im Frame abspeichern
                start_time_str = start_time.strftime('%H:%M')
                end_time_str = end_time.strftime('%H:%M')
                df_cluster_times = df_cluster_times.append({'time_from': start_time_str, 'time_to': end_time_str }, ignore_index=True)
            
            # Der Tagesabschnitt wird aus dem aktuellen Tag 'ausgeschnitten'
            df_current_day_daypart = self.df_current_day[( self.df_current_day['LocalTime'] >= from_minutes_str) & ( self.df_current_day['LocalTime'] <= to_minutes_str)]

            # Durchlaufen des Tagesabschnittes bzw. der Einzelnen Zeiteinträge des Abschnittes
            for index, row in df_current_day_daypart.iterrows():
                # LocalTime beinhaltet die aktuelle Zeit des Zeiteintrages
                # Wir suchen dann ensprechend den Cluster-Frame nach dem Cluster
                # in dem die aktuelle Zeit fällt. Z.b. wenn wir jetzt LocalTime = 02:18 
                # und die Cluster-Größe 10 Minuten beträgt, dann fällt das in den Cluster 02:10 - 02:20
                self.df_cluster_time = df_cluster_times[(row['LocalTime'] >= df_cluster_times['time_from']) & (row['LocalTime'] <= df_cluster_times['time_to'])]
                # Die Cluster-Beginn und Endzeit wird dann entsprechend in den Frame
                # des aktuellen Tages in zwei neue Hilfsspalten eingetragen
                self.df_current_day.at[index, 'ClusterTimeFrom'] =  self.df_cluster_time.time_from.iloc[0]
                self.df_current_day.at[index, 'ClusterTimeTo'] =  self.df_cluster_time.time_to.iloc[0]

            
            # Wir setzten jetzt die Tegesabschnitt-Beginnzeit 
            # auf das Ende des aktuellen Tagesabschnittes für die nächste interation
            daypart_start_time = daypart_end_time

        # Der aktuelle Tag, der in dieser Phase modifiziert wurde, wird zurückgegeben
        return self.df_current_day

    def __calc_metrics(self, avg_time_window, stddev_time_window, range_time_window):
        ''' Berechnet die Werte der einzelnen Algorithmus-Metriken
            gemäß der übergebenenZeitfenstergrößen

            :avg_time_window: Größe des Zeitfensters zur Berechnung der AVG-Metrik
            :stddev_time_window: Größe des Zeitfensters zur Berechnung der STDDEV-Metrik
            :range_time_window: Größe des Zeitfensters zur Berechnung der RANGE-Metrik
        '''

        # Überprufen ob der vorherige Tag übergeben wurde
        if (self.df_previous_day is not None):
            # Wir nehmen noch die letzten Einträge vom letzten Tag um die Mittelwerte für den Anfang des aktuellen Tages berechnen zu können
            # Wir nehmen den größten Zeitfensterwert für die Metriken um genau so viel Zeit vom vorherigen Tag 'abzuschneiden'
            # wie viel gebraucht wird um die Metriken für den Anfang des aktuellen Tages zu berechnen
            max_timewindow = max(avg_time_window, stddev_time_window, range_time_window) 
            day_start_time = self.date_current_day - datetime.timedelta(minutes=max_timewindow)
            # Das gebrauchte Teil des vorherigen Tages wird abgeschnitten 
            df_prev_day_window = self.df_previous_day[self.df_previous_day['LocalTime'] > day_start_time.strftime('%H:%M')].copy()
            # Das abgeschnittene Teil des vorherigen Tages wird an Anfange des aktuellen Tages angefügt
            self.df_current_day = pd.concat([df_prev_day_window, self.df_current_day])
            
        # Iterieren durch die Zeiteinträge des aktuellen Tages und die Metriken berechnen
        for index, row in self.df_current_day[self.df_current_day.index >= self.date_current_day].iterrows():
            cur_local_datetime = index
            
            # Metrik AVG wird gemäß der Zeitfenstergröße berechnet.
            # Die Summe der einzelnen Meter wird zur berechnung herangezogen.
            avg_lag_time = cur_local_datetime - datetime.timedelta(minutes=avg_time_window)
            avg_df_prev = self.df_current_day[(self.df_current_day.index > avg_lag_time) & (self.df_current_day.index <= cur_local_datetime)]
            avg_prev_value = avg_df_prev['Total'].mean()
            self.df_current_day.at[index, 'AVG_PrevMinutes'] = avg_prev_value # Füge den berechneten Wert in die neue Spalte hinzu

            # Metrik STDDEV wird gemäß der Zeitfenstergröße berechnet.
            # Die Summe der einzelnen Meter wird zur berechnung herangezogen.
            std_lag_time = cur_local_datetime - datetime.timedelta(minutes=stddev_time_window)
            std_df_prev = self.df_current_day[(self.df_current_day.index > std_lag_time) & (self.df_current_day.index <= cur_local_datetime)]
            std_prev_value = std_df_prev['Total'].std()
            self.df_current_day.at[index, 'STDDEV_PrevMinutes'] = std_prev_value # Füge den berechneten Wert in die neue Spalte hinzu

            # Metrik RANGE wird gemäß der Zeitfenstergröße berechnet.
            # Die Summe der einzelnen Meter wird zur berechnung herangezogen.
            range_lag_time = cur_local_datetime - datetime.timedelta(minutes=range_time_window)
            range_df_prev = self.df_current_day[(self.df_current_day.index > range_lag_time) & (self.df_current_day.index <= cur_local_datetime)]
            range_min_prev_value = range_df_prev['Total'].min()
            range_max_prev_value = range_df_prev['Total'].max() 
            range_prev_value = range_max_prev_value - range_min_prev_value # Berechne die Differenz zwischen den MIN und MAX Wert innerhalb des Zeitfensters
            self.df_current_day.at[index, 'RANGE_PrevMinutes'] = range_prev_value # Füge den berechneten Wert in die neue Spalte hinzu
        
        # Lösche die im Zuge des Rechenprozesses hinzugefügen Einträge des vorherigen Tages aus dem aktuellen Tag
        self.df_current_day.drop(self.df_current_day[self.df_current_day.index < self.date_current_day].index, inplace=True)
        return self.df_current_day
    
    def __calc_thresholds(self, threshold_time_from, threshold_time_to, avg_threshold_factors, std_threshold_factors, range_threshold_factors):
        '''

            :threshold_time_from: Beginnzeit ab der die Berechnung des initialen Schwellwertes für die Metriken berechnet werden soll
            :threshold_time_to: Endzeit ab der die Berechnung des initialen Schwellwertes für die Metriken berechnet werden soll
            :avg_threshold_factors: Liste, die die Faktoren für einzelne Tagesabschnitte, die mit dem initialen Schwellwert der AVG-Metrik multipliziert werden
            :stddev_threshold_factors: Liste, die die Faktoren für einzelne Tagesabschnitte, die mit dem initialen Schwellwert der STDDEV-Metrik multipliziert werden
            :range_threshold_factors: Liste, die die Faktoren für einzelne Tagesabschnitte, die mit dem initialen Schwellwert der RANGE-Metrik multipliziert werden
        '''

        from_minutes_str = threshold_time_from
        to_minutes_str = threshold_time_to

        # Aus dem aktuellen Tag wird der Teil, der zur Berechnung des Schwellwertes, verwendet wird, ausgeschnitten
        threshold_time_range = self.df_current_day[(self.df_current_day['LocalTime'] >= from_minutes_str) & (self.df_current_day['LocalTime'] < to_minutes_str)]

        # Für einzelne Metriken wird dann der Schwellwert berechnet - 
        # der Maximalwert innerhalb des abgeschnittenen Zeitintervalls
        avg_initial_threshold = threshold_time_range['AVG_PrevMinutes'].max()
        std_initial_threshold = threshold_time_range['STDDEV_PrevMinutes'].max()
        range_initial_threshold = threshold_time_range['RANGE_PrevMinutes'].max()

        # Iteriere über alle Tagesabschnitte und berechne den endgüligen Schwellwert für alle Metriken 
        # für die Tagesabschnitte in dem der initelle Schwellwert mit dem Faktor multipliziert wird
        for daypart in range(5):
            self.df_current_day.loc[self.df_current_day['DayPart'] == daypart, ['AVG_Threshold', 'STDDEV_Threshold', 'RANGE_Threshold']] = [avg_initial_threshold * avg_threshold_factors[daypart], 
                                                            std_initial_threshold * std_threshold_factors[daypart], 
                                                            range_initial_threshold * range_threshold_factors[daypart]] 

                    
        return self.df_current_day

    
    def __calc_occupation_events(self):
        ''' Berechnet die Anwesenheitsereignisse für die Metriken '''

        # Durchlaufe den ganzen Tag und vergleiche die Werte der Metriken mit dem 
        # im vorherigen Schritt berechneten Schwellwert
        for index, row in self.df_current_day.iterrows():
            avg_event = 1 if row['AVG_PrevMinutes'] > row['AVG_Threshold'] else 0
            std_event = 1 if row['STDDEV_PrevMinutes'] > row['STDDEV_Threshold'] else 0
            range_event = 1 if row['RANGE_PrevMinutes'] > row['RANGE_Threshold'] else 0
            

            self.df_current_day.at[index, 'AVG_OCCUPATION_EVENT'] = avg_event
            self.df_current_day.at[index, 'STDDEV_OCCUPATION_EVENT'] = std_event
            self.df_current_day.at[index, 'RANGE_OCCUPATION_EVENT'] = range_event

            # Das endgültige Anwesenheitsereignis für einen Zeiteintrag ist dann gegeben wenn 
            # ein Anwesenheitseregnis mindestens einer Metrik positiv ist
            self.df_current_day.at[index, 'OCCUPATION_EVENT'] = 1 if avg_event == 1 or std_event == 1 or range_event == 1 else 0
    
        return self.df_current_day

    
    def __calc_occupation_clustered(self):
        ''' Berechnet die endgültige Anwesenheit für einen ganzen Cluster der Zeiteinträge während eines Tages'''

        # Die positiven Anwesneheitsevents innerhalb eines Clusters werden gezählt
        df_event_counts = self.df_current_day[['ClusterTimeFrom', 'ClusterTimeTo', 'OCCUPATION_EVENT']][self.df_current_day['OCCUPATION_EVENT'] == 1].groupby(['ClusterTimeFrom', 'ClusterTimeTo']).count()

        # Die gruppierten Cluster werden durchlaufen und wenn in einem Cluster 
        # mehr als zwei Ereignisse positiv sind, dann wird der ganze Cluster als positiv gesetzt
        # Hier wird deutlich, warum wir die Zeiteinträge in Cluster eingeteilt haben,
        # so dass wir dann die Anwesnheit für den ganzen Cluster bzw. Zeiteinträge die sich
        # innerhalb eines Clusters befinden, setzen können
        for index, row in df_event_counts.iterrows():
            if (row['OCCUPATION_EVENT'] > 2):
                self.df_current_day.loc[(self.df_current_day['ClusterTimeFrom'] == index[0]) & (self.df_current_day['ClusterTimeTo'] == index[1]), 'OCCUPIED'] = 1

        # Weiter zählen wir wie viele Zeiteinträge (Zeit in Minuten) 
        # am Abend (Tagesabschnitt = 3) die Anwesenheit positiv berechnet wurde
        df_cur_eve = self.df_current_day[self.df_current_day['DayPart'] == 3]
        df_cur_eve_minutes = df_cur_eve[df_cur_eve['OCCUPIED'] == 1]['OCCUPIED'].count()

        # Wenn mehr als 60 Zeiteinträge (60 Minuten) positiv sind, dann wird auch 
        # der letzte Abschnitt des Tages auf Anwesenheit = positiv gesetzt
        # Das ist eine einfache Annahme, dass wenn jemand z.B. am Abend bis 22Uhr
        # mindestens 60 Minuten zu Hause war, dann nehmen wir an dass er auch Anfang
        # der Nacht, also von 22:00-00:00Uhr zu Hause sein wird (bzw. die Nacht verbringen wird)
        if (df_cur_eve_minutes > 60):
            self.df_current_day.loc[self.df_current_day['DayPart'] == 4, 'OCCUPIED'] = 1
        
        # Wir überprüfen ob der vorherige Tag übergeben wurde,
        # wenn ja, dann zählen wir die Minuten der Anwesenheit 
        # in den letzten zwei Abschnitten des vorherigen Tages (Abend u. Nacht)
        if (self.df_previous_day is not None):
            df_prev_night = self.df_previous_day[(self.df_previous_day['DayPart'] == 3) | (self.df_previous_day['DayPart'] == 4)]
            df_prev_night_minutes = df_prev_night[df_prev_night['OCCUPIED'] == 1]['OCCUPIED'].count()

            # Beträgt die Anwesenheit am Abend und Nacht des vorherigen Tages
            # länger als 60 Minuten, nehmen wir an, dass auch die ganze Nacht
            # des aktuellen Tages (Abschnitt=0) dann jemand zu Hause ist
            if (df_prev_night_minutes > 60):
                self.df_current_day.loc[self.df_current_day['DayPart'] == 0, 'OCCUPIED'] = 1
        else:
            # Wurde kein vorheriger Tag übergeben, nehmen wir standardmäßig für die Nacht
            # des aktuellen Tages eine Anwesenheit an, da es aus menschlichem Verhalten
            # am 'wahrscheinlichsten' ist, dass Menschen zu Hause die Nacht verbringen
            self.df_current_day.loc[self.df_current_day['DayPart'] == 0, 'OCCUPIED'] = 1

        return self.df_current_day

 