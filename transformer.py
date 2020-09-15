# -----------------------------------------------------------
# Klasse implementiert die Phasenweise Transformation 
# der Daten aus dem Originaldatensatz und bereitet
# diese für weitere Verwendung (Visualisieren und
# zum Trainieren der Deep Learning Modelle) vor.
# Die Phasen können einzeln aber auch gemeinsam in einem
# Transofmrationprozess ausgeführt werden
#
# (C) 2020 Haris Kekic, Fernuniversität in Hagen
# -----------------------------------------------------------

import numpy as np 
from decimal import Decimal
import datetime, pytz 
import pandas as pd
from pandas import DataFrame
import glob
from os import path
from os import stat
from occupancy_estimation import OccupancyEstimation
import date_helper as dh
from scipy.spatial import distance


class Transformer:

	def __init__(self, dataset):
		''' Initialisiert das Transformationsobjekt

			:dataset: Datensatz der Transformiert werden soll (buldingN)
		'''
		self.dataset = dataset
		self.csv_data_path_base = f'data/{self.dataset}/'	
		self.csv_trans_opt_path_base = f'data_trans_opt/{self.dataset}/'

	def transform_phase0_to_phase1(self, start_date, end_date):
		''' Transofmiert die Daten aus Phase 0 in Phase 1. 
			Transofmrationsdetails, der Dokumentation entnehmen

			:start_date: Das Anfangsdatum, ab welchem die Datentransformation ausgeführt werden soll
			:end_date: Das Enddatum, bis zu welchem die Datentransformation ausgeführt werden soll
		'''
		this_phase = 'phase0'
		next_phase = 'phase1'

		print(f'Transormiere {this_phase} in {next_phase}')

		# Die Pfade zu den Dateien setzen
		# Pfad zu den Dateien der aktuellen Phase
		csv_path_this_phase = self.csv_data_path_base + this_phase 
		# Pfad zu den Dateien der nächsten Phase als Ergebnis der Transformation
		csv_path_next_phase = self.csv_data_path_base + next_phase 

		# Der gewünschte Datumsbereich, aus welchem die Daten
		# transformiert werden sollen, wird erstellt
		date_range = pd.date_range(start_date, end_date)

		# Iteriere durch die Daten, denn jede Datei entspricht einem Tag
		# Somit werden Tage nacheinander bearbeitet
		for current_date in date_range:
    		###TIMESTAMP zu Performancemessung und Ausgabe###
			now = datetime.datetime.now()
			current_time = now.strftime("%H:%M:%S")
			print("START: ", current_time)
			###TIMESTAMP zu Performancemessung und Ausgabe###

			# Der Dateiname der zu bearbeitenden Datei wird erstellt
			# um die Daten für den aktuell zu bearbeitenden Tag zu lesen
			current_date_str = current_date.strftime('%Y-%m-%d')
			filename_cur_date = f'{csv_path_this_phase}/dataset_{current_date_str}.csv'
			
			print("Bearbeite: ", filename_cur_date)

			# Überprüfen ob die Datei für das gegebene Datum vorhanden ist
			# Wenn nicht, wird eine Nachricht ausgegeben und zum nächsten Datum iteriert
			if (not path.exists(filename_cur_date)):
				print('File existiert nicht und keine Transformation wird stattfinden')
				continue

			if (stat(filename_cur_date).st_size == 0):
				print('File existiert, aber leer.')
				continue
			
			# Die Datei wird in ein Pandas-DataFrame eingelesen
			df_current = pd.read_csv(filename_cur_date, skiprows=1, header=None)

			# Unzulässige Zeilen in Originaldaten werden entfernt
			# Diese Zeilen enthalten immer wieder den ganzen Header
			df_current.drop(df_current[df_current[0] == 'timestamp'].index, inplace=True)
			# Lösche alle Zeilen die eine NULL in der "timestamp" Spalte haben
			df_current.dropna(subset=[0], inplace=True) 

			# Inline Funktionen die einzeln auf Einträge des Frames später während des Mappings ausgeführt werden
			# Diese Funktionen wandeln das UNIX-Timestamp in ein UTC-Datum
			func_unix_ts_to_utc_datetime = lambda ts: datetime.datetime.utcfromtimestamp(int(float(ts))).strftime('%Y-%m-%d %H:%M') # Sekunden werden explizit weggelassen, da wir auf Minuten gruppieren werden
			func_unix_ts_to_utc_date = lambda ts: datetime.datetime.utcfromtimestamp(int(float(ts))).strftime('%Y-%m-%d')
			func_unix_ts_to_utc_time = lambda ts: datetime.datetime.utcfromtimestamp(int(float(ts))).strftime('%H:%M') # Sekunden werden explizit weggelassen, da wir auf Minuten gruppieren werden

			# Diese Funktionen wandeln das UNIX-Timestamp in ein lokales Datum
			func_unix_ts_to_local_datetime = lambda ts: datetime.datetime.fromtimestamp(int(float(ts))).strftime('%Y-%m-%d %H:%M')
			func_unix_ts_to_local_date = lambda ts: datetime.datetime.fromtimestamp(int(float(ts))).strftime('%Y-%m-%d')
			func_unix_ts_to_local_time = lambda ts: datetime.datetime.fromtimestamp(int(float(ts))).strftime('%H:%M')

			# Diese Funktion liefert für das lokale Datum den Wochentag, wobei Montag...Sonntag => 0...6
			func_weekday = lambda ts: datetime.datetime.fromtimestamp(int(float(ts))).weekday()

			func_season = lambda ts: dh.get_season(datetime.datetime.fromtimestamp(int(float(ts))))

			# Ergebnis der Transformation wird in einen neuen Frame zusammangefasst
			# der wiederum als neue Datei in den Ordner für die nächste Phase gespeichert werden soll
			df_next = DataFrame()

			# df_current[X] ist ein Pandas-Series Objekt. Mit map und einer Inline-Funltion haben wir dann Zugriff auf einzelne Einträge (Zeilen)
			# df_current[0] enthält das UNIX-Timestamp und wird entsprechend in den Inline-Funktionen in ein Datum konvertiert, sei es UTC oder Local
			df_next['LocalDate'] = df_current[0].map(func_unix_ts_to_local_date)
			df_next['LocalDateTime'] = df_current[0].map(func_unix_ts_to_local_datetime) 
			df_next['LocalTime'] = df_current[0].map(func_unix_ts_to_local_time)
			df_next['UtcDateTime'] = df_current[0].map(func_unix_ts_to_utc_datetime)
			df_next['UtcDate'] = df_current[0].map(func_unix_ts_to_utc_date)
			df_next['UtcTime'] = df_current[0].map(func_unix_ts_to_utc_time)
			df_next['Season'] = df_current[0].map(func_season)
			df_next['Weekday'] = df_current[0].map(func_weekday)
			df_next['Meter1'] = df_current[1].fillna(method='ffill').fillna(method='bfill').fillna(0).astype('float') / 1000 # Zuerst die NULL Werte mit LOCF befüllen und dann Meter W -> KW umwandeln
			df_next['Meter2'] = df_current[2].fillna(method='ffill').fillna(method='bfill').fillna(0).astype('float') / 1000 # Zuerst die NULL Werte mit LOCF befüllen und dann Meter W -> KW umwandeln
			df_next['Meter3'] = df_current[3].fillna(method='ffill').fillna(method='bfill').fillna(0).astype('float') / 1000 # Zuerst die NULL Werte mit LOCF befüllen und dann Meter W -> KW umwandeln
			df_next['Meter4'] = df_current[4].fillna(method='ffill').fillna(method='bfill').fillna(0).astype('float') / 1000 # Zuerst die NULL Werte mit LOCF befüllen und dann Meter W -> KW umwandeln
			df_next['Meter5'] = df_current[5].fillna(method='ffill').fillna(method='bfill').fillna(0).astype('float') / 1000 # Zuerst die NULL Werte mit LOCF befüllen und dann Meter W -> KW umwandeln
			df_next['Meter6'] = df_current[6].fillna(method='ffill').fillna(method='bfill').fillna(0).astype('float') / 1000 # Zuerst die NULL Werte mit LOCF befüllen und dann Meter W -> KW umwandeln
			df_next['Meter7'] = df_current[7].fillna(method='ffill').fillna(method='bfill').fillna(0).astype('float') / 1000 # Zuerst die NULL Werte mit LOCF befüllen und dann Meter W -> KW umwandeln
			df_next['Meter8'] = df_current[8].fillna(method='ffill').fillna(method='bfill').fillna(0).astype('float') / 1000 # Zuerst die NULL Werte mit LOCF befüllen und dann Meter W -> KW umwandeln
			df_next['Meter9'] = df_current[9].fillna(method='ffill').fillna(method='bfill').fillna(0).astype('float') / 1000 # Zuerst die NULL Werte mit LOCF befüllen und dann Meter W -> KW umwandeln
			
			# Downsampling auf Minuten -> siehe Oben, die Sekunden wurden in den Zeitstempel weggelassen
			df_next = df_next.groupby(['LocalDateTime', 'LocalDate', 'LocalTime', 'UtcDateTime', 'UtcDate', 'UtcTime', 'Season',  'Weekday']).mean()

			# Beim GroupBy werden angegebene Spalten zum Multiindex gemacht, 
			# und hier wird dieser wieder in normale Spalten umgewandelt
			df_next.reset_index(drop=False, inplace=True)

			# Die Verbrauchssumme wird berechnet
			df_next['Total']  = df_next[['Meter1', 'Meter2', 'Meter3', 'Meter4', 'Meter5', 'Meter6', 'Meter7', 'Meter8', 'Meter9']].sum(axis=1, skipna=True)
			
			# Speichern der Transformationsergebnisse in eine neue Datei für die nächste Phase
			# Index muss auf False gesetzt werden, da wir oben ja den Index gelöscht haben
			# Wenn wir nicht False setzen, dann erstellt Pandas einen Integr-Index in der CSV
			df_next.to_csv(f'{csv_path_next_phase}/{current_date_str}.csv', index=False)

			###TIMESTAMP zu Performancemessung und Ausgabe###
			now = datetime.datetime.now()
			current_time = now.strftime("%H:%M:%S")
			print("END: ", current_time)
			###TIMESTAMP zu Performancemessung und Ausgabe###
		

		pass
	

	def transform_phase1_to_phase2(self, start_date, end_date, detect_gaps=True):
		''' Transofmiert die Daten aus Phase 1 in Phase 2. 
			Transofmrationsdetails, der Dokumentation entnehmen.

			:start_date: Das Anfangsdatum, ab welchem die Datentransformation ausgeführt werden soll
			:end_date: Das Enddatum, bis zu welchem die Datentransformation ausgeführt werden soll
			:detect_gaps: Angabe, ob im Zuge der Transformation auch die 'buildingN_gaps.csv' Datei, die die Lücken in den Daten aufzeigt, erstellt werden soll
		'''

		this_phase = 'phase1'
		next_phase = 'phase2'

		print(f'Transormiere {this_phase} in {next_phase}')

		# Die Pfade zu den Dateien setzen
		# Pfad zu den Dateien der aktuellen Phase
		csv_path_this_phase = self.csv_data_path_base + this_phase
		# Pfad zu den Dateien der nächsten Phase als Ergebnis der Transformation
		csv_path_next_phase = self.csv_data_path_base + next_phase

		# Der gewünschte Datumsbereich, aus welchem die Daten
		# transformiert werden sollen, wird erstellt
		date_range = pd.date_range(start_date, end_date)

		# Erstellen des Frames in den wir die Daten aus den Dateien zunächst temporär laden werden
		df_current_all = DataFrame()

		# Iteriere durch die Daten, denn jede Datei entspricht einem Tag
		# Somit werden Tage nacheinander bearbeitet.
		# Da in dieser Phase die Tagesdaten korrenkt ohne Überlappungen in der Zeit in Dateien aufgeteilt werden sollen,
		# holen wir zunächst alle Daten aus den Dateien der aktuellen Phase und laden
		# sie iterativ in df_current_all temporär rein. Später werden wir diese Datei nutzen
		# und die Daten korrekt in Tage aufteilen und in der Phase2 abspeichern
		for current_date in date_range:
    		###TIMESTAMP zu Performancemessung und Ausgabe###
			now = datetime.datetime.now()
			current_time = now.strftime("%H:%M:%S")
			print("START: ", current_time)
			###TIMESTAMP zu Performancemessung und Ausgabe###

    		# Stringified Datum
			current_date_str = current_date.strftime('%Y-%m-%d')
			filename_cur_date = f'{csv_path_this_phase}/{current_date_str}.csv'
			
			print("Bearbeite: ", filename_cur_date)
			# überprüfen ob die Datei exisistiert.
			# wenn nicht, dann einfach überspringen und zum nächsten Datum wächseln
			if (not path.exists(filename_cur_date)):
				print('File existiert nicht und keine Transformation wird stattfinden')
				continue
			
			# Lese die Datei aus der aktuellen Phase1 ein
			df_current = pd.read_csv(filename_cur_date)
			# Hänge sie an das Frame, dass alle Daten aus allen Dateien enthalten wird
			df_current_all = df_current_all.append(df_current)

			###TIMESTAMP zu Performancemessung und Ausgabe###
			now = datetime.datetime.now()
			current_time = now.strftime("%H:%M:%S")
			print("END: ", current_time)
			###TIMESTAMP zu Performancemessung und Ausgabe###



		# Dateien aus dem All-Frame nehmen und auf einzelne Tagesdateien aufsplitten - jetzt aber ohne Überlappungen
		for current_date in date_range:
			current_date_str = current_date.strftime('%Y-%m-%d')
			# Aus dem All-Frame filtern wir die Daten für das aktuelle Datum 
			df_day = df_current_all[df_current_all['LocalDate'] == current_date.strftime('%Y-%m-%d')]
			# Falls Daten für das aktuelle Datum nicht exisiteren, einfach überspringen
			if (df_day.shape[0] == 0):
				continue
			
			# Sonst eine Datei in der neuen Phase anlegen
			df_day.to_csv(f'{csv_path_next_phase}/{current_date_str}.csv', index=False)

		# Wenn detect_gaps = True, dann wird über die neuen Daten in Phase 2 der Algorithmus
		# zur Bestimmung von Lücken in Daten ausgeführt und die ensprechende Datei wird angelet
		if (detect_gaps == True):
			self.detect_gaps(start_date, end_date)

		pass
	
	def transform_phase2_to_phase3(self, start_date, end_date):
		''' Transofmiert die Daten aus Phase 2 in Phase 3. 
			Transofmrationsdetails, der Dokumentation entnehmen.

			Löscht bzw. ignoriert Dateien die im data_trans_opt/buildingN/data_ignore.csv 
			zum Löschen bzw. nicht weiterreichen (ignorieren) im Transformationsprozess aufgeführt wurden. 
			Im Endeffekt, werden diese Dateien einfach ignoriert und der Rest wird dann in der Phase3 abgespeichert

         	:start_date: Das Anfangsdatum, ab welchem die Datentransformation ausgeführt werden soll
			:end_date: Das Enddatum, bis zu welchem die Datentransformation ausgeführt werden soll
        '''
		# Aktuelle Phase mit bestehenden Daten, die transformiert werden
		this_phase = 'phase2'
		# Die nächste Phase in der die Ergebnisse der Transformation abgespeichert werden
		next_phase = 'phase3'

		print(f'Transormiere {this_phase} in {next_phase}')

		# Pfad zur Dateien der Phasen aufbauen
		csv_path_this_phase = self.csv_data_path_base + this_phase
		csv_path_next_phase = self.csv_data_path_base + next_phase

		# Der gewünschte Datumsbereich, aus welchem die Daten
		# transformiert werden sollen, wird erstellt
		date_range = pd.date_range(start_date, end_date)

		# Die Imputation der fehlenden Einträge wird auf gesamten Datensatz durchgeführt.
		# Das heißt, die Daten aus einzelnen CSV Dateien werden am Ende 
		# in diesen einen Frame zusammengefasst und die Analyse wird
		# auf diesem Frame dann auch ausgeführt
		#df_current_all = DataFrame()

		# Das Löschen der gegebenen Einträge aus dem Frame
		# Welche Einträge/Tage gelöscht werden ist in der imutation_settings CSV-Datei hinterlegt
		csv_data_delete_settings_path = f'{self.csv_trans_opt_path_base}/data_ignore.csv'
		df_remove_list = pd.read_csv(csv_data_delete_settings_path, parse_dates=['Date'], index_col=False)

		# Iteriere durch alle Daten des gegebenen Datumbereichs
		for current_date in date_range:
			###TIMESTAMP zu Performancemessung und Ausgabe###
			now = datetime.datetime.now()
			current_time = now.strftime("%H:%M:%S")
			print("START: ", current_time)
			###TIMESTAMP zu Performancemessung und Ausgabe###

			current_date_str = current_date.strftime('%Y-%m-%d')
			filename_cur_date = f'{csv_path_this_phase}/{current_date_str}.csv'

			# Erstelle einen String-Zeitstempel aus dem aktuellen Datum durch den wir iterieren
			current_date_str = current_date.strftime('%Y-%m-%d')

			# Baue den Pfad und Filenamen zu der CSV Datei für 
			# das aktuelle Datum in der Iteration
			filename_cur_date = f'{csv_path_this_phase}/{current_date_str}.csv'
			
			print("Bearbeite: ", filename_cur_date)
			# Wenn die Datei für das aktuelle Datum nicht existiert, 
			# dann gebe meldung aus und mache beim nächsten Datum
			# aus dem zu iterierenden Bereich über
			if (not path.exists(filename_cur_date)):
				print('File existiert nicht und keine Transformation wird stattfinden')
				continue
			
			# Die CSV Datei für das aktuelle Datum existiert, so lade sie in ein Pandas-Frame
			df_current = pd.read_csv(filename_cur_date, parse_dates=['UtcDateTime', 'LocalDateTime'], index_col='LocalDateTime')
			# Normalerweise erstellt Pandas immer ein Index, wenn kein angegeben wurde, also löschen
			#df_current.reset_index(drop=True, inplace=True)


			###*** 1. Löschen unerwünschter Einträge/Tage  ***###
			# Suche alle Datum aus die in der Liste der zu löschenden Daten ist und überspringe sie
			if (np.datetime64(current_date) in df_remove_list['Date'].values):
				print('Die Settings-Datei beinhaltet das Datam zum Löschen. Die Datei wird ignoriert.')
				continue
			
			# Wir verwandeln den Index 'LocalDateTime' vor dem Speichern in eine Spalte, 
			# damit die Spaltenordnung aufrecht erhalten bleibt
			df_current.reset_index(inplace=True) 
			# Wir setzen noch Index=False, da Pandas sonst eigenen Index (integer) hinzufügt
			df_current.to_csv(f'{csv_path_next_phase}/{current_date_str}.csv', index=False)

			###TIMESTAMP zu Performancemessung und Ausgabe###
			now = datetime.datetime.now()
			current_time = now.strftime("%H:%M:%S")
			print("END: ", current_time)
			###TIMESTAMP zu Performancemessung und Ausgabe###
		
		pass

	def transform_phase3_to_phase4(self, start_date, end_date):
		''' Transofmiert die Daten aus Phase 3 in Phase 4. 
			Transofmrationsdetails, der Dokumentation entnehmen.
		
			Verwendet Standard-Imputation Methoden um kleinere Lücken 1-2 Studen
			zu befüllen. Welche Lücken und zu welchen Datum befüllt werden
			ist in der Datei data_trans_opt/buildingN/gap_std_fill.csv hinterlegt

         	:start_date: Das Anfangsdatum, ab welchem die Datentransformation ausgeführt werden soll
			:end_date: Das Enddatum, bis zu welchem die Datentransformation ausgeführt werden soll
        '''
		# Aktuelle Phase mit bestehenden Daten, die transformiert werden
		this_phase = 'phase3'
		# Die nächste Phase in der die Ergebnisse der Transformation abgespeichert werden
		next_phase = 'phase4'

		print(f'Transormiere {this_phase} in {next_phase}')

		# Pfad zur Dateien der Phasen aufbauen
		csv_path_this_phase = self.csv_data_path_base + this_phase
		csv_path_next_phase = self.csv_data_path_base + next_phase

		# Der gewünschte Datumsbereich, aus welchem die Daten
		# transformiert werden sollen, wird erstellt
		date_range = pd.date_range(start_date, end_date)
		

		# Die Imputation der fehlenden Einträge wird auf gesamten Datensatz durchgeführt.
		# Das heißt, die Daten aus einzelnen CSV Dateien werden am Ende 
		# in diesen einen Frame zusammengefasst und die Analyse wird
		# auf diesem Frame dann auch ausgeführt
		#df_current_all = DataFrame()


		# Die kleineren Lücken, die befüllt werden sind in der imputation_settings CSV Datei hinterlegt
		csv_data_imp_settings_path = f'{self.csv_trans_opt_path_base}/gap_std_fill.csv'
		df_smp_list = pd.read_csv(csv_data_imp_settings_path, parse_dates=['Date', 'GapStart', 'GapEnd'], index_col=False)


		# Iteriere durch alle Daten des gegebenen Datumbereichs
		for current_date in date_range:

    		###TIMESTAMP zu Performancemessung und Ausgabe###
			now = datetime.datetime.now()
			current_time = now.strftime("%H:%M:%S")
			print("START: ", current_time)
			###TIMESTAMP zu Performancemessung und Ausgabe###

			# Erstelle einen String-Zeitstempel aus dem aktuellen Datum durch den wir iterieren
			current_date_str = current_date.strftime('%Y-%m-%d')

			# Baue den Pfad und Filenamen zu der CSV Datei für 
			# das aktuelle Datum in der Iteration
			filename_cur_date = f'{csv_path_this_phase}/{current_date_str}.csv'
			
			print("Bearbeite: ", filename_cur_date)
			# Wenn die Datei für das aktuelle Datum nicht existiert, 
			# dann gebe meldung aus und mache beim nächsten Datum
			# aus dem zu iterierenden Bereich über
			if (not path.exists(filename_cur_date)):
				print('File existiert nicht und keine Transformation wird stattfinden')
				continue
			
			# Die CSV Datei für das aktuelle Datum existiert, so lade sie in ein Pandas-Frame
			df_current = pd.read_csv(filename_cur_date, parse_dates=['UtcDateTime', 'LocalDateTime'], index_col='LocalDateTime')
			# Normalerweise erstellt Pandas immer ein Index, wenn kein angegeben wurde, also löschen
			#df_current.reset_index(drop=True, inplace=True)

			# # Fasse die einzelnen Frames zu einem Frame zusammen
			# df_current_all = df_current_all.append(df_current, ignore_index=True)

			# In die Inputed-Spalten für einzelne Meter werden die neuen Werte, die für None
			# imputiert werden gespeichert. Diese Spalten können später zur Visualisierung
			# der eingefügten Daten im Plot heranzgozogen werden
			for meter_num in range(1, 10):
				df_current[f'Imputed_Meter{meter_num}'] = None

			# Neue temporäre Hilfs-Spalten für die Bearbeitung
			# Werden vor dem Speichern des Endergebnisses wieder entfernt
			df_current['IsImputed'] = False
			df_current['IsGap'] = False

			### *** 2. Daten-Imputation für kleinere Lücken / Simple Imputation ***###
			for index, row in df_smp_list[df_smp_list['Date'] == current_date].iterrows():
				df_current = self.__impute_simple(df_current, row['GapStart'], row['GapEnd'], row['GapStartDelta'], row['GapEndDelta'], row['Meter1'], row['Meter2'], row['Meter3'], row['Meter4'], row['Meter5'], row['Meter6'], row['Meter7'], row['Meter8'], row['Meter9'])

			df_current.reset_index(inplace=True)
			df_current.to_csv(f'{csv_path_next_phase}/{current_date_str}.csv', index=False)

			###TIMESTAMP zu Performancemessung und Ausgabe###
			now = datetime.datetime.now()
			current_time = now.strftime("%H:%M:%S")
			print("END: ", current_time)
			###TIMESTAMP zu Performancemessung und Ausgabe###
		
		pass

	def transform_phase4_to_phase5(self, start_date, end_date):
		''' Transofmiert die Daten aus Phase 4 in Phase 5. 
			Transofmrationsdetails, der Dokumentation entnehmen.

			Verwendet Similar Behavior Imputation Alogirthm um Lücken die über
			mehrere Studenen an einem Tag vorzufinden sind und befüllt diese durch Anwendung
			des SBI Algorithmus. Welche Lücken und zu welchen Datum befüllt werden
			ist in der Datei data_trans_opt/gap_sbi_fill.csv hinterlegt

         	:start_date: Anfangsdatum ab welchem die Transformation beginnen soll
         	:end_date: Enddatum ab welchem die Transformation beginnen soll
        '''
		# Aktuelle Phase mit bestehenden Daten, die transformiert werden
		this_phase = 'phase4'
		# Die nächste Phase in der die Ergebnisse der Transformation abgespeichert werden
		next_phase = 'phase5'

		print(f'Transormiere {this_phase} in {next_phase}')

		# Pfad zur Dateien der Phasen aufbauen
		csv_path_this_phase = self.csv_data_path_base + this_phase
		csv_path_next_phase = self.csv_data_path_base + next_phase

		# Der gewünschte Datumsbereich, aus welchem die Daten
		# transformiert werden sollen, wird erstellt
		date_range = pd.date_range(start_date, end_date)
		
		# Die kleineren Lücken, die befüllt werden sind in der entsprechenden CSV Datei hinterlegt
		# Die Datei wird hier geladen.
		csv_data_sbi_settings_path = f'{self.csv_trans_opt_path_base}/gap_sbi_fill.csv'
		df_sbi_list = pd.read_csv(csv_data_sbi_settings_path, parse_dates=['Date', 'GapStart', 'GapEnd'], index_col=False)

		# Iteriere durch alle Daten des gegebenen Datumbereichs
		for current_date in date_range:
    		###TIMESTAMP zu Performancemessung und Ausgabe###
			now = datetime.datetime.now()
			current_time = now.strftime("%H:%M:%S")
			print("START: ", current_time)
			###TIMESTAMP zu Performancemessung und Ausgabe###

			# Erstelle einen String-Zeitstempel aus dem aktuellen Datum durch den wir iterieren
			current_date_str = current_date.strftime('%Y-%m-%d')

			# Baue den Pfad und Filenamen zu der CSV Datei für 
			# das aktuelle Datum in der Iteration
			filename_cur_date = f'{csv_path_this_phase}/{current_date_str}.csv'
			
			print("Bearbeite: ", filename_cur_date)
			# Wenn die Datei für das aktuelle Datum nicht existiert, 
			# dann gebe meldung aus und mache beim nächsten Datum
			# aus dem zu iterierenden Bereich über
			if (not path.exists(filename_cur_date)):
				print('File existiert nicht und keine Transformation wird stattfinden')
				continue

			# Die CSV Datei für das aktuelle Datum existiert, so lade sie in ein Pandas-Frame
			df_current = pd.read_csv(filename_cur_date, parse_dates=['UtcDateTime', 'LocalDateTime'], index_col='LocalDateTime')
			# Normalerweise erstellt Pandas immer ein Index, wenn kein angegeben wurde, also löschen
			# df_current.reset_index(drop=True, inplace=True)

			### *** 3. Daten-Imputation für größere Lücken / Similar Behavior Interval Imputation ***###
			for index, row in df_sbi_list[df_sbi_list['Date'] == current_date].iterrows():
				df_current = self.__impute_sbi(df_current, csv_path_this_phase, row['GapStart'], row['GapEnd'], row['CompareStart'], row['CompareEnd'])

			df_current.reset_index(inplace=True)
			df_current.to_csv(f'{csv_path_next_phase}/{current_date_str}.csv', index=False)

			###TIMESTAMP zu Performancemessung und Ausgabe###
			now = datetime.datetime.now()
			current_time = now.strftime("%H:%M:%S")
			print("END: ", current_time)
			###TIMESTAMP zu Performancemessung und Ausgabe###
		
		pass

	def transform_phase5_to_phase6(self, start_date, end_date, algo='std'):
		''' Transofmiert die Daten aus Phase 5 in Phase 6. 
			Transofmrationsdetails, der Dokumentation entnehmen.

			Im Allgemeinen wird in diesem Phasenübergang der Occupancy-Estimation-Algorithmus ausgeführt.

			:algo: Algorithmus der Verwendet wird bzw. Parameter die dem Algoritmus übergeben werden sollen.
					Je nach dem welche Abkürzung übergeben wird, wird auch entsprechende alg_param_{algo}.csv
					mit gegebenen Parametern verwendet.
					Mögliche Angaben:
					std = standard
					ext = extended
			:start_date: Anfangsdatum ab welchem die Transformation beginnen soll
			:end_date: Enddatum ab welchem die Transformation beginnen soll
		'''
		# Aktuelle Phase mit bestehenden Daten, die transformiert werden
		this_phase = 'phase5'
		# Die nächste Phase in der die Ergebnisse der Transformation abgespeichert werden
		next_phase = 'phase6'

		print(f'Transormiere {this_phase} in {next_phase}, Algorithmus={algo}')

		# Pfade zu den Phasen aufbauen
		csv_path_this_phase = self.csv_data_path_base + this_phase
		csv_path_next_phase = self.csv_data_path_base + next_phase

		# Der gewünschte Datumsbereich, aus welchem die Daten
		# transformiert werden sollen, wird erstellt
		date_range = pd.date_range(start_date, end_date)

		# Die Parameter für den Occupancy Estimation Algorithmus werden aus der entsprechenden Datei geladen
		# abhängig davon welcher Algorithmus - standard oder extended verwendet werden soll
		df_parameter = pd.read_csv(f'{self.csv_trans_opt_path_base}/alg_param_{algo}.csv', index_col='date')

		# Iteriere durch alle Daten des gegebenen Datumbereichs
		for current_date in date_range:
			###TIMESTAMP zu Performancemessung und Ausgabe###
			now = datetime.datetime.now()
			current_time = now.strftime("%H:%M:%S")
			print("START: ", current_time)
			###TIMESTAMP zu Performancemessung und Ausgabe###

			# Das vorherige Datum, denn für die Bestimmung der Anwesenheiten
			# für den aktuellen Tag, wird der Tag davor benötigt
			prev_date = current_date - datetime.timedelta(days=1)

			# String Timestamps erstellen
			current_date_str = current_date.strftime('%Y-%m-%d')
			prev_date_str = prev_date.strftime('%Y-%m-%d')

			print("Bearbeite: ", current_date_str)

			# Dateinamen für die Datei des aktuellen Tages aus der vorherigen Phase erstellen
			filename_cur_date = f'{csv_path_this_phase}/{current_date_str}.csv'
			# Dateinamen für die Datei des vorherigen Tages aus der nächste Phase erstellen
			filename_prev_date = f'{csv_path_next_phase}/{algo}/{prev_date_str}.csv'
			
			df_current = None
			df_previous = None
			
			# Überprüfen ob die Dateien auch exisiteren
			if (path.exists(filename_cur_date)):
				df_current = pd.read_csv(filename_cur_date, index_col='LocalDateTime', parse_dates=['UtcDateTime', 'UtcDate', 'LocalDateTime', 'LocalDate'])

			if (path.exists(filename_prev_date)):
				df_previous = pd.read_csv(filename_prev_date, index_col='LocalDateTime', parse_dates=['UtcDateTime', 'UtcDate', 'LocalDateTime', 'LocalDate'])

			if (df_current is None):
				print('File existiert nicht und keine Transformation wird stattfinden')
				continue

			#*********************************** 
			# Es ist festzuhalten, das zur Berechnung der Anwesenheit für den aktuellen Tag, zuerst die Datei für den 
			# aktuellen Tag aus der vorherigen Phase benötigt wird. Auf diesen Daten wird dann der Algorithmus ausgeführt.
			# Aber, noch dazu werden die Daten des vorherigen Tages aus der nächsten Phase, d.h. aus der Phase, in der
			# sich die Dateien befinden, auf denen der Algorithmus schon ausgeführt wurde, aber eben der Tag davor.
			# Die Anwesneheit des Tages davor wird dann beispielsweise im Algorithmus zu Bestimmung der Anwesenheit in der Nacht benötigt
			#***********************************

			# Nun werden die Parameter für den Occupancy Estimation Algorithmus zum aktuellen Tag werden geladen und geparst
			df_current_param = df_parameter.loc[current_date_str]
			daypart_durations = [int(item) for item in df_current_param.daypart_durations.split(';')]
			cluster_sizes = [int(item) for item in df_current_param.cluster_sizes.split(';')]

			threshold_time_from = df_current_param.calc_threshold_from
			threshold_time_to = df_current_param.calc_threshold_to

			avg_threshold_factors = [float(item) for item in df_current_param.avg_threshold_factors.split(';')]
			stddev_threshold_factors = [float(item) for item in df_current_param.stddev_threshold_factors.split(';')]
			range_threshold_factors = [float(item) for item in df_current_param.range_threshold_factors.split(';')]

			avg_time_window = int(df_current_param.avg_timewindow)
			stddev_time_window = int(df_current_param.avg_timewindow)
			range_time_window = int(df_current_param.range_timewindow)

			# Der Algorithmus wird mit Parametern intialisiert und für den aktuellen Tag gestartet
			occ_est_algo = OccupancyEstimation(current_date, df_current, prev_date, df_previous)
			df_current = occ_est_algo.run(daypart_durations, cluster_sizes, threshold_time_from, threshold_time_to, avg_threshold_factors, stddev_threshold_factors, range_threshold_factors, avg_time_window, stddev_time_window, range_time_window)
			
			# Lösche die Imputed-Spalten, denn wir brauchen sie nicht mehr für die nächste Phase
			# Diese Spalten dienen in frühereh Phasen nur zu Visualisierung der befüllten Lücken
			for meter in range(1, 10):
				df_current = df_current.drop([f'Imputed_Meter{meter}'], axis=1)

			# Index in Spalte umwandeln und die Ergebnisse des Algorithmus abspeichern
			df_current.reset_index(inplace=True)
			df_current.to_csv(f'{csv_path_next_phase}/{algo}/{current_date_str}.csv', index=False)

			###TIMESTAMP zu Performancemessung und Ausgabe###
			now = datetime.datetime.now()
			current_time = now.strftime("%H:%M:%S")
			print("END: ", current_time)
			###TIMESTAMP zu Performancemessung und Ausgabe###


		pass


	def transform_phase6_to_phase7(self, start_date, end_date, algo='std', sample_size=15):
		''' Transofmiert die Daten aus Phase 5 in Phase 6. 
			Transofmrationsdetails, der Dokumentation entnehmen.

			Im Allgemeinen werden die Daten aus der vorherigen Phase runtergesampelt.

			:algo: Algorithmus der Verwendet wird bzw. Parameter die dem Algoritmus übergeben werden sollen.
					Je nach dem welche Abkürzung übergeben wird, wird auch entsprechende alg_param_{algo}.csv
					mit gegebenen Parametern verwendet.
					Mögliche Angaben:
					std = standard
					ext = extended
			:sample_size: Minuten auf die die Daten runtergesampelt werden sollen
			:start_date: Anfangsdatum ab welchem die Transformation beginnen soll
			:end_date: Enddatum ab welchem die Transformation beginnen soll
		'''

		this_phase = 'phase6'
		next_phase = 'phase7'

		print(f'Transormiere {this_phase} in {next_phase}, Algorithmus={algo}, Sample Size={sample_size}')

		# Pfade zu Phasen werden aufgebaut
		csv_path_this_phase = self.csv_data_path_base + this_phase
		csv_path_next_phase = self.csv_data_path_base + next_phase

		# Der gewünschte Datumsbereich, aus welchem die Daten
		# transformiert werden sollen, wird erstellt
		date_range = pd.date_range(start_date, end_date)

		# Iteriere über die Daten aus dem Datumsbereich
		for current_date in date_range:
			###TIMESTAMP zu Performancemessung und Ausgabe###
			now = datetime.datetime.now()
			current_time = now.strftime("%H:%M:%S")
			print("START: ", current_time)
			###TIMESTAMP zu Performancemessung und Ausgabe###

			
			current_date_str = current_date.strftime('%Y-%m-%d')
			filename_cur_date = f'{csv_path_this_phase}/{algo}/{current_date_str}.csv'
			
			print("Bearbeite: ", filename_cur_date)

			if (not path.exists(filename_cur_date)):
				print('File existiert nicht und keine Transformation wird stattfinden')
				continue
			
			# Daten aus der vorherigen Phase werden geladen
			df_current = pd.read_csv(filename_cur_date, parse_dates=['UtcDateTime', 'UtcDate', 'LocalDateTime', 'LocalDate'], index_col=False)

			# Diese Funktion berechnet die Zeiten auf die runtergesampelt werden soll
			# Wenn sample_size = 15, werden dann Zeiteinträge in 15 Minutentakt erstellt [14:00, 14:15, 14:30 ...]
			func_resample_time = lambda value: datetime.datetime(value.year, value.month, value.day, value.hour, int(value.minute / sample_size) * sample_size)
			# Wenn die Werte auf eine bestimmte sample_size zusammengefasst werden, muss auch definiert werden wann dan für die Samples jetzt eine
			# Anwesenheit gegeben ist. Wir rechnen für den OCCUPIED Flag auch den Mittelwert und falls er größer ist als 0.5, dann ist innerhalb
			# der 15 Minuten Anwesenheit gegeben, sonst nicht.
			func_occupation_mean = lambda occ: 1 if occ >= .5 else 0

			# Berechnet aus der Zeit die Minuten
			func_local_to_timestamp = lambda ts: int(ts.strftime('%H')) * 60 + int(ts.strftime('%M'))

			# Sample Zeiteinträge berechnen
			df_current['LocalDateTime'] = df_current['LocalDateTime'].transform(func_resample_time)
			df_current['UtcDateTime'] = df_current['UtcDateTime'].transform(func_resample_time)

			# Gruppieren und Mittelwerte für Stromverbrauch und Anwesenheit fur die neuen Samples berechnen
			df_current = df_current[['LocalDateTime', # Diese Spalten werden zum MultiIndex durch GroupBy
											'UtcDateTime', # Diese Spalten werden zum MultiIndex durch GroupBy
											'Season', # Diese Spalten werden zum MultiIndex durch GroupBy
											'Weekday', # Diese Spalten werden zum MultiIndex durch GroupBy
											'Meter1', 
											'Meter2', 
											'Meter3', 
											'Meter4', 
											'Meter5', 
											'Meter6', 
											'Meter7', 
											'Meter8', 
											'Meter9', 
											'Total',
											'AVG_PrevMinutes',
											'STDDEV_PrevMinutes',
											'RANGE_PrevMinutes',
											'AVG_Threshold',
											'STDDEV_Threshold',
											'RANGE_Threshold',
											'OCCUPIED'
											]].groupby(['LocalDateTime', 'UtcDateTime', 'Season', 'Weekday']).mean()

			# Wir holen die LocalDateTime Series aus dem MutliIndex
			local_date_time_series = df_current.index.get_level_values(level=0)
			# Wir fügen eine neue Spalte, die die Minuten des Tages enthalten soll und berechnen aus dem 
			# LocalDateTime die Minuten für jeden Eintrag im Frame
			df_current.insert(0, 'MinuteTimeStamp', local_date_time_series.map(func_local_to_timestamp))
			# Bestimmung der Anwesenheit für die neuen Samples gemäß der angegebene Samplegröße
			df_current['OCCUPIED'] = df_current['OCCUPIED'].transform(func_occupation_mean)


			# Den Multiindex in Spalten umwandeln und abspeichern
			df_current.reset_index(inplace=True)
			df_current.to_csv(f'{csv_path_next_phase}/{algo}/{current_date_str}.csv', index=False)

			###TIMESTAMP zu Performancemessung und Ausgabe###
			now = datetime.datetime.now()
			current_time = now.strftime("%H:%M:%S")
			print("END: ", current_time)
			###TIMESTAMP zu Performancemessung und Ausgabe###



	def detect_gaps(self, start_date, end_date):
		''' 
			Ermittelt fehlende Einträge bzw. Einträge, die zu bestimmten Zeiten ganz fehlen
			und speichert sie in die [dataset]gap.csv Datei.
			Der Zieldatensatz sind die CSV Dateien aus der Phase 2
			
			:start_date: Anfangsdatum ab welchem die Transformation beginnen soll
			:end_date: Enddatum ab welchem die Transformation beginnen soll
		'''
		# Zielphase
		this_phase = 'phase2'

		# Pfad zur Zielphase aufbauen
		csv_path_this_phase = self.csv_data_path_base + this_phase

		# Datumbereich erstellen
		date_range = pd.date_range(start_date, end_date)

		# Die Lückenerkennung wird auf gesamten Datensatz durchgeführt.
		# Das heißt, die Daten aus einzelnen CSV Dateien werden am Ende 
		# in diesen einen Frame zusammengefasst und die Analyse wird
		#  auf diesem Frame dann auch ausgeführt
		df_current_all = DataFrame()

		# Iteriere durch alle Daten des gegebenen Datumbereichs
		for current_date in date_range:
			# Erstelle einen String-Zeitstempel aus dem aktuellen Datum durch den wir iterieren
			current_date_str = current_date.strftime('%Y-%m-%d')

			# Baue den Pfad und Filenamen zu der CSV Datei für 
			# das aktuelle Datum in der Iteration
			filename_cur_date = f'{csv_path_this_phase}/{current_date_str}.csv'
			
			print("Bearbeite: ", filename_cur_date)
			# Wenn die Datei für das aktuelle Datum nicht existiert, 
			# dann gebe meldung aus und mache beim nächsten Datum
			# aus dem zu iterierenden Bereich über
			if (not path.exists(filename_cur_date)):
				print('File existiert nicht und keine Transformation wird stattfinden')
				continue
			
			# Die CSV Datei für das aktuelle Datum existiert, so lade sie in ein Pandas-Frame
			df_current = pd.read_csv(filename_cur_date, parse_dates=['UtcDateTime', 'UtcDate', 'LocalDateTime', 'LocalDate'], index_col=False)
			# Normalerweise erstellt Pandas immer ein Index, wenn kein angegeben wurde, also löschen
			#df_current.reset_index(drop=True, inplace=True)

			# Fasse die einzelnen Frames zu einem Frame zusammen
			df_current_all = df_current_all.append(df_current, ignore_index=True)
		
		# Berecnet die Differenz zwischen zwei aufeinanderfolgenden Einträgen
		# Also normalerweise ist es immer 1 Minute, da die Daten in Phase2
		# auf Minutentakt downgesampelt wurden. 
		# In deltas wird auch die Zeilennumer gespeichert
		deltas = df_current_all['LocalDateTime'].diff()[1:]

		print(deltas)

		# In das Frame gaps, werden entsprechend die Delta-Einträge abgespeichert
		# bei denen eine Lücke die länger als 1 Minute dauert, entdeckt wird
		# Gaps enthält die Zeilennummern von df_current_all bei denen eine Lücke festgestellt wurde
		# Die Zeilennummern entsprechen aber dem anderen Ende der Lücke z.B. wenn wir folgendes haben
		# ZN   Datum
		# 4001 2014-05-01 10:00
		# 4002 2014-06-01 15:00
		# Hier ist eine Lücke von einem Tag und 5 Stunden, dann wird in gaps gespeichert
		# 4002 '1 day 5 hours'
		gaps = deltas[deltas > datetime.timedelta(minutes=1)]

		print(gaps)

		# Die berechneten Lücken samt der Daten und Dauer werden in diesem Frame abgespeichert
		df_gaps = DataFrame(columns=['GapStart', 'GapEnd', 'DurationMinutes'])

		for i, g in gaps.iteritems():
			# Wir wollen die Dauer der Lücke in Minuten darstellen
			duration_minutes = int(g.total_seconds() / 60)
			#Siehe oben, wenn in gaps Zeilennummer = 4002 abgespeichert ist, um den Anfangsdatum aus der Tabelle zu bekommen ziehen wir 1 von der ID ab und kommen auf 4001
			gap_start = df_current_all.iloc[i - 1]['LocalDateTime'] 
			# Siehe oben, die total_seconds des Enddatum
			gap_end = df_current_all.iloc[i]['LocalDateTime'] 
			# Eine neue Zeile für das Frame df_gaps
			df_row = {'GapStart': datetime.datetime.strftime(gap_start, "%Y-%m-%d %H:%M"), 'GapEnd': datetime.datetime.strftime(gap_end, "%Y-%m-%d %H:%M"), 'DurationMinutes': str(duration_minutes) }
			# Hinzufügen der Zeile
			df_gaps = df_gaps.append(df_row, ignore_index=True)


		# Abspeichern des Frames, in dem die gefundenen Lücken entdeckt wurden
		df_gaps.to_csv(f'data_gaps/{self.dataset}/{self.dataset}_gaps.csv', index=False)

	pass

	def create_clean_alg_param_file(self, from_phase='phase5', algo='std'):
		csv_param_path = f'{self.csv_trans_opt_path_base}/alg_param_{algo}.csv'	
		csv_path_this_phase = self.csv_data_path_base + from_phase
		csv_data_path = f'{csv_path_this_phase}/*.csv'
		# Lade alle CSV Dateien mit dem Pfad
		df_params = DataFrame(columns=['date', 'weekday', 'daypart_durations', 'cluster_sizes', 'avg_timewindow', 'stddev_timewindow', 'range_timewindow', 'calc_threshold_from', 'calc_threshold_to', 'avg_threshold_factors', 'stddev_threshold_factors', 'range_threshold_factors'])
		csv_files = glob.glob(csv_data_path)
		for file in csv_files:
			date_str = file.replace('.csv', '').split('\\')[1]
			date = pd.to_datetime(date_str, format='%Y-%m-%d', errors='ignore')
			df_row = {'date': date, 'weekday': date.weekday(), 'daypart_durations': '450;90;450;330;120', 'cluster_sizes': '10;10;10;10;10', 'avg_timewindow': '15', 'stddev_timewindow': '15', 'range_timewindow': '15', 'calc_threshold_from': '01:00', 'calc_threshold_to': '05:30', 'avg_threshold_factors': '1;1;1;1;1', 'stddev_threshold_factors': '1;1;1;1;1', 'range_threshold_factors': '1;1;1;1;1'}
			df_params = df_params.append(df_row, ignore_index=True)
		
		df_params.set_index('date', inplace=True)
		df_params.to_csv(csv_param_path, index=True)


	pass

	def __impute_simple(self, df_current, gap_start, gap_end, gap_start_delta, gap_end_delta, *meter_imp_methods):

		gap_start = gap_start + datetime.timedelta(minutes=1)
		gap_end = gap_end - datetime.timedelta(minutes=1)
		gap_time_range = pd.date_range(gap_start, gap_end, freq="1min", tz="Europe/Vienna")
		around_gap_start = gap_start - datetime.timedelta(minutes=gap_start_delta)
		around_gap_end = gap_end + datetime.timedelta(minutes=gap_end_delta)
		df_current_around_gap = df_current[(df_current.index >= around_gap_start) & (df_current.index < around_gap_end)].copy()

		for gap_datetime in gap_time_range:
			# Daten für das DataFrame vorbereiten
			utc_conv = gap_datetime.astimezone(pytz.UTC) #die lokale Zeit wird in UTC konvertiert
			utc_date_time = datetime.datetime(utc_conv.year, utc_conv.month, utc_conv.day, utc_conv.hour, utc_conv.minute)
			utc_date = datetime.date(utc_date_time.year, utc_date_time.month, utc_date_time.day)
			utc_time = datetime.time(utc_date_time.hour, utc_date_time.minute)
			local_date_time = datetime.datetime(gap_datetime.year, gap_datetime.month, gap_datetime.day, gap_datetime.hour, gap_datetime.minute)
			local_date = datetime.date(gap_datetime.year, gap_datetime.month, gap_datetime.day)
			local_time = datetime.time(gap_datetime.hour, gap_datetime.minute)
			season = dh.get_season(local_date)
			weekday = local_date_time.weekday()

			# Hiermit wird einer neuer Eintrag/Zeile mit local_date_time Index und ensprechenden Werten hinzugefügt
            # Die Werte für die einzelnen Smart Meter von 1-9 werden auf NONE, also LEER, gesetzt, weil der Pandas-Interpolation
            # Algorithmus diese dann später automatisch mit gewählter methode befüllen kann
																					 		  #M1   M2    M3    M4    M5    M6    M7    M8    M9    Total IM1   IM2   IM3   IM4   IM5   IM6   IM7   IM8   IM9   IsImp  IsGap
			row_values = [local_date, local_time, utc_date_time, utc_date, utc_time, season, weekday, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, True, True]
                                                                                                         
			df_current_around_gap.loc[local_date_time] = row_values
			# df_current_around_gap.loc[local_date_time] = { 'UtcDateTime': utc_date_time, 'UtcDate': utc_date, 'UtcTime': utc_time, 'LocalDate': local_date, 'LocalTime': local_time, 'IsGap': True }
		
		# Die eingefügten Zeilen werden an das Ende gestellt.
		# Deshalb sortieren wir hier das Frame neu nach dem Index, also 'LocalDateTime'
		df_current_around_gap = df_current_around_gap.sort_index()

		# Es werden für jeden Smart Meter neue Spalten mit dem Prefix 'imputed_' hinzugefügt, wobei sie dann mit den
		# durch verschiedene Interpolationsmethoden berechneten Wert befüllt werden.
		# Für jeden Smart Meter können hier entsprechende Zeilen kommentiert oder dekommentiert weren.
		# Es bieten sich drei Möglichkeiten:
		# 1. Eine Interpolationsmethode auswählen
		# 2. Forward oder Backward-Fill (LOCF, BACF)
		# 3. Mean oder Median Rolling

	

		# Oben haben wir die neuen Einträge hinzugefügt und mit IsGap=True gekennzeichnet
		# Jetzt iterieren wir durch diese Einträge und befüllen die Meter-Daten entsprechend
		# den Einstellungen in der Settings-Datei. Dabei iterieren wir aber nur über die
		# Lücken wo noch keine Imputation durchgeführt wurde. Das hilft, wenn wir
		# am selben Tag an mehreren Stellen Lücken haben
		#for index, row in df_current[df_current['IsGap'] == True & df_current['IsImputed'] == False]:

		for index, meter_method in enumerate(meter_imp_methods):
			meter_num = index + 1
			if (meter_method.startswith('ip')):
				method = meter_method.split('/')[1]
				df_current_around_gap[f'Imputed_Meter{meter_num}']=df_current_around_gap[f'Meter{meter_num}'].astype('float64').interpolate(method=method)
				df_current_around_gap[f'Meter{meter_num}']=df_current_around_gap[f'Meter{meter_num}'].astype('float64').interpolate(method=method)
				
			elif (meter_method.startswith('mean')):
				window_size = int(meter_method.split('/')[1])
				df_current_around_gap[f'Imputed_Meter{meter_num}']=df_current_around_gap[f'Meter{meter_num}'].rolling(window_size,min_periods=1,).mean()
				df_current_around_gap[f'Meter{meter_num}']=df_current_around_gap[f'Meter{meter_num}'].rolling(window_size,min_periods=1,).mean()
			
			elif (meter_method.startswith('median')):
				window_size = int(meter_method.split('/')[1])
				df_current_around_gap[f'Imputed_Meter{meter_num}']=df_current_around_gap[f'Meter{meter_num}'].rolling(window_size,min_periods=1,).median()
				df_current_around_gap[f'Meter{meter_num}']=df_current_around_gap[f'Meter{meter_num}'].rolling(window_size,min_periods=1,).median()
			else:
				df_current_around_gap[f'Imputed_Meter{meter_num}']=df_current_around_gap[f'Meter{meter_num}'].fillna(method='ffill')
				df_current_around_gap[f'Meter{meter_num}']=df_current_around_gap[f'Meter{meter_num}'].fillna(method='ffill')
		
		df_filled_gap = df_current_around_gap[df_current_around_gap['IsGap'] == True]

		df_filled_gap['Total'] = df_filled_gap[['Meter1', 'Meter2', 'Meter3', 'Meter4', 'Meter5', 'Meter6', 'Meter7', 'Meter8', 'Meter9']].sum(axis=1, skipna=True)

		df_current = df_current.append(df_filled_gap)
		df_current.sort_index(inplace=True)

		return df_current
	pass

	def __impute_sbi(self, df_current, cur_phase_path, gap_start, gap_end, comparision_interval_start, comparision_interval_end, days_lookback=60, days_lookforward=60, take_min_observations=5):
		'''
			Fügt fehlende Einträge (Zeitmessungen) anch dem Ähnlichkeitsprinzip einer bestehenden Zeitreihe.

			:range_start: Anfang des Zeitintervalls innerhalb welchem sich die Datenlücke befinden
			:range_start: Ende des Zeitintervalls innerhalb welchem sich die Datenlücke befinden
			:comp_interval_start: Anfang des Zeitintervalls, der mit Anderen auf Ähnlichkeit vergichen wird
			:comp_interval_end: Ende des Zeitintervalls, der mit Anderen auf Ähnlichkeit vergichen wird
			:gap_start: Anfang der Zeitlücke, die befüllt werden soll
			:gap_end: Ende der Zeitlücke, die befüllt werden soll
			:days_lookback: Tage rückwärts, die mit dem aktuellen Zeitintervall verglichen werden sollen
			:days_lookforward: Tage vorwärts, die mit dem aktuellen Zeitintervall verglichen werden sollen
			:take_min_observations: Anzahl der ähnlichsten Tage (Zeitinteralle) die für die Berechnung der Lücke herangezogen werden sollen
		'''
		# Kopie erstellen und mit der Kopie weiterarbeiten
		df_current = df_current.copy()

		# Holen der Zeitreihe, die mit anderen verglichen werden soll. 
		# Hier ist meist die Zeitreihe die vor oder nach einer Lücke steht gedacht.
		ts_compare_interval_start = comparision_interval_start.strftime('%H:%M')
		ts_compare_interval_end = comparision_interval_end.strftime('%H:%M')
		df_target_interval = df_current[(df_current['LocalTime'] >= ts_compare_interval_start) & (df_current['LocalTime'] <= ts_compare_interval_end)]

		# In die Dictionary werden die kleinsten Zeitintervalle für einzelne Meter abgespeichert
		meter_nearest_mean_dic = dict()

		# Iteriere durch die 9 Smart Meter und berechne die Zeitreihen einzeln für jeden Meter
		# Iteriere durch die 9 Smart Meter, denn die Berechnung der Zeitreihen findet einzeln statt			
		for meter in range(1, 10):

			fill_column_meter = f'Meter{meter}' # Name des Smart Meters
			euclid_distances = [] # Die euklidischen Distanzen

			# Iterire die Tage rückwärts vor dem Zielzeitintervall über die definierte Anzahl an Tagen
			for day in range(1, days_lookback):
				# Wir subtrahieren die Tage vom aktuellen Anfang und Ende der Zeitreihe
				# Das heißt, wir definieren den Anfang und Ende der Zeitabschnitte des "-day" Tages vor dem aktuellen
				# damit wir später die Distanz berechnen können 
				prev_compare_start = comparision_interval_start - datetime.timedelta(days=day)
				prev_compare_end =  comparision_interval_end - datetime.timedelta(days=day)

				# Nur Wochenenden oder Werktage werden miteinander verglichen
				comparable = (dh.is_weekend(prev_compare_start) and dh.is_weekend(comparision_interval_start)) or (dh.is_workday(prev_compare_start) and dh.is_workday(comparision_interval_start))
				
				# Falls der aktuelle Tag ein Werktag ist und der zu Vergleichende nicht und umgekehrt, wird er übersprungen
				if (not comparable):
					continue

				# Hole den zu vergleichenden Zeitintervall vom anderen (vorigen Tag)
				ts_date_compare = prev_compare_start.strftime('%Y-%m-%d')
				fn_date_compare = f'{cur_phase_path}/{ts_date_compare}.csv'
				# Falls der zu vergleichende Tag nicht existiert, dann überspringe
				if (not path.exists(fn_date_compare)):
					continue
				
				# Lade den zu vergleichenden Tag
				df_prev_compare = pd.read_csv(fn_date_compare, index_col=False)

				# Hole das Intervall zum Vergleichen von dem zu vergleichenden Tag
				df_prev_compare_interval = df_prev_compare[(df_prev_compare['LocalTime'] >= ts_compare_interval_start) & (df_prev_compare['LocalTime'] <= ts_compare_interval_end)]

				# Nehme aus dem Pandas DataFrame das Array mit den Werten des entsprechenden Meters als Array.
				target_values = df_target_interval[fill_column_meter].values # Array der zu vergleichenden Zeitriehe
				compare_values = df_prev_compare_interval[fill_column_meter].values # Array mit den Werten des Tages der auf ähnlichkeit mit der Zielzeitreihe verglichen werden soll

				# Überprüfe ob die Zeitreihen kompatibel sind, bzw. gleiche Anzahl an Werten haben
				# denn es kann sein, dass wir gerade ein Tag erwischt haben, der im Zeitintervall ebenfalls Lücken hat
				if (target_values.shape[0] == compare_values.shape[0]):
					dist = distance.euclidean(target_values, compare_values) # Berchne die euklidische Distanz zwischen den Zeitreihen
					euclid_distances.append((fn_date_compare, datetime.date(prev_compare_start.year, prev_compare_start.month, prev_compare_start.day), dist)) #tuples (Datei, Datum, Zeitreihen Distance)

					   # Iteriere die Tage vorwärts nach dem Zielzeitintervall über die vorgegebene Anzahl der Tage
			for day in range(1, days_lookforward):
				# Wir addieren die Tage vom aktuellen Anfang und Ende der Zeitreihe
				# Das heißt, wir definieren den Anfang und Ende der Zeitabschnitte des "-day" Tages vor dem aktuellen
				# damit wir später die Distanz berechnen können 
				next_compare_start = comparision_interval_start + datetime.timedelta(days=day)
				next_compare_end =  comparision_interval_start + datetime.timedelta(days=day)

				# Nur Wochenenden oder Werktage werden miteinander verglichen
				comparable = (dh.is_weekend(next_compare_start) and dh.is_weekend(comparision_interval_start)) or (dh.is_workday(next_compare_start) and dh.is_workday(comparision_interval_start))
				
				# Falls der aktuelle Tag ein Werktag ist und der zu Vergleichende nicht und umgekehrt, wird er übersprungen
				if (not comparable):
					continue

				# Hole den zu vergleichenden Zeitintervall vom anderen (vorigen Tag)
				ts_date_compare = next_compare_start.strftime('%Y-%m-%d')
				fn_date_compare = f'{cur_phase_path}/{ts_date_compare}.csv'
				# Falls der zu vergleichende Tag nicht existiert, dann überspringe
				if (not path.exists(fn_date_compare)):
					continue
				
				# Lade den zu vergleichenden Tag
				df_next_compare = pd.read_csv(fn_date_compare, index_col=False)

				# Hole das Intervall zum Vergleichen von dem zu vergleichenden Tag
				df_next_compare_interval = df_next_compare[(df_next_compare['LocalTime'] >= ts_compare_interval_start) & (df_next_compare['LocalTime'] <= ts_compare_interval_end)]

				# Nehme aus dem Pandas DataFrame das Array mit den Werten des entsprechenden Meters als Array.
				target_values = df_target_interval[fill_column_meter].values
				compare_values = df_next_compare_interval[fill_column_meter].values

				# Überprüfe ob die Zeitreihen kompatibel sind, bzw. gleiche Anzahl an Werten haben
				# denn es kann sein, dass wir gerade ein Tag erwischt haben, der im Zeitintervall ebenfalls Lücken hat
				if (target_values.shape[0] == compare_values.shape[0]):
					dist = distance.euclidean(target_values, compare_values) # Berchne die euklidische Distanz zwischen den Zeitreihen
					euclid_distances.append((fn_date_compare, datetime.date(next_compare_start.year, next_compare_start.month, next_compare_start.day), dist)) #tuples (Datei, Datum, Zeitreihen Distance)
			
			# Wir haben oben Tage in beiden Richtungen vom aktuellen Zieltag in dem sich das Intervall befindet verglichen und die 
			# Werte im Intervall für den Smart Meter in das Array euclid_distances zusammen mit dem entsprechenden Datum (Tag) abgespeichert. 
			# Nun sortieren wir die Distanzen damit wir dan die im Parameter 'take_min_observations' definierte Anzahl an Tagen  für
			# das generieren der fehlenden Zeitreihe nutzen werden
			print(euclid_distances)
			min_distances = sorted(euclid_distances, key=lambda tup: (tup[2]))

			# In dieses DataFrame werden die Zeitreihen 'take_min_observations' der ähnlichsten Tage gespeichert, die wir später nutzen werden
			# um die fehlende Zeitreihe für den Zieltag zu generieren. 
			nearest_series = DataFrame()

			# Wir nehmen die take_min_observation Tage und nutzen sie um die fehlende Zeitreihe für den Zieltag zu generieren
			for tup in min_distances[0:take_min_observations]:
				# Wir nehmen das Zeitintervall (Zeitreihe) vom ähnlichen Tag, die uns im Zieltag fehlt (eine Lücke besteht)
				# Hier wird der Anfang und Ende vom Zeitintervall definiert
				start_datetime = datetime.datetime(tup[1].year, tup[1].month, tup[1].day, gap_start.hour, gap_start.minute, gap_start.second)
				end_datetime = datetime.datetime(start_datetime.year, start_datetime.month, start_datetime.day, gap_end.hour, gap_end.minute, gap_end.second)
				# Hole noch mal die Datei aus der wir dann das Interval, das uns fehlt 'rausschneiden" werden
				df_full = pd.read_csv(tup[0], index_col=False)
				ts_gap_start_time = gap_start.strftime('%H:%M')
				ts_gap_end_time = gap_end.strftime('%H:%M')
				df_missing_interval = df_full[(df_full['LocalTime'] > ts_gap_start_time) & (df_full['LocalTime'] < ts_gap_end_time) ]
				# Hole die Daten aus der Datenbank und füge sie in das DataFrame
				nearest_series = nearest_series.append(df_missing_interval)

			# Im DataFrame nearest_series befinden sich dann die Werte für den bestimmten Meter für alle take_min_observations ähnlichen Tage
			# Hier werden sie dann nach den Minuten gruppiert und es wird der Mittelwert für jede Minute berechnet und so ein neuer Wert
			# für den Minutenzeitpunkt generiert, den wir dann für unsere Lücke im Zieltag verwenden werden
			nearest_mean_per_minute = nearest_series[['LocalTime', fill_column_meter]].groupby('LocalTime').mean()

			# Wir speichern die ganze Zeitrahe mit den neuen Werten als Dictionary mit dem dazugehörigen Meter ab
			meter_nearest_mean_dic.update({fill_column_meter: nearest_mean_per_minute})


		# Für den Zieltag holen wir dann einen bestimmten Zeitbereich um ihn zu befüllen
		# und eventuell wieder anzeigen zu können
		# result_around_gap = self.fetch_aggr_bymin(range_start, range_end)
		# result_around_gap_data = result_around_gap.data
		# # Wir fügen einge neue Spalte hinzu, die angibt ob es sich um den fehlenden neuhinzugefügen Eintrag/Zeile handelt
		# result_around_gap_data= result_around_gap_data.assign(Gap=np.full((len(result_around_gap_data.index), ), False))

		# # Wir generieren die Zeiten pro Minute für das fehlende Zeitintervall
		gap_time_range = pd.date_range(gap_start + datetime.timedelta(minutes=1), gap_end - datetime.timedelta(minutes=1), freq="1min", tz="Europe/Vienna")

		        # Iterieren durch die Zeiten
		for gap_datetime in gap_time_range:
			# Initialisierung der Basiswerte des neuen Eintrags, der hinzugefügt werden wird
			# Daten für das DataFrame vorbereiten
			utc_conv = gap_datetime.astimezone(pytz.UTC) #die lokale Zeit wird in UTC konvertiert
			utc_date_time = datetime.datetime(utc_conv.year, utc_conv.month, utc_conv.day, utc_conv.hour, utc_conv.minute)
			utc_date = datetime.date(utc_date_time.year, utc_date_time.month, utc_date_time.day)
			utc_time = datetime.time(utc_date_time.hour, utc_date_time.minute)
			local_date_time = datetime.datetime(gap_datetime.year, gap_datetime.month, gap_datetime.day, gap_datetime.hour, gap_datetime.minute)
			local_date = datetime.date(gap_datetime.year, gap_datetime.month, gap_datetime.day)
			local_time = datetime.time(gap_datetime.hour, gap_datetime.minute)
			season = dh.get_season(local_date)
			weekday = local_date_time.weekday()

			ts_local_time = local_time.strftime('%H:%M')
			# Da wir in die Dictionary für die enzelnen Meter die generierten Werte abgespeichert haben
			# holen wir sie hier wieder aus der Dictionary mit dem Meter-Schlüssel
			# Anschließend verwenden wir den LocalTime Index um den Einzelwert zu der bestimmten Zeit
			# zu holen und speichern ihn dann in die meter Variablen 
			df_meter1 = meter_nearest_mean_dic['Meter1']
			meter1 = df_meter1.loc[ts_local_time].Meter1
			df_meter2 = meter_nearest_mean_dic['Meter2']
			meter2 = df_meter2.loc[ts_local_time].Meter2
			df_meter3 = meter_nearest_mean_dic['Meter3']
			meter3 = df_meter3.loc[ts_local_time].Meter3
			df_meter4 = meter_nearest_mean_dic['Meter4']
			meter4 = df_meter4.loc[ts_local_time].Meter4
			df_meter5 = meter_nearest_mean_dic['Meter5']
			meter5 = df_meter5.loc[ts_local_time].Meter5
			df_meter6 = meter_nearest_mean_dic['Meter6']
			meter6 = df_meter6.loc[ts_local_time].Meter6
			df_meter7 = meter_nearest_mean_dic['Meter7']
			meter7 = df_meter7.loc[ts_local_time].Meter7
			df_meter8 = meter_nearest_mean_dic['Meter8']
			meter8 = df_meter8.loc[ts_local_time].Meter8
			df_meter9 = meter_nearest_mean_dic['Meter9']
			meter9 = df_meter9.loc[ts_local_time].Meter9
			total = meter1 + meter2 + meter3 + meter4 + meter5 + meter6 + meter7 + meter8 + meter9

			# Die neue Zeile mit den neu generierten Werten wird in das ursprüngliche DataFrame eingefügt
			new_row = [local_date, 
						local_time, 
						utc_date_time, 
						utc_date, 
						utc_time,
						season, 
						weekday, 
						meter1, meter2, meter3, meter4, meter5, meter6, meter7, meter8, meter9, total, 
						meter1, meter2, meter3, meter4, meter5, meter6, meter7, meter8, meter9, True, True]
			df_current.loc[local_date_time] = new_row

		# Sortieren nach dem LocalDateTime Index, da die neuen Zeilen am Ende hinzugefügt wurden
		df_current = df_current.sort_index()
		
		return df_current