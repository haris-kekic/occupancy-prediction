# -----------------------------------------------------------
# Sktript zur Datentransformation
#
# (C) 2020 Haris Kekic, Fernuniversität in Hagen
# -----------------------------------------------------------

import numpy as np 
from transformer import Transformer
import datetime
import pandas as pd

def transform_dataset(dataset, start_date, end_date, detect_gaps=True, to_phase='all'):
	''' Fast die Datentransformation zusammen und ermöglicht die Transformation
		sowohl für einzelne Phasen als auch für alle Phasen in einem Durchlauf

		:start_date: Das Anfangsdatum, ab welchem die Datentransformation ausgeführt werden soll
		:end_date: Das Enddatum, bis zu welchem die Datentransformation ausgeführt werden soll
		:detect_gaps: Angabe, ob im Zuge der Transformation auch die 'buildingN_gaps.csv' Datei, 
						die die Lücken in den Daten aufzeigt, erstellt werden soll
		:to_phase: Gibt die Zielphase an, in die Transormation durchgeführt werden soll von (1-7)
				oder führt alle Phasen im gegebenen Zeitraum aus 'all'
	'''
	transformer = Transformer(dataset)

	if (to_phase == 'all' or to_phase == 'phase1'):
		transformer.transform_phase0_to_phase1(start_date, end_date)

	if (to_phase == 'all' or to_phase == 'phase2'):
		transformer.transform_phase1_to_phase2(start_date, end_date, detect_gaps)

	if (to_phase == 'all' or to_phase == 'phase3'):
		transformer.transform_phase2_to_phase3(start_date, end_date)

	if (to_phase == 'all' or to_phase == 'phase4'):
		transformer.transform_phase3_to_phase4(start_date, end_date)

	if (to_phase == 'all' or to_phase == 'phase5'):
		transformer.transform_phase4_to_phase5(start_date, end_date)

	if (to_phase == 'all' or to_phase == 'phase6'):
		transformer.transform_phase5_to_phase6(start_date, end_date, algo='ext')
		transformer.transform_phase5_to_phase6(start_date, end_date, algo='std')

	if (to_phase == 'all' or to_phase == 'phase7'):
		transformer.transform_phase6_to_phase7(start_date, end_date, algo='ext')
		transformer.transform_phase6_to_phase7(start_date, end_date, algo='std')

def create_clean_algo_param_file(dataset, algo='std'):
	transformer = Transformer(dataset)
	transformer.create_clean_alg_param_file(algo=algo)

# Starte die Transformation für alle Phasen stufenweise
#transform_dataset('building4', datetime.datetime(2014, 1, 21), datetime.datetime(2014, 11, 6), False, 'all')

# Einzelne Phasen ausführen...
# transform_dataset('building4', datetime.datetime(2014, 1, 21), datetime.datetime(2014, 11, 6), False, 'phase6')
# transform_dataset('building4', datetime.datetime(2014, 1, 21), datetime.datetime(2014, 11, 6), False, 'phase7')

#transform_dataset('building1', datetime.datetime(2014, 3, 12), datetime.datetime(2015, 6, 29), False, 'phase6')

#transform_dataset('building1', datetime.datetime(2014, 4, 4), datetime.datetime(2014, 4, 4), False, 'phase4')

#create_clean_algo_param_file('building1', algo='ext')

# transform_dataset('building5', datetime.datetime(2014, 3, 7), datetime.datetime(2015, 3, 21), True, 'phase1')
# transform_dataset('building5', datetime.datetime(2014, 3, 7), datetime.datetime(2015, 3, 21), True, 'phase2')

# transform_dataset('building6', datetime.datetime(2014, 9, 2), datetime.datetime(2015, 3, 16), True, 'phase1')
# transform_dataset('building6', datetime.datetime(2014, 2, 5), datetime.datetime(2015, 3, 16), True, 'phase2')

# transform_dataset('building7', datetime.datetime(2014, 4, 24), datetime.datetime(2015, 3, 26), True, 'phase1')
# transform_dataset('building7', datetime.datetime(2014, 4, 24), datetime.datetime(2015, 3, 26), True, 'phase2')

# transform_dataset('building4', datetime.datetime(2014, 2, 26), datetime.datetime(2014, 2, 26), True, 'phase5')
# transform_dataset('building4', datetime.datetime(2014, 2, 26), datetime.datetime(2014, 2, 26), True, 'phase6')
# transform_dataset('building4', datetime.datetime(2014, 2, 26), datetime.datetime(2014, 2, 26), True, 'phase7')

# transform_dataset('building4', datetime.datetime(2014, 2, 1), datetime.datetime(2014, 10, 31), True, 'phase7')

# transform_dataset('building1', datetime.datetime(2014, 3, 13), datetime.datetime(2015, 6, 29), True)

#transform_dataset('building0', datetime.datetime(2014, 6, 18), datetime.datetime(2014, 10, 12), True)

# transform_dataset('building2', datetime.datetime(2014, 11, 29), datetime.datetime(2014, 11, 29), False, 'phase3')
# transform_dataset('building2', datetime.datetime(2014, 11, 29), datetime.datetime(2014, 11, 29), False, 'phase4')


# transform_dataset('building2', datetime.datetime(2014, 12, 16), datetime.datetime(2014, 12, 16), False, 'phase3')
# transform_dataset('building2', datetime.datetime(2014, 12, 16), datetime.datetime(2014, 12, 16), False, 'phase4')

# transform_dataset('building2', datetime.datetime(2015, 5, 15), datetime.datetime(2015, 5, 15), False, 'phase3')
# transform_dataset('building2', datetime.datetime(2015, 5, 15), datetime.datetime(2015, 5, 15), False, 'phase4')

# transform_dataset('building2', datetime.datetime(2015, 5, 22), datetime.datetime(2015, 5, 22), False, 'phase3')
# transform_dataset('building2', datetime.datetime(2015, 5, 22), datetime.datetime(2015, 5, 22), False, 'phase4')

# transform_dataset('building2', datetime.datetime(2015, 6, 9), datetime.datetime(2015, 6, 9), False, 'phase3')
# transform_dataset('building2', datetime.datetime(2015, 6, 9), datetime.datetime(2015, 6, 9), False, 'phase4')

# transform_dataset('building2', datetime.datetime(2015, 1, 22), datetime.datetime(2015, 1, 22), False, 'phase3')
# transform_dataset('building2', datetime.datetime(2015, 1, 22), datetime.datetime(2015, 1, 22), False, 'phase4')
# transform_dataset('building2', datetime.datetime(2015, 1, 22), datetime.datetime(2015, 1, 22), False, 'phase5')

# transform_dataset('building2', datetime.datetime(2015, 1, 23), datetime.datetime(2015, 1, 23), False, 'phase3')
# transform_dataset('building2', datetime.datetime(2015, 1, 23), datetime.datetime(2015, 1, 23), False, 'phase4')
# transform_dataset('building2', datetime.datetime(2015, 1, 23), datetime.datetime(2015, 1, 23), False, 'phase5')

# transform_dataset('building2', datetime.datetime(2015, 1, 27), datetime.datetime(2015, 1, 27), False, 'phase3')
# transform_dataset('building2', datetime.datetime(2015, 1, 27), datetime.datetime(2015, 1, 27), False, 'phase4')
# transform_dataset('building2', datetime.datetime(2015, 1, 27), datetime.datetime(2015, 1, 27), False, 'phase5')
# transform_dataset('building2', datetime.datetime(2015, 1, 27), datetime.datetime(2015, 1, 27), False, 'phase6')

# transform_dataset('building2', datetime.datetime(2014, 2, 20), datetime.datetime(2015, 6, 29), False, 'phase6')
# transform_dataset('building2', datetime.datetime(2014, 2, 20), datetime.datetime(2015, 6, 29), False, 'phase7')

#transform_dataset('building0', datetime.datetime(2014, 6, 18), datetime.datetime(2014, 10, 12), True)
#transform_dataset('building1', datetime.datetime(2014, 3, 13), datetime.datetime(2015, 6, 29), True)
#transform_dataset('building2', datetime.datetime(2014, 2, 20), datetime.datetime(2015, 6, 29), False)
#transform_dataset('building3', datetime.datetime(2014, 4, 8), datetime.datetime(2015, 3, 13), False)
transform_dataset('building4', datetime.datetime(2014, 2, 1), datetime.datetime(2014, 10, 31), False)
#transform_dataset('building5', datetime.datetime(2014, 3, 8), datetime.datetime(2015, 3, 21), False)
transform_dataset('building7', datetime.datetime(2014, 7, 15), datetime.datetime(2014, 11, 2), False)

# df_building5_gaps = pd.read_csv('data_trans_opt/building5/gap_std_fill.csv', index_col="Date",  parse_dates=['Date'])

# for index, row in df_building5_gaps.iterrows():
# 	transform_dataset('building5', index, index, False, 'phase1')
# 	transform_dataset('building5', index, index, False, 'phase2')
# 	transform_dataset('building5', index, index, False, 'phase3')
# 	transform_dataset('building5', index, index, False, 'phase4')


# df_building7_gaps = pd.read_csv('data_trans_opt/building7/gap_std_fill.csv', index_col="Date",  parse_dates=['Date'])

# for index, row in df_building7_gaps.iterrows():
# 	transform_dataset('building7', index, index, False, 'phase1')
# 	transform_dataset('building7', index, index, False, 'phase2')
# 	transform_dataset('building7', index, index, False, 'phase3')
# 	transform_dataset('building7', index, index, False, 'phase4')
		
		
		
		
		
		
		
		




