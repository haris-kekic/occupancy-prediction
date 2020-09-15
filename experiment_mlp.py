# -----------------------------------------------------------
# Skript mit Funktionen zum Erstellen von Experimenten
# und Ausführen dieser Experimente parallel oder sequentiell.
# 
# (C) 2020 Haris Kekic, Fernuniversität in Hagen
# -----------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_prepare as dp
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from pandas import DataFrame
from pandas import Series
from model_mlp import MHMLP
from model_validator import ModelValidator
import itertools
import os
import pathlib
from visualizer import Visualizer
import multiprocessing
import glob 
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
# Oben die Einstellungen um alle GPUs zu nutzen

# Basispfad mit Experimenten und Ergebnissen
mlp_artifacts_path = 'experiments/mlp'


def generate_experiment_hyperparams(num_of_files=4):
    ''' Erstellt alle angegebenen Kombinationen von Parametern zum Experimentieren

        :num_of_files: Anzahl der Dateien auf die die Experimente gesplittet werden sollen
    '''


    DATASETS = [['building1'], ['building0', 'building1', 'building2', 'building3', 'building4', 'building5'] ] # Die Datensätze für Gebäude die Kombiniert werden sollen
    ALGORITHMS = ['std'] # Algorithmen
    FEAT_COMB = [['Total'], 
                            ['MinuteTimeStamp', 'Total'], 
                            ['Weekday', 'Total'],
                            ['Season', 'Total'],
                            ['Weekday', 'Season', 'Total'],
                            ['MinuteTimeStamp', 'Weekday', 'Total'],
                            ['MinuteTimeStamp', 'Season', 'Total'],  
                            ['MinuteTimeStamp', 'Weekday', 'Season', 'Total'],
                            ['Meter1', 'Meter2', 'Meter3', 'Meter4', 'Meter5', 'Meter6', 'Meter7', 'Meter8', 'Meter9'],
                            ['MinuteTimeStamp', 'Meter1', 'Meter2', 'Meter3', 'Meter4', 'Meter5', 'Meter6', 'Meter7', 'Meter8', 'Meter9'],
                            ['Weekday', 'Meter1', 'Meter2', 'Meter3', 'Meter4', 'Meter5', 'Meter6', 'Meter7', 'Meter8', 'Meter9'],
                            ['Season', 'Meter1', 'Meter2', 'Meter3', 'Meter4', 'Meter5', 'Meter6', 'Meter7', 'Meter8', 'Meter9'],
                            ['Weekday', 'Season', 'Meter1', 'Meter2', 'Meter3', 'Meter4', 'Meter5', 'Meter6', 'Meter7', 'Meter8', 'Meter9'],
                            ['MinuteTimeStamp', 'Weekday', 'Meter1', 'Meter2', 'Meter3', 'Meter4', 'Meter5', 'Meter6', 'Meter7', 'Meter8', 'Meter9'],
                            ['MinuteTimeStamp', 'Season', 'Meter1', 'Meter2', 'Meter3', 'Meter4', 'Meter5', 'Meter6', 'Meter7', 'Meter8', 'Meter9'],
                            ['MinuteTimeStamp', 'Weekday', 'Season', 'Meter1', 'Meter2', 'Meter3', 'Meter4', 'Meter5', 'Meter6', 'Meter7', 'Meter8', 'Meter9']]

    TARGET_LABELS = ['OCCUPIED']

    HIDDEN_LAYER_NEURONS = [['25'], ['500'], ['100', '50', '25']]

    # Minuten: 3std, 12std 24std 7Tage
    TIMESTEPS = [30, 96, 288, 672]

    # Minuten: 1 std, 6Std, 12Std, 24Std, 7Tage
    FORECASTS = [5, 30, 96, 288]

    SPLITS = 4
    EPOCHS = 120

    # Der endgültige DataFrame in dem alle Parameterkombinationen gespeichert werden
    df_experiments = DataFrame()
    exp = 0
    # Alle Kombinationen von den oben angegebenen Parametern werden 
    # in den Schleifen unten zusammengestellt
    for ds in DATASETS:
        ALGORITHMS = ['std']
        if len(ds) == 1:
            if (ds[0] == 'building4' or ds[0] == 'building1'):
                ALGORITHMS = ['std', 'ext']
                
        for alg in ALGORITHMS:
            for feat_comb in FEAT_COMB:
                feat_comb = feat_comb.copy()
                feat_comb.extend(TARGET_LABELS)
                for time_step in TIMESTEPS:

                    for forecast in FORECASTS:

                        for hidden_neuron in HIDDEN_LAYER_NEURONS:

                            regular_dropout_comb = list(itertools.product([False, True], repeat=2)) # itertools erstellt Kombinationen

                            for reg_comb in regular_dropout_comb:
                                print(f'Experiment {exp} wird hinzugefügt...')
                                reg, dropout = reg_comb
                                df_experiment_row = Series({'Dataset': ';'.join(ds), 
                                                            'Algo': alg, 
                                                            'FeaturesLabels': ';'.join(feat_comb), 
                                                            'TimeSteps': time_step, 
                                                            'ForecastSteps': forecast, 
                                                            'HiddenLayerNeurons': ';'.join(hidden_neuron), 
                                                            'UseRegularization': reg, 
                                                            'UseDropout': dropout,
                                                            'Splits': SPLITS,
                                                            'Epochs': EPOCHS, 
                                                            'Done': False })
                                df_experiments = df_experiments.append(df_experiment_row, ignore_index=True)
                                exp = exp + 1

    # Unglücklicherweise speichert pandas einen DataFrame mit Integern als Float-Zahlen
    # Um das zu umgehen, müssen wir hier die Typen for dem Speichern genau angeben
    df_experiments = df_experiments.astype({"Done": bool, "TimeSteps": int, "ForecastSteps": int, "UseRegularization": bool, "UseDropout" : bool, "Splits": int, "Epochs": int })

    # Wir möchten auch, dass die Spalte Id der Index wird
    df_experiments.index.name = 'Id'

    # Nehme die Anzahl der Reihen (Experimente) und teile sie 
    # in die übergebene Anzahl der Dateien für Multi-Prozessing
    split_step = int(df_experiments.shape[0] / num_of_files)
    en = 1
    # Wir zählen von 0-Anzahl der Experimente mit berechneten Schritt
    for i in range(0, df_experiments.shape[0], split_step):
        start = i
        end = start + split_step
        df_exp_split = df_experiments.iloc[start:end].copy()
        df_exp_split[['Dataset', 
                    'Algo', 
                    'FeaturesLabels', 
                    'TimeSteps', 
                    'ForecastSteps', 
                    'HiddenLayerNeurons', 
                    'UseRegularization', 
                    'UseDropout', 
                    'Splits',
                    'Epochs',
                    'Done']].to_csv(f'{mlp_artifacts_path}/experiments_part{en}.csv')
        en = en + 1


def run_experiment(file_path):
    ''' Starte die Experimente aus der übergebenen Datei

        :file_path: Pfad zu der Datei mit Experimenten die ausgeführt werden sollen
        :epochs: Anzahl der Epochen die durchgeführt werden beim Durchführen der Experimente
    '''
    print('Experimentdatei: ' + file_path)

    # Lese die Experiment-Datei ein
    df_experiments = pd.read_csv(file_path, index_col='Id')

    # Durchlaufe die einzelnen "Experimente" oder Parameter aus der Experiment Datei
    # die noch nicht durchgeführt worden sind Done == FALSE
    for exp_num, row in df_experiments[df_experiments['Done'] == False].iterrows():
        print(f'Starte Experiment {exp_num}')
        # Lese die Parameter ein und parse sie falls nötig
        datasets = row.Dataset.split(';')
        algo = row.Algo
        feat_labs = row.FeaturesLabels.split(';')
        time_steps = int(row.TimeSteps)
        for_steps = int(row.ForecastSteps)
        hid_neurons = [int(i) for i in row.HiddenLayerNeurons.split(';')]
        use_reg = row.UseRegularization
        use_drop = row.UseDropout
        splits = row.Splits
        epochs = row.Epochs

        # Lese den Datensatz ein 
        # # HIER MUSST DU VIELLEICHT NOCH ERGÄNZEN, FALLS ZWEI DATENSÄTZE MITEINANDER KOMBINIERT WERDEN
        df_datasets = dp.read_datasets(datasets, algo) 
        df_datasets = df_datasets[feat_labs]
        # Erstelle das MH MLP Modell aufgrund der Parameter aus der Experiment-Datei
        model = MHMLP((hid_neurons, use_reg, use_drop))
        # Starte das Training und Evaluierung mit dem Datensatz auf dem erstellten Modell
        validator = ModelValidator('Multiheaded MLP', model, df_datasets)
        df_full_score, df_aggr_score = validator.validate(epochs=epochs, time_steps=time_steps, forcast_steps=for_steps, splits=splits)

        # Nehme die 5 besten LOSS und ACCURACY Ergebnisse, 
        # die dann später in den Results-Ordner gespeichert werden
        best_val_loss = df_aggr_score.nsmallest(5, 'EpochSmoothValLoss')
        best_val_acc = df_aggr_score.nlargest(5, 'EpochSmoothValAcc')

        # Der Pfad zum Results-Ordner, in dem die Ergebnisse 
        # des Trainings und Valideriung, sprich des Experiments, abgespeichert werden
        csv_results_path = f'{mlp_artifacts_path}/results_{exp_num}'

        # Erstelle den Results-Ordner für das Experiment falls es nicht bereits existiert
        pathlib.Path(csv_results_path).mkdir(parents=True, exist_ok=True) 

        # Setze den Index neu für die Dateien der besten Ergebnisse
        best_val_loss.reset_index(inplace=True)
        best_val_acc.reset_index(inplace=True)

        # Abspeichern der Ergebnis-Dateien des Experiments in den Results-Ordner
        best_val_loss.to_csv(f'{csv_results_path}/score_best_loss.csv', index=False)
        best_val_acc.to_csv(f'{csv_results_path}/score_best_acc.csv', index=False)
        df_full_score.to_csv(f'{csv_results_path}/score_full.csv', index=False)
        df_aggr_score.to_csv(f'{csv_results_path}/score_aggragated.csv')
        model.visualize_model(csv_results_path, 'network_model') # Das Netwzerk-Model wird auch abgespeichert
        visualizer = Visualizer(datasets)
        visualizer.visualize_model_scores(df_aggr_score, best_val_loss, exp_num, mlp_artifacts_path) # Hier auch noch die Accuracy und Validierungsdiagramme
        # Parameter mit denen das Experiment durchführt wurde, wird auch dann in den Ordner gespeichert um später einen Lookup zu erleichtern
        df_params = DataFrame(row)
        df_params.to_csv(f'{csv_results_path}/parameters.csv') 
        # In der Experiment-Dateie, setze die Zeile des Experiments auf DONE und speichere die Datei wieder
        # also eigentlich ein Update der Datei
        df_experiments.loc[exp_num, 'Done'] = True
        df_experiments.to_csv(file_path)


def run_experiments_async():
    ''' Starte parallel in mehreren Prozessen die Abarbeitung 
        der in die mehrere Dateien gesplitteten Experimente.
        :epochs: Anzahl der Epochen für die Modelle trainiert werden
    '''
    experiment_files = glob.glob(f"{mlp_artifacts_path}/*.csv")
    processes = list()
    # Durchlaufe alle Experiment-Dateien und starte für jede Datei einen neuen
    # Prozess und bearbeite die Dateien und Experimente in diesen Dateien
    for file in experiment_files:
        process = multiprocessing.Process(target=run_experiment, args=(file,))
        processes.append(process)
        process.start()

    # Beende alle gestarteten Prozesse   
    for proc in processes:
        proc.join()


def compose_all_results():   
    ''' Kann erst nach dem Experiment ausgeführt werden um die besten Ergebnisse
        aus den Result-Ordnern zusammenzufassen und in eine Datei zu speichern.
        Separat für Loss und Accuracy werden zwei Dateien mit den besten Ergebnissen zusammengestellt.
    ''' 
    result_folders = glob.glob(f"{mlp_artifacts_path}/results_*/")
    df_best_losses = DataFrame(columns=["Id", "ResultId", "Dataset", "Algo", "FeaturesLabels", "TimeSteps", "ForecastSteps", "HiddenLayerNeurons", "Epoch", "EpochSmoothValAcc", "EpochSmoothValLoss", "EpochTrainAcc", "EpochTrainLoss", "EpochValAcc", "EpochValLoss"], index=None)
    df_best_accs = DataFrame(columns=["Id", "ResultId", "Dataset", "Algo", "FeaturesLabels", "TimeSteps", "ForecastSteps", "HiddenLayerNeurons", "Epoch", "EpochSmoothValAcc", "EpochSmoothValLoss", "EpochTrainAcc", "EpochTrainLoss", "EpochValAcc", "EpochValLoss"], index=None)

    for result_folder in result_folders:
        file_score_best_loss = result_folder + "score_best_loss.csv"
        file_score_best_acc = result_folder + "score_best_acc.csv"
        file_parameter = result_folder + "parameters.csv"

        df_score_best_loss = pd.read_csv(file_score_best_loss)
        df_score_best_acc = pd.read_csv(file_score_best_acc)
        df_parameter = pd.read_csv(file_parameter)

        id = df_parameter.columns[1]
        dataset = df_parameter.iloc[0, 1]
        algo = df_parameter.iloc[1, 1]
        features = df_parameter.iloc[2, 1]
        timesteps = df_parameter.iloc[3, 1]
        forecaststeps = df_parameter.iloc[4, 1]
        hidden_neurons = df_parameter.iloc[5, 1]

        # Hinzufügen von besten Losses zum Hauptframe
        ids = np.full( shape=df_score_best_loss.shape[0], fill_value=id, dtype=np.int)
        datasets = np.full( shape=df_score_best_loss.shape[0], fill_value=dataset)
        algos = np.full( shape=df_score_best_loss.shape[0], fill_value=algo)
        featuress = np.full( shape=df_score_best_loss.shape[0], fill_value=features.replace(';OCCUPIED', ''))
        timestepss = np.full( shape=df_score_best_loss.shape[0], fill_value=timesteps, dtype=np.int)
        forecaststepss = np.full( shape=df_score_best_loss.shape[0], fill_value=forecaststeps, dtype=np.int)
        hidden_neuronss = np.full( shape=df_score_best_loss.shape[0], fill_value=hidden_neurons)
        df_score_best_loss["ResultId"] = ids
        df_score_best_loss["Dataset"] = datasets
        df_score_best_loss["Algo"] = algos
        df_score_best_loss["FeaturesLabels"] = featuress
        df_score_best_loss["TimeSteps"] = timestepss
        df_score_best_loss["ForecastSteps"] = forecaststepss
        df_score_best_loss["HiddenLayerNeurons"] = hidden_neuronss
        df_best_losses = df_best_losses.append(df_score_best_loss)

        # Hinzufügen von besten Accuracies zum Hauptframe
        ids = np.full( shape=df_score_best_acc.shape[0], fill_value=id, dtype=np.int)
        datasets = np.full( shape=df_score_best_acc.shape[0], fill_value=dataset)
        algos = np.full( shape=df_score_best_acc.shape[0], fill_value=algo)
        featuress = np.full( shape=df_score_best_acc.shape[0], fill_value=features.replace(';OCCUPIED', ''))
        timestepss = np.full( shape=df_score_best_acc.shape[0], fill_value=timesteps, dtype=np.int)
        forecaststepss = np.full( shape=df_score_best_acc.shape[0], fill_value=forecaststeps, dtype=np.int)
        hidden_neuronss = np.full( shape=df_score_best_acc.shape[0], fill_value=hidden_neurons)
        df_score_best_acc["ResultId"] = ids
        df_score_best_acc["Dataset"] = datasets
        df_score_best_acc["Algo"] = algos
        df_score_best_acc["FeaturesLabels"] = featuress
        df_score_best_acc["TimeSteps"] = timestepss
        df_score_best_acc["ForecastSteps"] = forecaststepss
        df_score_best_acc["HiddenLayerNeurons"] = hidden_neuronss

        df_best_accs = df_best_accs.append(df_score_best_acc)

    df_best_losses["Id"] = [x for x in range(1, len(df_best_losses.values)+1)]
    df_best_accs["Id"] = [x for x in range(1, len(df_best_accs.values)+1)]

    df_best_losses.set_index(["Id"], inplace=True)
    df_best_accs.set_index(["Id"], inplace=True)
    
    # Abspeichern in den all Ordner
    df_best_losses.to_csv(f"{mlp_artifacts_path}/all_results/score_best_losses.csv")
    df_best_accs.to_csv(f"{mlp_artifacts_path}/all_results/score_best_accs.csv")
    
def get_all_results():
    ''' Kann erst aufgerufen werden nachdem die besten Ergebnisse in Datei zusammengefasst wurden.
        Holt die zusammengefassten Ergebnisse und erstellt Pandas-Frame für Loss und Accuracy CSV Dateien
        und liefert diese zurück.
    ''' 
    df_best_losses = pd.read_csv(f"{mlp_artifacts_path}/all_results/score_best_losses.csv", index_col="Id")
    df_best_accs = pd.read_csv(f"{mlp_artifacts_path}/all_results/score_best_accs.csv", index_col="Id")
    return df_best_losses, df_best_accs

def arrange_best_scores(title, dataset, algo='std'):
    ''' Sortiert die besten Ergebnisse aller Modelle aus der zusammenfassenden Datei entsprechend der Metrik 
        und speichert diese ab als score_{title}_{algo}_best_losses.csv und score_{title}_{algo}_best_acc.csv
        Für Loss, werden Modelle nach kleinstem Verlust und für Accurady nach der größten Trefferquote sortiert.
    '''
    df_best_losses, df_best_acss = get_all_results()

    df_arranged_losses = df_best_losses[(df_best_losses['Dataset'] == dataset) & (df_best_losses['Algo'] == algo)].sort_values(['EpochSmoothValLoss'], ascending=True)
    df_arranged_accs = df_best_acss[(df_best_acss['Dataset'] == dataset) & (df_best_acss['Algo'] == algo)].sort_values(['EpochSmoothValAcc'], ascending=False)

    df_arranged_losses.to_csv(f"{mlp_artifacts_path}/all_results/score_{title}_{algo}_best_losses.csv")
    df_arranged_accs.to_csv(f"{mlp_artifacts_path}/all_results/score_{title}_{algo}_best_acc.csv")

def evaluate_model(id, dataset_eval, epochs):
    ''' Traininert ein Modell bzw. Parameterkombination mit dem übergebenen Datensatz 
        aus der Experiment-Datei mit der entsprechenden ID und Anzahl an Epochen.

        :id: Model ID aus der Experiment-Datei
        :dataset: Name des Datensatzes mit dem das tra
        :epochs: Anzahl an Epochen mit dem das übergebenen Modell trainiert werden soll vor der Evaluierung
    '''
    experiment_files = glob.glob(f"{mlp_artifacts_path}/experiments*.csv")
    
    for file_path in experiment_files:
        print('Experimentdatei: ' + file_path)

        # Lese die Experiment-Datei ein
        df_experiments = pd.read_csv(file_path, index_col='Id')
        df_row = df_experiments[df_experiments.index == id]
        if df_row.empty:
            continue
        row = df_row.iloc[0]
        # Lese die Parameter ein und parse sie falls nötig
        datasets = row.Dataset.split(';')
        datasets_eval = dataset_eval.split(';')
        algo = row.Algo
        feat_labs = row.FeaturesLabels.split(';')
        time_steps = int(row.TimeSteps)
        for_steps = int(row.ForecastSteps)
        hid_neurons = [int(i) for i in row.HiddenLayerNeurons.split(';')]
        use_reg = row.UseRegularization
        use_drop = row.UseDropout
        splits = row.Splits
        epochs = epochs

        # Lese den Datensatz ein 
        # # HIER MUSST DU VIELLEICHT NOCH ERGÄNZEN, FALLS ZWEI DATENSÄTZE MITEINANDER KOMBINIERT WERDEN
        df_datasets = dp.read_datasets(datasets, algo) 
        df_datasets = df_datasets[feat_labs]

        df_datasets_eval = dp.read_datasets(datasets_eval, algo)
        df_datasets_eval = df_datasets_eval[feat_labs]

        # Erstelle das MH MLP Modell aufgrund der Parameter aus der Experiment-Datei
        model = MHMLP((hid_neurons, use_reg, use_drop))
        validator = ModelValidator('Multiheaded MLP', model, df_datasets, df_datasets_eval)
        test_score = validator.evaluate(epochs=epochs, time_steps=time_steps, forcast_steps=for_steps, splits=splits)

        return test_score

def evaluate_models(title, *eval_items):
    '''Trainiert und Evaluiert eine Menge von Modellen die als Parameter übergeben werden

        :titel: Dieser Titel wird in den Dateinnamen eingebettet
        :eval_items: Liste von Dictionaries mit folgender Signatur {id: XX, dataset: XX, epochs: XX}
    '''

    df_eval_results = DataFrame(columns=['Id', 'Loss', 'Acc'], index=None)

    for eval_item in eval_items:
        id = eval_item['id']
        dataset = eval_item['dataset']
        epochs = eval_item['epochs']
        
        test_score = evaluate_model(id, dataset, epochs)

        df_eval_results = df_eval_results.append({'Id': id, 'Loss': test_score[0], 'Acc': test_score[1]}, ignore_index=True)

    df_eval_results = df_eval_results.astype({'Id': int})
    df_eval_results.set_index('Id', inplace=True)
    df_eval_results.to_csv(f"{mlp_artifacts_path}/all_results/score_{title}_eval.csv")

def evaluate_model_fragment(id, dataset_eval, train_quota, eval_quota, epochs):
    ''' Traininert ein Modell bzw. Parameterkombination mit dem übergebenen Quoten
        aus der Experiment-Datei mit der entsprechenden ID und Anzahl an Epochen.
        Es handelt sich um die gleichen Datensäte für Training und Evaluierung, allerdings
        werden unterschiedliche Fragemente die durch den Prozentsatz in Quoten definiert werden.

        :id: Model ID aus der Experiment-Datei
        :dataset: Datensatz mit dem evaluiert werden soll
        :train_quota: Prozentsatz der besagt, wie viel von den Datensätzen für das Trainineren verwendet werden sollen
        :test_quota: Prozentsatz der besagt, wie viel von den Datensätzen für das Evaluieren verwendet werden sollen
        :epochs: Anzahl an Epochen mit dem das übergebenen Modell trainiert werden soll vor der Evaluierung
    '''

    experiment_files = glob.glob(f"{mlp_artifacts_path}/experiments*.csv")
    
    for file_path in experiment_files:
        print('Experimentdatei: ' + file_path)

        # Lese die Experiment-Datei ein
        df_experiments = pd.read_csv(file_path, index_col='Id')
        df_row = df_experiments[df_experiments.index == id]
        if df_row.empty:
            continue
        row = df_row.iloc[0]
        # Lese die Parameter ein und parse sie falls nötig
        datasets = row.Dataset.split(';')
        datasets_eval = dataset_eval.split(';')
        algo = row.Algo
        feat_labs = row.FeaturesLabels.split(';')
        time_steps = int(row.TimeSteps)
        for_steps = int(row.ForecastSteps)
        hid_neurons = [int(i) for i in row.HiddenLayerNeurons.split(';')]
        use_reg = row.UseRegularization
        use_drop = row.UseDropout
        splits = row.Splits
        epochs = epochs

        # Lese den Datensatz ein 
        # # HIER MUSST DU VIELLEICHT NOCH ERGÄNZEN, FALLS ZWEI DATENSÄTZE MITEINANDER KOMBINIERT WERDEN
        df_datasets = dp.read_datasets_fragment(datasets, 0.7, 'from_begin', algo) 
        df_datasets = df_datasets[feat_labs]

        df_datasets_eval = dp.read_datasets_fragment(datasets_eval, 0.3, 'from_end', algo)
        df_datasets_eval = df_datasets_eval[feat_labs]

        # Erstelle das MH MLP Modell aufgrund der Parameter aus der Experiment-Datei
        model = MHMLP((hid_neurons, use_reg, use_drop))
        validator = ModelValidator('Multiheaded MLP', model, df_datasets, df_datasets_eval)
        test_score = validator.evaluate(epochs=epochs, time_steps=time_steps, forcast_steps=for_steps, splits=splits)

        return test_score

def evaluate_models_fragment(title, train_quota, eval_quota, *eval_items):
    '''Trainiert und Evaluiert eine Menge von Modellen die als Parameter übergeben werden

        :titel: Dieser Titel wird in den Dateinnamen eingebettet
        :train_quota: Prozentsatz der besagt, wie viel von den Datensätzen für das Trainineren verwendet werden sollen
        :test_quota: Prozentsatz der besagt, wie viel von den Datensätzen für das Evaluieren verwendet werden sollen
        :eval_items: Liste von Dictionaries mit folgender Signatur {id: XX, dataset: XX, epochs: XX}
    '''
    
    df_eval_results = DataFrame(columns=['Id', 'Loss', 'Acc'], index=None)

    for eval_item in eval_items:
        id = eval_item['id']
        dataset = eval_item['dataset']
        epochs = eval_item['epochs']
        
        test_score = evaluate_model_fragment(id, dataset, train_quota, eval_quota, epochs)

        df_eval_results = df_eval_results.append({'Id': id, 'Loss': test_score[0], 'Acc': test_score[1]}, ignore_index=True)

    df_eval_results = df_eval_results.astype({'Id': int})
    df_eval_results.set_index('Id', inplace=True)
    df_eval_results.to_csv(f"{mlp_artifacts_path}/all_results/score_{title}_eval.csv")



