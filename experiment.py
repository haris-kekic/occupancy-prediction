# -----------------------------------------------------------
# Skript zum Starten der für die Arbeit vorgehsehenen Experimente
#
# (C) 2020 Haris Kekic, Fernuniversität in Hagen
# -----------------------------------------------------------

import experiment_mlp as exp_mlp
import experiment_rnn_lstm as exp_rnn


# Generiert die Einstellungen (Kombinationen) von Parametern
# die als Experiment durchgeführt werden sollen
# Die Experimente, können in beliebige Anzahl von Dateien
# aufgesplittet werden um sie parallel abarbeiten zu können
#exp_mlp.generate_experiment_hyperparams(num_of_files=8)
exp_rnn.generate_experiment_hyperparams(num_of_files=20)

# Parallele Abarbeitung der Experiment-Parameter Dateien
# if __name__ == '__main__':
#     exp_mlp.run_experiments_async()

# if __name__ == '__main__':
#     exp_rnn.run_experiments_async()


# Zum Testen! Nur eine Experiment-Datei wird ausgeführt
#run_experiment(f'{mlp_artifacts_path}/experiments_part8.csv', 200)




#exp_rnn.run_experiment(f'{exp_rnn.rnn_artifacts_path}/experiments_part5.csv')

exp_mlp.run_experiment(f'{exp_mlp.mlp_artifacts_path}/experiments_part5.csv')