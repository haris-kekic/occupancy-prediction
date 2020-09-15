import experiment_mlp as exp_mlp
import experiment_rnn_lstm as exp_rnn


def mlp_collect_results():
    exp_mlp.compose_all_results()
    exp_mlp.arrange_best_scores('building1', 'building1', 'std')
    exp_mlp.arrange_best_scores('building1', 'building1', 'ext')
    exp_mlp.arrange_best_scores('building0_5', 'building0;building1;building2;building3;building4;building5', 'std')

def mlp_eval_case1():
    eval_dataset = 'building4'
    exp_mlp.evaluate_models( 'building1_std',
                        {'id': 192, 'dataset': eval_dataset, 'epochs': 60 }, 
                        {'id': 240, 'dataset': eval_dataset, 'epochs': 35 },
                        {'id': 196, 'dataset': eval_dataset, 'epochs': 40 }, 
                        {'id': 200, 'dataset': eval_dataset, 'epochs': 32 },
                        {'id': 960, 'dataset': eval_dataset, 'epochs': 56 } )
    

def mlp_eval_case2():
    eval_dataset = 'building4'
    exp_mlp.evaluate_models( 'building1_ext',
                        {'id': 3264, 'dataset': eval_dataset, 'epochs': 80 }, 
                        {'id': 3312, 'dataset': eval_dataset, 'epochs': 35 },
                        {'id': 4080, 'dataset': eval_dataset, 'epochs': 35 }, 
                        {'id': 3268, 'dataset': eval_dataset, 'epochs': 50 },
                        {'id': 4032, 'dataset': eval_dataset, 'epochs': 70 } )

def mlp_eval_case3():
    eval_dataset = 'building7'
    exp_mlp.evaluate_models( 'building0_5',
                        {'id': 7152, 'dataset': eval_dataset, 'epochs': 32 }, 
                        {'id': 6336, 'dataset': eval_dataset, 'epochs': 42 },
                        {'id': 7104, 'dataset': eval_dataset, 'epochs': 32 }, 
                        {'id': 7344, 'dataset': eval_dataset, 'epochs': 8 },
                        {'id': 6384, 'dataset': eval_dataset, 'epochs': 8 } )

def mlp_eval_case4():
    eval_dataset = 'building0;building1;building2;building3;building4;building5'
    exp_mlp.evaluate_models_fragment( 'building0_5_fragment', 0.7, 0.3,
                        {'id': 7152, 'dataset': eval_dataset, 'epochs': 32 }, 
                        {'id': 6336, 'dataset': eval_dataset, 'epochs': 42 },
                        {'id': 7104, 'dataset': eval_dataset, 'epochs': 32 }, 
                        {'id': 7344, 'dataset': eval_dataset, 'epochs': 8 },
                        {'id': 6384, 'dataset': eval_dataset, 'epochs': 8 } )


def rnn_collect_results():
    exp_rnn.compose_all_results()
    exp_rnn.arrange_best_scores('building1', 'building1', 'std')
    exp_rnn.arrange_best_scores('building1', 'building1', 'ext')
    exp_rnn.arrange_best_scores('building0-5', 'building0;building1;building2;building3;building4;building5', 'std')

def rnn_eval_case1():
    eval_dataset = 'building4'
    exp_rnn.evaluate_models( 'building1_std',
                        {'id': 72, 'dataset': eval_dataset, 'epochs': 100 }, 
                        {'id': 4005, 'dataset': eval_dataset, 'epochs': 60 },
                        {'id': 80, 'dataset': eval_dataset, 'epochs': 55 }, 
                        {'id': 656, 'dataset': eval_dataset, 'epochs': 75 },
                        {'id': 664, 'dataset': eval_dataset, 'epochs': 35 } )

def rnn_eval_case2():
    eval_dataset = 'building4'
    exp_rnn.evaluate_models( 'building1_ext',
                        {'id': 4019, 'dataset': eval_dataset, 'epochs': 40 }, 
                        {'id': 1224, 'dataset': eval_dataset, 'epochs': 35 },
                        {'id': 1232, 'dataset': eval_dataset, 'epochs': 70 }, 
                        {'id': 1512, 'dataset': eval_dataset, 'epochs': 48 },
                        {'id': 1520, 'dataset': eval_dataset, 'epochs': 75 } )

def rnn_eval_case3():
    eval_dataset = 'building7'
    exp_rnn.evaluate_models( 'building0_5',
                        {'id': 2304, 'dataset': eval_dataset, 'epochs': 20 }, 
                        {'id': 2448, 'dataset': eval_dataset, 'epochs': 19 },
                        {'id': 2392, 'dataset': eval_dataset, 'epochs': 22 }, 
                        {'id': 2744, 'dataset': eval_dataset, 'epochs': 10 },
                        {'id': 2528, 'dataset': eval_dataset, 'epochs': 25 } )

def rnn_eval_case4():
    eval_dataset = 'building0;building1;building2;building3;building4;building5'
    exp_rnn.evaluate_models_fragment( 'building0_5_fragment', 0.7, 0.3,
                         {'id': 2304, 'dataset': eval_dataset, 'epochs': 20 }, 
                        {'id': 2448, 'dataset': eval_dataset, 'epochs': 19 },
                        {'id': 2392, 'dataset': eval_dataset, 'epochs': 22 }, 
                        {'id': 2744, 'dataset': eval_dataset, 'epochs': 10 },
                        {'id': 2528, 'dataset': eval_dataset, 'epochs': 25 } )


#mlp_collect_results()
rnn_collect_results()
# mlp_eval_case3()
# mlp_eval_case4()
# rnn_eval_case2()
#rnn_eval_case3()
