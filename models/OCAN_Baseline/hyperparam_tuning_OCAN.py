# hyperparam_tuning_OCAN.py
import os

import numpy as np
import pandas as pd
import datetime
import pytz
import numpy as np
import tensorflow as tf
import pandas as pd

import time
import optuna
from optuna.samplers import TPESampler, RandomSampler
from models.OCAN_Baseline.oc_gan_1 import GANModel

# Standard library imports

import warnings
import shutil
# Third-party imports
import numpy as np
import pandas as pd
import tensorflow as tf




from models.OCAN_Baseline.regularGan import regular_GANModel

from utils.utils import   BlockingTimeSeriesSplit, get_git_commit_hash, getBenignData,  F1Score, print_box_with_header

# Set TensorFlow and MLflow configurations
tf.config.run_functions_eagerly(False)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import mlflow
import mlflow.tensorflow
import mlflow.pyfunc

# Suppress warnings
warnings.filterwarnings("ignore")
def train_regular_gan(ocan_model, X_ben_train, y_ben_train, X_val, y_val, PHASE2_TRAIN_EPOCHS):

    train_data = tf.data.Dataset.from_tensor_slices((X_ben_train, y_ben_train)).shuffle(buffer_size=1024, seed=42).batch(512)
    test_data = (X_val, y_val)
    history = ocan_model.fit(train_data, test_data, epochs=PHASE2_TRAIN_EPOCHS)

    return history

def HyperparamTuningOCAN( combined_val_df,expermiment_name,epochs,max_f1_score_dict_params,dim_inp,n_trials=150,  patience=3):

    n_splits = 3

    mlflow.set_experiment(expermiment_name + "_ht")
    best_params = None
    combined_val_df = combined_val_df.sort_values('ORDER_PROCESSED_TIMESTAMP').reset_index(drop=True)
    def objective(trial):
        
        g_d_layers_choice = trial.suggest_categorical('G_D_layers', [
           [[50, 100,150], [150,100,50]],[[100, 150,300], [300,150,100]]
            ])
        num_layers_disc = len(g_d_layers_choice[1])
        max_dropout = 0.5
        dropout_rates_disc = []
        for i in range(num_layers_disc):
            dropout_rate = trial.suggest_uniform(f'dropout_rate_{i}', 0.1, max_dropout)
            dropout_rates_disc.append(dropout_rate)
            max_dropout = dropout_rate

        num_layers_gen = len(g_d_layers_choice[0])
        max_dropout = 0.5
        dropout_rates_gen = []
        for i in range(num_layers_gen):
            dropout_rate = trial.suggest_uniform(f'dropout_rate_gen_{i}', 0.1, max_dropout)
            dropout_rates_gen.append(dropout_rate)
            max_dropout = dropout_rate
       
        batcNorm  = trial.suggest_categorical('batch_norm', [True, False])

    
        params = {
            'G_D_layers': g_d_layers_choice,
            'd_lr': trial.suggest_loguniform('d_lr', 1e-8, 1e-2),
            'g_lr': trial.suggest_loguniform('g_lr', 1e-8,1e-2),
            'mb_size': 256,
            'Z_dim': g_d_layers_choice[0][0],
            'dim_inp': trial.suggest_int('dim_inp',dim_inp,dim_inp),
            "g_dropouts": dropout_rates_gen,
            "d_dropouts": dropout_rates_disc,
            'batch_norm_g': trial.suggest_categorical('batch_norm', [True, False]),
            'batch_norm_d': False,
            'beta1_g' : trial.suggest_float('beta1_g', 0.5, 0.9) if trial.suggest_categorical('batch_norm_g', [True, False]) else trial.suggest_float('beta1_g', 0.3, 0.5),
            'beta2_g' : trial.suggest_float('beta2_g', 0.990, 0.999),
            'beta1_d' : trial.suggest_float('beta1_d', 0.3, 0.5),
            'beta2_d' : trial.suggest_float('beta2_d', 0.990, 0.999),
            "lambda_pt": trial.suggest_float('lambda_pt', 0.1, 2),
            "lambda_gp": trial.suggest_float('lambda_gp', 1, 20),	
            "lambda_fm": trial.suggest_float('lambda_fm', 0.1, 5),
            "lambda_ent":  trial.suggest_float('lambda_ent', 0.1, 2)
        }
    #     params = {'G_D_layers': [[50, 100], [100, 50]], 
    # "d_lr": 1e-06,
    # "g_lr": 1e-05,
    # "mb_size": 512,  
    # "Z_dim": 50,     
    # "dim_inp": 128,   
    # "g_dropouts": [0.08, 0.08],
    # "d_dropouts": [ 0.34,  0.34],
    # "batch_norm_g": 1,
    # "batch_norm_d": 0,       
    # "beta1_g": 0.5,            
    # "beta2_g": 0.990,          
    # "beta1_d": 0.5,            
    # "beta2_d": 0.999,
    # "lambda_pt": 1,
    # "lambda_gp": 2,
    # "lambda_fm": 1,
    # "lambda_ent": 1}
        print_box_with_header('params set')
        print(params)
        base_path = f'saved_models/pre_trained_disc_val/{max_f1_score_dict_params["ph1_tags.mlflow.runName"]}/'
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        
        start_time = time.time()
        fold_results = []
        k = 1
        btss = BlockingTimeSeriesSplit(n_splits=3)
        for train_index, val_index in btss.split(combined_val_df):
            print_box_with_header(f'FOLD {k}')
            combined_train =combined_val_df.iloc[train_index]
            combined_val = combined_val_df.iloc[val_index]

            X_ben_train,y_ben_train = getBenignData(combined_train,'hidden_rep')
            X_ben_val,y_ben_val = getBenignData(combined_val,'hidden_rep')

            X_val = np.vstack(combined_val['hidden_rep'].to_numpy())
            y_val = np.vstack(combined_val['label'].to_numpy())

            y_val = tf.one_hot(y_val, depth=2)
            y_val = tf.squeeze(y_val, axis=1)

            y_ben_train = tf.one_hot(y_ben_train, depth=2)
            y_ben_train = tf.squeeze(y_ben_train, axis=1)


            train_data = tf.data.Dataset.from_tensor_slices((X_ben_train, y_ben_train)).shuffle(buffer_size=1024, seed=42).batch(params['mb_size'])
            test_data = (X_val, y_val)
            val_data_tar = tf.data.Dataset.from_tensor_slices((X_val, y_val)).shuffle(buffer_size=1024, seed=42).batch(512)
            train_data_tar = tf.data.Dataset.from_tensor_slices((X_ben_train, y_ben_train)).shuffle(buffer_size=2024, seed=42).batch(512)
            
            model = GANModel(params, X_ben_train.shape[0])
 
            fold_base_path = os.path.join(base_path, str(k))
            os.makedirs(fold_base_path, exist_ok=True)

            model_path = os.path.join(fold_base_path, 'discriminator_tar')
            if False:
                model.discriminator_tar = tf.keras.models.load_model(model_path)
            else:
                reg_model = regular_GANModel(params, X_ben_train.shape[0],model_path)
                history_reg  = train_regular_gan(reg_model, X_ben_train, y_ben_train, X_ben_val, y_ben_val, 10)
                # model.discriminator_tar.compile(optimizer=model.discriminator_tar_optimizer,
                # metrics=['accuracy'])
                # history = model.discriminator_tar.fit_pretrained_disc(train_data_tar,val_data_tar, epochs=200,optimizer=model.discriminator_tar_optimizer)
                model.discriminator_tar = reg_model.discriminator
                model.discriminator_tar.save(model_path, save_format='tf')
            history = model.fit(train_data, test_data, epochs=epochs)  

            evaluation_result  = model.evaluate_model(X_val, y_val)
            k += 1
            fold_results.append(list(evaluation_result.values()))

        average_score = np.mean(fold_results, axis=0)
        std_score = np.std(fold_results, axis=0)
        print(f"Trial {trial.number} completed. f1 score: {average_score[1]}")

        with mlflow.start_run() as run:
            # Log metrics and parameters
            mlflow.log_metric("roc_Auc_score", average_score[0])
            mlflow.log_metric("F1score", average_score[1])
            mlflow.log_metric("F1score_std", std_score[1])
            mlflow.log_metric("avg_prec", average_score[2])
            mlflow.log_metric("avg_prec_std", std_score[2])
            mlflow.log_metric("accuracy", average_score[3])
            mlflow.log_metric("recall", average_score[4])
            mlflow.log_metric("recall_std", std_score[4])
            mlflow.log_metric("precision", average_score[5])
            mlflow.log_metric("precision_std", std_score[5])
            mlflow.log_metric('Tp', average_score[6])
            mlflow.log_metric('Tn', average_score[7])
            mlflow.log_metric('Fp', average_score[8])
            mlflow.log_metric('Fn', average_score[9])

            mlflow.log_params(params)
            mlflow.log_params({'epochs': epochs})
            mlflow.set_tag('git_commit_hash', get_git_commit_hash())
            mlflow.set_tag('time_elapsed', (time.time() - start_time)/60 )
            mlflow.set_tag('tag_name', 'added normalization layers in discriminator')
            mlflow.log_params(max_f1_score_dict_params)    


        return average_score[1]

    study = optuna.create_study(sampler=RandomSampler(), study_name='fdt_optuna_phase2_hyp', direction='maximize')

    study.sampler = TPESampler()

    study.optimize(objective, n_trials=500)

    # Retrieve the best parameters and the best score
    best_params = study.best_params
    best_score = study.best_value

    print(f"Best score: {best_score}")
    print(f"Best params: {best_params}")

    return best_params

