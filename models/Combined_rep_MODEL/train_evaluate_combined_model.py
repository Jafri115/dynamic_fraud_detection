# Standard library imports

import time
import warnings

# Third-party imports
import numpy as np
import pandas as pd
from optuna.samplers import TPESampler, RandomSampler
import tensorflow as tf
import optuna
from optuna.samplers import TPESampler, RandomSampler
from tensorflow.python.keras.metrics  import Precision, Recall, AUC, TruePositives, TrueNegatives, FalsePositives, FalseNegatives 
import tensorflow_addons as tfa
from keras.callbacks import ModelCheckpoint
from models.Combined_rep_MODEL.combined_rep_integ_model import CombinedModel
from utils.utils import   BlockingTimeSeriesSplit, CombinedCustomLoss,   print_box_with_header , transform_inputs, prepare_data_for_training,get_git_commit_hash,F1Score,evaluate_model
# Set TensorFlow and MLflow configurations
tf.config.run_functions_eagerly(False)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import mlflow
import mlflow.tensorflow
import mlflow.pyfunc

import os
# Suppress warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def prepare_data( train_data, val_data, params, resample_dict, output_label_name, n_event_code, selected_col, train_flags):

    if resample_dict:
        train_data, _ = prepare_data_for_training(train_data, output_label_name, resample_dict)
        
    x_train, y_train = transform_inputs(train_data, n_event_code,  params['max_len'],selected_col[:params['num_of_tabular_columns']],train_flags,output_label_name)
    x_val, y_val = transform_inputs(val_data ,n_event_code,  params['max_len'],selected_col[:params['num_of_tabular_columns']],train_flags,output_label_name)
    
    train_data = (x_train, y_train)
    val_data = (x_val, y_val)

    return train_data, val_data 


def create_model(params, n_event_code, batch_size, is_public_dataset):
    classifier_model =  CombinedModel(params['num_of_tabular_columns'], 
                            params['max_len'], 
                            params['combined_hidden_layers'], 
                            params['dropout_rate_comb'], 
                            params['dropout_rate_seq'],
                            params['droput_rate_tab'],
                            params['layer'], 
                            params['tab_hidden_states'], 
                            batch_size,
                            n_event_code,
                            is_public_dataset,
                            l2_lambda_comb=params['l2_lambda_comb'],
                            l2_lambda_tab=params['l2_lambda_tab'],
                            l2_lambda_seq=params['l2_lambda_seq'],
                            model_dim= params['model_dim']
                            )      
    optimizer = tfa.optimizers.AdamW(learning_rate=params['learning_rate'], weight_decay=params['weight_decay'])
    classifier_model.compile(optimizer=optimizer,
                            loss= CombinedCustomLoss(),
                            metrics=[
                                    'accuracy', 
                                    F1Score(),
                                    Precision(name='precision', thresholds=0.5),
                                    Recall(name='recall', thresholds=0.5),
                                    AUC(name='auc_roc', curve='ROC'),
                                    AUC(name='auc_pr', curve='PR'),
                                    TruePositives(name='tp'),
                                    TrueNegatives(name='tn'),
                                    FalsePositives(name='fp'),
                                    FalseNegatives(name='fn'),
                            ])
    return classifier_model

def hypertune_combined_model(merged_train_df,
                             n_event_code,
                             batch_size,
                             selected_col,
                             train_flags,
                             resample_dict,
                             output_label_name,
                             experiment_name,
                             epochs=10,
                             is_public_dataset=False,
                             n_trials=50,  # Number of Bayesian optimization trials
                             patience=3,
                             tag_name =''
                             ):  # Patience for early stopping

    
    experiment_name = experiment_name + '_ht'
    mlflow.set_experiment(experiment_name)

    merged_train_df = merged_train_df.sort_values('ORDER_PROCESSED_TIMESTAMP').reset_index(drop=True)
    def objective(trial):
        start_time = time.time()
      

        params = {
            'num_of_tabular_columns': trial.suggest_int('num_of_tabular_columns', 140, 140),
            'max_len': trial.suggest_int('max_len', 50, 70),
            'combined_hidden_layers': trial.suggest_categorical('combined_hidden_layers', [[200, 128]]),
            'dropout_rate_comb': trial.suggest_float('dropout_rate', 1e-2, 1e-1),
            'dropout_rate_seq': trial.suggest_float('dropout_rate_seq', 1e-2, 0.5),
            'droput_rate_tab': trial.suggest_float('droput_rate_tab',1e-3, 0.3),
            'layer': trial.suggest_int('layer', 1,1),
            'tab_hidden_states':  trial.suggest_categorical('tab_hidden_states', [[64]]),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-6, 1e-1),
            'weight_decay' : trial.suggest_float('weight_decay',  1e-8, 1e-6),
            'l2_lambda_comb': trial.suggest_float('l2_lambda_comb', 1e-8, 1e-1),
            'l2_lambda_tab': trial.suggest_float('l2_lambda_tab', 1e-8, 1e-6),
            'l2_lambda_seq': trial.suggest_float('l2_lambda_seq', 1e-8, 1e-1),
            'model_dim': trial.suggest_categorical('model_dim', [64]),

        }
        print_box_with_header('params set')
        print(params)
        evaluation_array = []
        k = 1
        starttime_param = time.time()
        

        btss = BlockingTimeSeriesSplit(n_splits=3)
        for train_index, val_index in btss.split(merged_train_df):
            starttime_fold = time.time()
            print_box_with_header(f'FOLD {k}')
            # Prepare the data (resampling and transformations)
            train_data = merged_train_df.iloc[train_index]
            val_data = merged_train_df.iloc[val_index]
            train_data, val_data = prepare_data( train_data, val_data, params, resample_dict, output_label_name, n_event_code, selected_col, train_flags)
            # Create the model
            model = create_model(params, n_event_code, batch_size, is_public_dataset)
            print('train percentage : 1 ',np.mean(train_data[1]))
            print('val percentage : 1 ',np.mean(val_data[1]))
            model.fit(
                train_data=train_data,
                val_data=val_data, 
                epochs=epochs,
                train_flags=train_flags,
            )

            evaluation = evaluate_model(model, val_data, batch_size,train_flags)
            evaluation_array.append(evaluation)

            print(f"Fold {k} completed. f1 score: {evaluation[1]}")

            print(f'time elapsed for fold :{((time.time()-starttime_fold)/60)}')
            k = k + 1

        avg_score = np.array(evaluation_array).mean(axis=0)
        std_score = np.array(evaluation_array).std(axis=0)
        print(f"Trial {trial.number} completed. f1 score: {evaluation[1]}")
        print(f'time elapsed for parameter set:{((time.time()-starttime_param)/60)}')


        # Log metrics and parameters
        with mlflow.start_run() as run:
            mlflow.log_metric("accuracy", avg_score[0])
            mlflow.log_metric("avg_prec", avg_score[13])
            mlflow.log_metric("avg_prec_std", std_score[13])
            mlflow.log_metric("f1_score", avg_score[1])
            mlflow.log_metric("f1_score_std", std_score[1])
            mlflow.log_metric("precision", avg_score[2])
            mlflow.log_metric("precision_std", std_score[2])
            mlflow.log_metric("recall", avg_score[3])
            mlflow.log_metric("recall_std", std_score[3])
            mlflow.log_metric("auc_roc", avg_score[4])
            mlflow.log_metric("auc_pr", avg_score[5])
            mlflow.log_metric("tp", avg_score[6])
            mlflow.log_metric("tn", avg_score[7])
            mlflow.log_metric("fp", avg_score[8])
            mlflow.log_metric("fn", avg_score[9])

            mlflow.log_params(params)
            mlflow.set_tag('git_commit_hash', get_git_commit_hash())
            mlflow.set_tag('time_elapsed',(time.time() - start_time)/3600)
            mlflow.set_tag('tag_name',tag_name)
            train_flags_name = 'train_combined' if train_flags['train_combined'] else 'train_seq' if train_flags['train_seq'] else 'train_tab'
            resample_tech = 'smote' if resample_dict['smote'] else 'rus' if resample_dict['rus'] else 'ns'
            mlflow.log_params({"eval_size": len(val_data[1]),
                               "train_size": len(train_data[1]),
                               "train_label_1_percentage": np.mean(train_data[1]),
                               "val_label_1_percentage": np.mean(val_data[1]),
                               "train_flags_name": train_flags_name,
                               'epochs': epochs,
                               'resample_tech': resample_tech,
                               })
        return avg_score[1] # average_precision


    study = optuna.create_study(sampler=RandomSampler(), study_name='fdt_optuna_hyp', direction='maximize')

    # Define the number of initial trials with RandomSampler and total number of trials
    initial_random_trials = 5
    total_trials = n_trials  # Total number of trials including the initial random trials
    # Start the initial optimization with RandomSampler
    if len(study.trials) < initial_random_trials:
        study.optimize(objective, n_trials=initial_random_trials - len(study.trials))
    # After the initial random trials, switch to TPESampler for the remaining trials
    study.sampler = TPESampler()
    remaining_trials = total_trials - len(study.trials)
    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials)


    # Retrieve the best parameters and the best score
    best_params = study.best_params
    best_score = study.best_value

    print(f"Best score: {best_score}")
    print(f"Best params: {best_params}")

    return best_params



def TrainedMergedModels_new(merged_train_df,
                            merged_val_df,
                            merged_test_df,
                            n_event_code,
                            batch_size,
                            params,
                            selected_col,
                            train_flags,
                            resample_dict,
                            output_label_name,
                            experiment_name,
                            is_public_dataset,
                            epochs=10,
                            train_model=1,
                            freez_layers={"seq": 0, "tab": 1},
                            tag_name =''
                        ):
        
        experiment_name = experiment_name + '_tr'
        mlflow.set_experiment(experiment_name)
        print_box_with_header('params set')
        print(params)
 

        if freez_layers['no_freez'] :

            
            train_data, val_data  = prepare_data(merged_train_df,  merged_val_df, params, resample_dict, output_label_name, n_event_code, selected_col, train_flags)
            x_test, y_test = transform_inputs(merged_test_df ,n_event_code,  params['max_len'],selected_col[:params['num_of_tabular_columns']],train_flags,output_label_name)
            test_data = (x_test, y_test)
                
            classifier_model = create_model(params, n_event_code, batch_size, is_public_dataset)
            base_path = 'saved_models/Fraud/Phase1/'
            filename = base_path + time.strftime("%Y%m%d-%H%M%S") + '_combined_model.h5'
                        # Example sample input creation
            start_time = time.time()
            checkpoint = ModelCheckpoint(filename, save_best_only=True,save_weights_only=True, monitor='val_f1_score', mode='max',verbose=1)

            history = classifier_model.fit(
                train_data=train_data,
                val_data=val_data,  
                epochs=epochs,
                train_flags=train_flags,
                callbacks=[checkpoint]

            )
            
            print('saving model')

        else:
            train_flags = { 'train_combined':1, 'train_tab': 0, 'train_seq': 0}            
            train_data, val_data  = prepare_data(merged_train_df,  merged_val_df, params, resample_dict, output_label_name, n_event_code, selected_col, train_flags)
            x_test, y_test = transform_inputs(merged_test_df ,n_event_code,  params['max_len'],selected_col[:params['num_of_tabular_columns']],train_flags,output_label_name)
            test_data = (x_test, y_test)
                
            classifier_model = create_model(params, n_event_code, batch_size, is_public_dataset)
            base_path = 'saved_models/Fraud/Phase1/'
            filename = base_path + time.strftime("%Y%m%d-%H%M%S") + '_combined_model.h5'
                        # Example sample input creation
            start_time = time.time()


            if freez_layers['seq']:

                checkpoint = ModelCheckpoint(filename, save_best_only=True,save_weights_only=True, monitor='val_f1_score', mode='max',verbose=1)
                for layer in classifier_model.tabular_model.layers:
                    layer.trainable = False
                # classifier_model.tabular_model.trainable = False
                weights_before = [layer.get_weights() for layer in classifier_model.tabular_model.layers]

                history = classifier_model.fit(
                    train_data=train_data,
                    val_data=val_data,  
                    epochs=epochs,
                    train_flags=train_flags,
                    callbacks=[checkpoint]

                )
                weights_after = [layer.get_weights() for layer in classifier_model.tabular_model.layers]
                for before, after in zip(weights_before, weights_after):
                    if np.any([not np.array_equal(b, a) for b, a in zip(before, after)]):
                        print("Weights changed.")
                    else:
                        print("Weights not changed.")
                classifier_model.sequential_model.trainable = False
            elif freez_layers['tab']:
                # for layer in classifier_model.sequential_model.layers:
                #     layer.trainable = False
                # classifier_model.tabular_model.trainable = False
                weights_before = [layer.get_weights() for layer in classifier_model.sequential_model.layers]
                classifier_model.sequential_model.trainable = False
                checkpoint = ModelCheckpoint(filename, save_best_only=True,save_weights_only=True, monitor='val_f1_score', mode='max',verbose=1)
                history = classifier_model.fit(
                    train_data=train_data,
                    val_data=val_data,  
                    epochs=epochs,
                    train_flags=train_flags,
                    callbacks=[checkpoint]

                )
                weights_after = [layer.get_weights() for layer in classifier_model.sequential_model.layers]
                for before, after in zip(weights_before, weights_after):
                    if np.any([not np.array_equal(b, a) for b, a in zip(before, after)]):
                        print("Weights changed.")
                    else:
                        print("Weights not changed.")
                for layer in classifier_model.tabular_model.layers:
                    layer.trainable = False
            
            base_path = 'saved_models/Fraud/Phase1/'
            filename = base_path + time.strftime("%Y%m%d-%H%M%S") + '_combined_model.h5'
            checkpoint = ModelCheckpoint(filename, save_best_only=True,save_weights_only=True, monitor='val_f1_score', mode='max',verbose=1)

            history = classifier_model.fit(
                train_data=train_data,
                val_data=val_data,  
                epochs=epochs,
                train_flags=train_flags,
                callbacks=[checkpoint]

            )
            
            print('saving model')

        if not freez_layers['both_freeze']:    
            accuracy_naive = [1-np.mean(test_data[1])] * epochs

            evaluation = classifier_model.evaluate_model( test_data, batch_size , train_flags)
            with mlflow.start_run() as run:
                mlflow.log_metric("accuracy", evaluation['accuracy'])
                mlflow.log_metric('avg_prec', evaluation['avg_precision'])
                mlflow.log_metric("f1_score", evaluation['f1_score'])
                mlflow.log_metric("precision", evaluation['precision'])
                mlflow.log_metric("recall", evaluation['recall'])
                mlflow.log_metric("auc_roc", evaluation['auc_roc'])
                mlflow.log_metric("auc_pr", evaluation['auc_pr'])
                mlflow.log_metric("tp", evaluation['tp'])
                mlflow.log_metric("tn", evaluation['tn'])
                mlflow.log_metric("fp", evaluation['fp'])
                mlflow.log_metric("fn", evaluation['fn'])

                mlflow.log_params(params)
                mlflow.set_tag('git_commit_hash', get_git_commit_hash())
                mlflow.set_tag('time_elapsed',(time.time() - start_time)/60)
                mlflow.set_tag('tag_name',tag_name+'loss3_tl')
                train_flags_name = 'train_combined' if train_flags['train_combined'] else 'train_seq' if train_flags['train_seq'] else 'train_tab'
                resample_tech = 'smote' if resample_dict['smote'] else 'rus' if resample_dict['rus'] else 'ns'
                freez_layers_name = 'seq' if freez_layers['seq'] else 'tab' if freez_layers['tab'] else 'no_freez'
                mlflow.log_params({"eval_size": len(val_data[1]),
                                "train_size": len(train_data[1]),
                                "train_label_1_percentage": np.mean(train_data[1]),
                                "val_label_1_percentage": np.mean(val_data[1]),
                                "train_flags_name": train_flags_name,
                                'epochs': epochs,
                                'resample_tech': resample_tech,
                                'freez_layers_name': freez_layers_name,
                                'filename': filename,
                            
                                })
                run_name = run.info.run_name

            # plot_combined_metrics(history,accuracy_naive,experiment_name, run_name)

            if os.path.exists(filename):
                #classifier_model.load_weights('saved_models/Fraud/Phase1/20240402-080143_combined_model.h5') 
                classifier_model.load_weights(filename)
        if resample_dict['smote'] or resample_dict['rus']:
            train_data, val_data  = prepare_data(merged_train_df,  merged_val_df, params, None, output_label_name, n_event_code, selected_col, train_flags)

        combined_representation_train,concat_representation_train = classifier_model.get_representation_in_batches(train_data,512 ,train_flags)
        combined_representation_val,concat_representation_val = classifier_model.get_representation_in_batches(val_data,512,train_flags)
        combined_representation_test,concat_representation_test = classifier_model.get_representation_in_batches(test_data,512,train_flags)

        prediction_train = classifier_model.predict_in_batches(train_data,512,train_flags)
        prediction_test = classifier_model.predict_in_batches(test_data,512,train_flags)
        prediction_val = classifier_model.predict_in_batches(val_data,512,train_flags)

        threshold = 0.5
        prediction_train_label = tf.cast(prediction_train > threshold, tf.int32)

        prediction_test_label = tf.cast(prediction_test > threshold, tf.int32)
        prediction_val_label = tf.cast(prediction_val > threshold, tf.int32)

        if not is_public_dataset:
            combined_representation_train_df = pd.DataFrame({
            'hidden_rep': list(combined_representation_train),
            'label': merged_train_df['SEQ_ORDER_LABEL'].to_numpy(),
            'ORDER_PROCESSED_TIMESTAMP': merged_train_df['ORDER_PROCESSED_TIMESTAMP'].to_numpy(),
            'predict_label': list(prediction_train_label),
            'prediction': list(prediction_train),
            'concat_rep': list(concat_representation_train),
                })
            combined_representation_val_df = pd.DataFrame({
            'hidden_rep': list(combined_representation_val),
            'label': merged_val_df['SEQ_ORDER_LABEL'].to_numpy(),
            'ORDER_PROCESSED_TIMESTAMP': merged_val_df['ORDER_PROCESSED_TIMESTAMP'].to_numpy(),
            'predict_label': list(prediction_val_label),
            'prediction': list(prediction_val),
            'concat_rep': list(concat_representation_val),
                })
            combined_representation_test_df = pd.DataFrame({
            'ORDER_PROCESSED_TIMESTAMP': merged_test_df['ORDER_PROCESSED_TIMESTAMP'].to_numpy(),
            'hidden_rep': list(combined_representation_test),
            'label': merged_test_df['SEQ_ORDER_LABEL'].to_numpy(),
            'ORDER_PROCESSED_TIMESTAMP': merged_test_df['ORDER_PROCESSED_TIMESTAMP'].to_numpy(),
            'predict_label': list(prediction_test_label),
            'prediction': list(prediction_test),
            'concat_rep': list(concat_representation_test),
                })
        elif is_public_dataset:
            combined_representation_train_df = pd.DataFrame({
            'hidden_rep': list(combined_representation_train),
            'label': merged_train_df['SEQ_'+output_label_name].to_numpy(),
                })
            combined_representation_test_df = pd.DataFrame({
            'hidden_rep': list(combined_representation_test),
            'label': merged_test_df['SEQ_'+ output_label_name].to_numpy(),
                })
            combined_representation_val_df = pd.DataFrame({
            'hidden_rep': list(combined_representation_val),
            'label': merged_val_df['SEQ_'+output_label_name].to_numpy(),
                })


        return combined_representation_train_df,combined_representation_val_df, combined_representation_test_df,run_name

def train_representation_learning(dataset_name,
                                merged_train_df,
                                merged_val_df,
                                merged_test_df,
                                selected_col,
                                output_label_name,
                                n_event_code,
                                best_param,
                                train_flags,
                                resample_dict,
                                experiment_name,
                                phase1_train_epochs,
                                phase1_tune_epochs,
                                tune_representation_phase1,
                                freez_layers,
                                is_public_dataset,
                                tag_name ='test'
                                ):
    np.random.seed(42)
    batch_size = 50
    if tune_representation_phase1==1:
        print_box_with_header('HYPERTUNING PHASE 1: REPRESENTATION LEARNING')
        best_param = hypertune_combined_model(merged_val_df,
                                              n_event_code, 
                                              batch_size,  
                                              selected_col,
                                              train_flags,
                                              resample_dict,
                                              output_label_name,
                                              experiment_name,
                                              epochs=phase1_tune_epochs,
                                              is_public_dataset=is_public_dataset,
                                              tag_name= tag_name)

    print_box_with_header('TRAINING PHASE 1: REPRESENTATION LEARNING')

    combined_trained_df ,combined_val_df, combined_test_df ,  run_name = (

        TrainedMergedModels_new(merged_train_df, merged_val_df,
                                merged_test_df,
                                n_event_code,
                                batch_size,
                                best_param,
                                selected_col,train_flags,
                                resample_dict,
                                output_label_name,
                                experiment_name=experiment_name,
                                is_public_dataset = is_public_dataset,
                                epochs = phase1_train_epochs,
                                freez_layers = freez_layers,
                                tag_name = tag_name
                                )
    )


    base_path = 'data/processed_data/'+ dataset_name +'/Phase1_latent_representations'
    hidden_rep_shape  = combined_trained_df['hidden_rep'].iloc[0].shape[0]

    dir_name = base_path + '/' + run_name
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


    file_path_train = os.path.join(dir_name, f"combined_trained_df_{hidden_rep_shape}.pkl")
    file_path_val = os.path.join(dir_name, f"combined_val_df_{hidden_rep_shape}.pkl")
    file_path_test = os.path.join(dir_name, f"combined_test_df_{hidden_rep_shape}.pkl")
    print(file_path_train)
    print(file_path_val)
    print(file_path_test)
    combined_trained_df.to_pickle(file_path_train)
    combined_val_df.to_pickle(file_path_val)
    combined_test_df.to_pickle(file_path_test)

    return combined_trained_df ,combined_val_df, combined_test_df ,best_param , run_name


