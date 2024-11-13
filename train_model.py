# Standard library imports
import os
import time
import warnings

# Third-party imports
import numpy as np
import pandas as pd
import tensorflow as tf

# Local application/library specific imports
from models.Combined_rep_MODEL.train_evaluate_combined_model import train_representation_learning
from models.OCAN_Baseline.oc_gan_1 import GANModel
from utils.main_utils import get_tag_name, load_data_and_setup_parameters, log_results, prepare_data, train_ocan_model
from utils.utils import get_run_by_runname, print_box_with_header

# Suppress warnings
warnings.filterwarnings("ignore")

# TensorFlow and MLflow configurations
import mlflow
import sys

# Set MLflow configurations
mlflow.set_tracking_uri('http://127.0.0.1:5005')
NUM_THREADS = 96  # or try 96 if hyper-threading proves beneficial

# Global Variables
if len(sys.argv) > 1:
    dataset = int(sys.argv[1])  # Dataset number or name
    load_from_disk = int(sys.argv[2])
    train_representation_phase1 = int(sys.argv[3])
    train_OCAN_phase2 = int(sys.argv[4])
    TRAIN_COMBINED = 0
    TRAIN_TABULAR = 1  # Default to tabular model
    TRAIN_SEQUENCE = 0
else:
    dataset = 1
    load_from_disk = 1
    train_representation_phase1 = 1
    train_OCAN_phase2 = 0
    TRAIN_COMBINED = 0
    TRAIN_TABULAR = 1
    TRAIN_SEQUENCE = 0

# Constants and configurations
PHASE1_TRAIN_EPOCHS = 1
PHASE2_TRAIN_EPOCHS = 1
LOAD_PRETRAINED_DISCRIMINATOR = 1

TAG_ACTIVITY = {'session_changed': 1, 'city_changed': 1}

def Train_Phase1(train_representation_phase1, phase1_train_run_name, tag_name, dataset_name):
    experiment_name = f'fdt_{dataset_name}_ph1_'
    
    if train_representation_phase1 == 1:
        train_flags, resample_dict = {'train_combined': TRAIN_COMBINED, 'train_tab': TRAIN_TABULAR, 'train_seq': TRAIN_SEQUENCE}, {'smote': 0, 'rus': 0, 'ns': 1}
        freez_layers = {'seq': 0, 'tab': 0, 'no_freez': 1, 'both_freeze': 0}

        # Load data and set up other parameters dynamically
        merged_train_df, merged_val_df, merged_test_df, n_event_code, selected_col, output_label_name = load_data_and_setup_parameters(load_from_disk, tag_name, TAG_ACTIVITY, dataset)

        print("Loaded Data and Configured Training Environment")

        # Predefined best parameters for Phase 1 (no tuning)
        best_param_ph1 = {
            'num_of_tabular_columns': len(selected_col),
            'combined_hidden_layers': [128, 64], 
            "tab_hidden_states": [32],
            "max_len": 50,
            "dropout_rate_comb": 0.01934845312006321,
            "dropout_rate_seq": 0.5,
            "droput_rate_tab": 0.2955191314359026,
            "learning_rate": 1e-4, 
            "layer" : 1,
            "weight_decay": 6.925306775722288e-07,
            "l2_lambda_comb": 1.5543542930847382e-07,
            "l2_lambda_tab": 9.69978877385772e-07,
            "l2_lambda_seq": 1e-3,
            "model_dim": 256
        }

        # Train the model
        combined_trained_df, combined_val_df, combined_test_df, best_param_phase1, run_name = train_representation_learning(
            dataset_name, merged_train_df, merged_val_df, merged_test_df, selected_col, output_label_name, n_event_code, best_param_ph1,
            train_flags, resample_dict, experiment_name, PHASE1_TRAIN_EPOCHS,0, train_flags['train_combined'], freez_layers, True, tag_name
        )

        return combined_trained_df, combined_val_df, combined_test_df, best_param_phase1, run_name

    else:
        best_params_ph1_log = get_run_by_runname(experiment_name + '_tr', phase1_train_run_name)
        base_path = f'data/processed_data/{dataset_name}/Phase1_latent_representations/{str(phase1_train_run_name)}/'
        hidden_rep_shape = best_params_ph1_log['ph1_params.combined_hidden_layers'][-1]

        combined_trained_df = pd.read_pickle(base_path + f'combined_trained_df_{hidden_rep_shape}.pkl')
        combined_val_df = pd.read_pickle(base_path + f'combined_val_df_{hidden_rep_shape}.pkl')
        combined_test_df = pd.read_pickle(base_path + f'combined_test_df_{hidden_rep_shape}.pkl')

        return combined_trained_df, combined_val_df, combined_test_df, best_params_ph1_log, phase1_train_run_name

if __name__ == "__main__":

    phase1_train_run_name = 'bouncy-ant-73'
    dataset_name = 'wiki'  # Set the dataset name dynamically here (or use sys.argv to pass it)
    experiment_name_phase2 = f'fdt_{dataset_name}_ph2_'
    tag_name = get_tag_name(TAG_ACTIVITY)

    combined_trained_df, combined_val_df, combined_test_df, best_params_ph1_log, run_name = Train_Phase1(train_representation_phase1, phase1_train_run_name, tag_name, dataset_name)

    # Predefined Phase 2 parameters (no hyperparameter tuning)
    best_params_ph2 = {
        'dim_inp': combined_trained_df['hidden_rep'].to_numpy()[0].shape[0],
        'batch_size': 256,  # Example predefined batch size
        'epochs': PHASE2_TRAIN_EPOCHS , # Using predefined epochs for training,
        "G_D_layers": [[50, 100], [100, 50]],  
        "d_lr": 4e-05,
        "g_lr": 0.0009,
        "mb_size": 256,  
        "Z_dim": 100,     
        "dim_inp": 64,   
        "g_dropouts": [0.35, 0.25],
        "d_dropouts": [0.45, 0.45],
        "batch_norm_g": 1,
        "batch_norm_d": 0,       
        "beta1_g": 0.45,            
        "beta2_g": 0.990,          
        "beta1_d": 0.5,            
        "beta2_d": 0.999,
        "lambda_pt": 1,
        "lambda_gp": 4,
        "lambda_fm": 1,
        "lambda_ent": 1
    }

    mlflow.set_experiment(experiment_name_phase2 + '_tr')
    print_box_with_header('TRAINING PHASE 2 OCAN GANs')
    print_box_with_header('params set')
    print(best_params_ph2)
    rep_layer_name = 'hidden_rep'

    X_ben_train, y_ben_train, X_ben_val, y_ben_val, X_ben_test, y_ben_test, X_val, y_val, X_test, y_test = prepare_data(combined_trained_df, combined_val_df, combined_test_df, rep_layer_name)


    base_dir = 'saved_models/Phase2/'
    current_time = time.strftime("%Y%m%d-%H%M%S")
    check_point_dir = os.path.join(base_dir, run_name, current_time)

    if not os.path.exists(check_point_dir):
        os.makedirs(check_point_dir)
        print(f"Directory created at {check_point_dir}")
    else:
        print(f"Directory already exists at {check_point_dir}")

    filename = base_dir + time.strftime("%Y%m%d-%H%M%S") + '_combined_model.h5'
    start_time = time.time()

    ocan_model = GANModel(best_params_ph2, X_ben_train.shape[0], check_point_dir)

    history = train_ocan_model(
        ocan_model, X_ben_train, y_ben_train, X_val, y_val, X_ben_val, y_ben_val, PHASE2_TRAIN_EPOCHS, run_name
    )

    best_checkpoint_path = ocan_model.ckpt_manager.latest_checkpoint
    if best_checkpoint_path:
        ocan_model.checkpoint.restore(best_checkpoint_path)
        print(f"Restored model from best checkpoint: {best_checkpoint_path}")

    evalation_result = ocan_model.evaluate_model(X_test, y_test)

    print("Training and evaluation complete.")
