# main_utils.py
from data.Datasets.publicDataset.load_datasets import  load_wiki_data
import time
import tensorflow as tf
import os
import pickle
import numpy as np

from utils.utils import  get_git_commit_hash


def get_tag_name(tag_activity):
    if tag_activity['session_changed'] and tag_activity['city_changed']:
        tag_name = 'session_city_changed'
    elif tag_activity['city_changed']:
        tag_name = 'city_changed'
    elif tag_activity['session_changed']:
        tag_id = 'session_changed'
    else:
        tag_name = 'no_session_city_changed'
    return tag_name + '-1_3_days_det_pro_non_neg_non_pay_mul_y_prev'


def load_data_and_setup_parameters(load_from_disk, tag_name, tag_activity, dataset):


    data = load_wiki_data()
    dict_file = 'data\Datasets\publicDataset\wiki\category_dict.pkl'

    merged_train_df, merged_val_df, merged_test_df, selected_col, output_label_name = data
    code2id = pickle.load(open(dict_file, 'rb'))
    n_event_code = len(code2id) + 1
    return merged_train_df, merged_val_df, merged_test_df, n_event_code, selected_col, output_label_name

def prepare_data(df_train, df_val, df_test, rep_layer_name):
    
    
    def extract_features_and_labels(df):
        X = np.vstack(df[rep_layer_name].to_numpy())
        y = tf.one_hot(df['label'].to_numpy(), depth=2).numpy().squeeze()
        return X, y

    # Extract benign (class 0) samples for train, validation, and test
    X_ben_train, y_ben_train = extract_features_and_labels(df_train[df_train['label'] == 0])
    X_ben_val, y_ben_val = extract_features_and_labels(df_val[df_val['label'] == 0])
    X_ben_test, y_ben_test = extract_features_and_labels(df_test[df_test['label'] == 0])

    # Extract full validation and test sets
    X_val, y_val = extract_features_and_labels(df_val)
    X_test, y_test = extract_features_and_labels(df_test)

    return X_ben_train, y_ben_train, X_ben_val, y_ben_val, X_ben_test, y_ben_test, X_val, y_val, X_test, y_test


def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size=512, run_name=None, discriminator=None):
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

    if discriminator:
        model_path = f'saved_models/pre_trained_disc/{run_name}'
        if not os.path.exists(model_path):
            model.discriminator_tar = discriminator
            model.discriminator_tar.compile(optimizer=model.discriminator_tar_optimizer, metrics=['accuracy'])
            model.discriminator_tar.fit(train_data, val_data, epochs=epochs)
            model.discriminator_tar.save(model_path, save_format='tf')
        else:
            model.discriminator_tar = tf.keras.models.load_model(model_path)

    history = model.fit(train_data, (X_val, y_val), epochs=epochs)
    return history


def log_results(mlflow, eval_result, best_params_ph2, best_params_ph1_log, epochs, best_checkpoint_path, start_time, tag_name):
    with mlflow.start_run() as run:
        mlflow.log_metrics({
            "roc_Auc_score": eval_result['roc_auc_score'],
            "F1score": eval_result['F1_score'],
            "avg_prec": eval_result['average_precision_score'],
            "accuracy": eval_result['accuracy'],
            "recall": eval_result['recall'],
            "precision": eval_result['precision'],
            "Tp": eval_result['TruePositives'],
            "Tn": eval_result['TrueNegatives'],
            "Fp": eval_result['FalsePositives'],
            "Fn": eval_result['FalseNegatives']
        })
        mlflow.log_params({**best_params_ph2, **best_params_ph1_log, 'epochs': epochs, "checkpoint_location": best_checkpoint_path})
        mlflow.set_tags({
            'git_commit_hash': get_git_commit_hash(),
            'time_elapsed': (time.time() - start_time) / 60,
            'activity': 'Fail_bits only',
            'tag_name': tag_name
        })
        run_name = run.info.run_name
    return run_name


def train_ocan_model(ocan_model, X_ben_train, y_ben_train, X_val, y_val,X_ben_val, y_ben_val,PHASE2_TRAIN_EPOCHS,run_name,PHASE_2_PRE_TRAIN_EPOCHS=10,batch_size=512,discriminator_tar=None):

    train_data = tf.data.Dataset.from_tensor_slices((X_ben_train, y_ben_train)).batch(batch_size)
    val_data_tar = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
    test_data = (X_val, y_val)
    base_path ='saved_models/pre_trained_disc'
    if(discriminator_tar is not None):
            
            base_path ='saved_models/pre_trained_disc/'+str(run_name)+'/'
            ocan_model.discriminator_tar = discriminator_tar
    elif not os.path.exists(os.path.join(base_path,str(run_name))):
        train_data_tar = tf.data.Dataset.from_tensor_slices((X_ben_train, y_ben_train)).batch(batch_size)

        # Compile the model
        ocan_model.discriminator_tar.compile(optimizer=ocan_model.discriminator_tar_optimizer,
                    metrics=['accuracy'])

        history_pre = ocan_model.discriminator_tar.fit_pretrained_disc(train_data_tar,val_data_tar, epochs=PHASE_2_PRE_TRAIN_EPOCHS,optimizer=ocan_model.discriminator_tar_optimizer)
        ocan_model.discriminator_tar.save(base_path, save_format='tf')
    else:
        
        base_path ='saved_models/pre_trained_disc/'+str(run_name)+'/'
        ocan_model.discriminator_tar = tf.keras.models.load_model(base_path)

    history = ocan_model.fit(train_data, test_data, epochs=PHASE2_TRAIN_EPOCHS)

    return history