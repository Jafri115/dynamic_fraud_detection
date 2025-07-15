# load_datasets.py
# Standard library imports
from datetime import datetime
import os
import pickle
import warnings

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split



# Suppress warnings
warnings.filterwarnings("ignore")

# TensorFlow and MLflow configurations
tf.config.run_functions_eagerly(False)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def load_ehr_data():
    base_path = 'data/Datasets/publicDataset/EHR_dataset/'
    data = pickle.load(open(base_path + 'hf_sample_training_new.pickle', 'rb'))

    merged_train_df = pd.DataFrame({
        'EVENT_SEQUENCE': list(data[0]),
        'TIME_DIFF_ORDER': list(data[2]),
        'EVENT_FAILURE_SYSTEM_BIT':list([[0] * len(data[0][i]) for i in range(len(data[0]))]) ,# List comprehension to create a list of lists
        'EVENT_FAILURE_USER_BIT': list([[0] * len(data[0][i]) for i in range(len(data[0]))]), 
        # 'order_envelope_id': merged_train_df['ORDER_ENVELOPE_ID'].to_numpy(),
        'label': list(data[1]),
        })
    data = pickle.load(open(base_path + 'hf_sample_validation_new.pickle', 'rb'))
    combined_representation_val_df = pd.DataFrame({
        'EVENT_SEQUENCE': list(data[0]),
        'TIME_DIFF_ORDER': list(data[2]),
        'EVENT_FAILURE_SYSTEM_BIT':list([[0] * len(data[0][i]) for i in range(len(data[0]))]) ,# List comprehension to create a list of lists
        'EVENT_FAILURE_USER_BIT': list([[0] * len(data[0][i]) for i in range(len(data[0]))]), 
        # 'order_envelope_id': merged_train_df['ORDER_ENVELOPE_ID'].to_numpy(),
        'label': list(data[1]),
        })
    
    merged_train_df_val = pd.concat([merged_train_df, combined_representation_val_df], ignore_index=True)

    data = pickle.load(open(base_path + 'hf_sample_testing_new.pickle', 'rb'))
    merged_test_df = pd.DataFrame({
    'EVENT_SEQUENCE': list(data[0]),
    'TIME_DIFF_ORDER': list(data[2]),
    'EVENT_FAILURE_SYSTEM_BIT':list([[0] * len(data[0][i]) for i in range(len(data[0]))]) ,# List comprehension to create a list of lists
    'EVENT_FAILURE_USER_BIT': list([[0] * len(data[0][i]) for i in range(len(data[0]))]), 
    # 'order_envelope_id': merged_train_df['ORDER_ENVELOPE_ID'].to_numpy(),
    'label': list(data[1]),
        })
    output_label_name = 'label'
    selected_col = ['EVENT_SEQUENCE','TIME_DIFF_SEQUENCE']
    
    return merged_train_df,combined_representation_val_df,merged_test_df,selected_col,output_label_name

def load_wiki_data():
    base_path = 'data/Datasets/publicDataset/wiki/vews_dataset_v1.1/'
    merged_train_df = pd.read_pickle(base_path + 'user_edits_train.pkl')
    merged_val_df = pd.read_pickle(base_path + 'user_edits_test.pkl')
    merged_test_df = pd.read_pickle(base_path + 'user_edits_test.pkl')
    
    output_label_name = 'LABEL'
    selected_col = ['total_edits', 'unique_pages', 'avg_stiki_score',
       'cluebot_revert_count', 'edit_frequency', 'night_edits', 'day_edits',
       'weekend_edits', 'page_category_diversity']
    
    return merged_train_df,merged_val_df,merged_test_df,selected_col,output_label_name

# Define a function to print the specs for each dataset
def print_dataset_specs(df, dataset_name, output_label_name='SEQ_ORDER_LABEL'):
    print(f"Specifications for {dataset_name}:")
    print(f"Total entries: {df.shape[0]}")
    if output_label_name == '':

        fruad_detection = df[(df['SEQ_ORDER_LABEL'] == 1) & (df['TAB_ORDER_LABEL'] == 1)]['SEQ_ORDER_LABEL'].count()
        fraud_protection = df[(df['SEQ_ORDER_LABEL'] == 1) & (df['TAB_ORDER_LABEL'] == 0)]['SEQ_ORDER_LABEL'].count()
        non_fraud = df[(df['SEQ_ORDER_LABEL'] == 0) & (df['TAB_ORDER_LABEL'] == 0)]['SEQ_ORDER_LABEL'].count()

        print(f"Fraud Detection: {fruad_detection}")
        print(f"Fraud Protection: {fraud_protection}")
        print(f"Non Fraud: {non_fraud}")
        print(f"Fraud Detection Percentage: {fruad_detection/df.shape[0]}")
        print(f"Fraud Protection Percentage: {fraud_protection/df.shape[0]}")
        print(f"Non Fraud Percentage: {non_fraud/df.shape[0]}")
        print(f'total cases sum: {fruad_detection+fraud_protection+non_fraud}')
    value_counts = df[output_label_name].value_counts()
    print("Value counts of 'label':")
    print(value_counts)
    percentage_of_1 = (value_counts.get(1, 0) / df.shape[0]) * 100
    print(f"Percentage of 'label' == 1: {percentage_of_1:.2f}%")
    print()  # Blank line for better readability



def load_ccfraud_data():

    merged_train_df = pd.read_csv('data/Datasets/publicDataset/credit_card/creditcard.csv')
    rob_scaler = RobustScaler()

    merged_train_df['scaled_amount'] = rob_scaler.fit_transform(merged_train_df['Amount'].values.reshape(-1,1))
    merged_train_df['scaled_time'] = rob_scaler.fit_transform(merged_train_df['Time'].values.reshape(-1,1))

    merged_train_df.drop(['Time','Amount'], axis=1, inplace=True)
    scaled_amount = merged_train_df['scaled_amount']
    scaled_time = merged_train_df['scaled_time']

    merged_train_df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    merged_train_df.insert(0, 'scaled_amount', scaled_amount)
    merged_train_df.insert(1, 'scaled_time', scaled_time)
    X = merged_train_df.drop('Class', axis=1)
    y = merged_train_df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    merged_train_df = pd.concat([X_train, y_train], axis=1)
    merged_test_df = pd.concat([X_test, y_test], axis=1)

    selected_col = ["scaled_time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","scaled_amount"]
    output_label_name = 'Class'
    return merged_train_df,merged_test_df,selected_col,output_label_name