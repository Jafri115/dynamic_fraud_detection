# TABULAR_MODEL_NN.py
import os
import datetime
import pytz
import json
import numpy as np
import sys
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score,average_precision_score,roc_auc_score
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
class ClassificationModel(tf.keras.Model):

    def __init__(self, input_size, hidden_layer_sizes, dropout_rate, l2_lambda,name =None):
        super(ClassificationModel, self).__init__(name=name)
        self.model = Sequential()
        self.model.add(Input(shape=(input_size,)))
        for size in hidden_layer_sizes:
            self.model.add(Dense(size, kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda),name='Dense_tab'+str(size)))
            self.model.add(BatchNormalization())  # Batch Normalization layer
            self.model.add(Activation('relu'))
            self.model.add(Dropout(dropout_rate))


    def call(self, inputs, training = True): 
        return self.model(inputs)
    
    def get_last_hidden_layer_representation(self):
        last_hidden_layer = self.model.layers[-2].output
        hidden_representation_model = Model(inputs=self.model.input, outputs=last_hidden_layer)
        return hidden_representation_model
    
    def evaluate_model(self, x_test, y_test):
        y_pred = self.predict(x_test)
        if y_pred.shape[1] > 1:
            y_pred_binary = np.argmax(y_pred, axis=1)
        else:
            y_pred_binary = (y_pred > 0.5).astype(int)

        F1score = f1_score(y_test, y_pred_binary, zero_division=1)
        roc_Auc_score = roc_auc_score(y_test, y_pred_binary)
        auc_pr = average_precision_score(y_test, y_pred_binary)
        acc_score = accuracy_score(y_test, y_pred_binary)
        class_rep_dic = classification_report(y_test, y_pred_binary, output_dict=True, zero_division=1)

        return F1score, roc_Auc_score, auc_pr, acc_score, class_rep_dic
    
def save_model(model, data_type, param_dict, X_train, y_train, base_path='./saved_models'):
    
    os.makedirs(base_path, exist_ok=True)
        
    # Calculate metrics
    y_pred = model.predict(X_train)
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_train, y_pred_binary)
    precision = precision_score(y_train, y_pred_binary)
    recall = recall_score(y_train, y_pred_binary)
    f1 = f1_score(y_train, y_pred_binary)

    # Create a timestamp
    germany_tz = pytz.timezone('Europe/Berlin')
    germany_time = datetime.datetime.now(germany_tz)
    timestamp = germany_time.strftime('%Y%m%d-%H%M%S')
    
    # Create the data type specific path
    data_type_path = os.path.join(base_path, data_type)
    data_time_path = os.path.join(data_type_path, f"{accuracy:.2f}__{timestamp}")

    # Ensure the data type specific path exists
    os.makedirs(data_time_path, exist_ok=True)

    # Create a filename for model and metrics
    model_filename = f"D_{data_type}__{timestamp}"
    metrics_filename = f"D_{data_type}__{timestamp}__metrics.json"
    
    model_filepath = os.path.join(data_time_path, model_filename)
    metrics_filepath = os.path.join(data_time_path, metrics_filename)

    # Save the model
    model.save(model_filepath, save_format="tf")

    # Save the metrics in a JSON file
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    metrics.update(param_dict)

    with open(metrics_filepath, 'w') as f:
        json.dump(metrics, f)
    
    return model_filepath, metrics_filepath

def create_combined_dataframe(X, y, meta_data):


    meta_data = pd.DataFrame(meta_data)

    activity_hid_rep_df = pd.DataFrame(
        {
            'tabular_hidden':list(X),
            'first_shield_id':list(meta_data[0]) ,
            'customer_hash': list(meta_data[1]),
            'label':list(y)
            })

    return activity_hid_rep_df


def save_representation(model,X_train,y_train,X_test, y_test,meta_data_train,meta_data_test ,data_type, input_dim ,laten_dim, base_path='./data/processed_data/tab_data_representations_nn'):
    

    X_ben_train= model.predict(X_train)
    X_test = model.predict(X_test)

    # Create a timestamp
    germany_tz = pytz.timezone('Europe/Berlin')
    germany_time = datetime.datetime.now(germany_tz)
    timestamp = germany_time.strftime('%Y%m%d-%H%M%S')

    # create the data type specific path
    os.makedirs(base_path, exist_ok=True)
    file_path = os.path.join(base_path, f"{data_type}__{input_dim}__{laten_dim}__{timestamp}")
    os.makedirs(file_path, exist_ok=True)
    repre_filename =  f"{data_type}__{input_dim}__{laten_dim}__{timestamp}.npz"
    model_filepath = os.path.join(file_path, repre_filename)

    combined_train_df = create_combined_dataframe(X_ben_train, y_train, meta_data_train)
    combined_test_df = create_combined_dataframe(X_test, y_test, meta_data_test)

    combined_train_df.to_csv(os.path.join(file_path, f"{data_type}__{input_dim}__{laten_dim}__{timestamp}__train.csv"))
    combined_test_df.to_csv(os.path.join(file_path, f"{data_type}__{input_dim}__{laten_dim}__{timestamp}__test.csv"))   

    np.savez(model_filepath, X_ben_train=X_ben_train, y_ben_train=y_train, X_test=X_test, y_test=y_test)
    
    return combined_train_df,combined_test_df

