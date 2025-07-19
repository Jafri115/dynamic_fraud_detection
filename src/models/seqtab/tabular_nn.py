# src/models/seqtab/tabular_nn.py

import os
import json
import datetime
import pytz
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2


class TabularNNModel(tf.keras.Model):
    def __init__(self, input_dim, hidden_layers, dropout_rate=0.3, l2_lambda=0.01, name="TabularNNModel"):
        super().__init__(name=name)
        self.model = Sequential()
        self.model.add(Input(shape=(input_dim,)))

        for size in hidden_layers:
            self.model.add(Dense(size, kernel_regularizer=l2(l2_lambda), kernel_initializer='he_normal'))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(Dropout(dropout_rate))

        self.model.add(Dense(1, activation='sigmoid'))

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def get_hidden_representation_model(self):
        return Model(inputs=self.model.input, outputs=self.model.layers[-2].output)

    def evaluate_model(self, x_test, y_test):
        y_pred = self.predict(x_test)
        y_binary = (y_pred > 0.5).astype(int)

        return {
            "accuracy": accuracy_score(y_test, y_binary),
            "precision": precision_score(y_test, y_binary, zero_division=1),
            "recall": recall_score(y_test, y_binary, zero_division=1),
            "f1_score": f1_score(y_test, y_binary, zero_division=1),
            "roc_auc": roc_auc_score(y_test, y_binary),
            "average_precision": average_precision_score(y_test, y_binary),
            "classification_report": classification_report(y_test, y_binary, output_dict=True, zero_division=1)
        }


def save_model(model, x_train, y_train, data_type, params: dict, base_dir='./saved_models'):
    os.makedirs(base_dir, exist_ok=True)

    # Evaluate
    y_pred = model.predict(x_train)
    y_binary = (y_pred > 0.5).astype(int)

    scores = {
        "accuracy": accuracy_score(y_train, y_binary),
        "precision": precision_score(y_train, y_binary),
        "recall": recall_score(y_train, y_binary),
        "f1_score": f1_score(y_train, y_binary),
    }

    # Timestamped directory
    timestamp = datetime.datetime.now(pytz.timezone("Europe/Berlin")).strftime('%Y%m%d-%H%M%S')
    subdir = os.path.join(base_dir, data_type, f"{scores['accuracy']:.2f}__{timestamp}")
    os.makedirs(subdir, exist_ok=True)

    # Save model
    model_path = os.path.join(subdir, f"{data_type}__model")
    model.save(model_path, save_format='tf')

    # Save metrics
    metrics = {**scores, **params}
    with open(os.path.join(subdir, f"{data_type}__metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    return model_path


def save_hidden_representations(model, x_train, y_train, x_test, y_test, meta_train, meta_test,
                                 data_type, input_dim, latent_dim, out_dir='./data/processed/tabular/repr'):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now(pytz.timezone("Europe/Berlin")).strftime('%Y%m%d-%H%M%S')

    path = os.path.join(out_dir, f"{data_type}__{input_dim}__{latent_dim}__{timestamp}")
    os.makedirs(path, exist_ok=True)

    rep_model = model.get_hidden_representation_model()
    x_train_repr = rep_model.predict(x_train)
    x_test_repr = rep_model.predict(x_test)

    train_df = pd.DataFrame({
        "rep": list(x_train_repr),
        "meta1": meta_train[0],
        "meta2": meta_train[1],
        "label": y_train
    })

    test_df = pd.DataFrame({
        "rep": list(x_test_repr),
        "meta1": meta_test[0],
        "meta2": meta_test[1],
        "label": y_test
    })

    train_df.to_csv(os.path.join(path, "train_repr.csv"), index=False)
    test_df.to_csv(os.path.join(path, "test_repr.csv"), index=False)

    np.savez_compressed(os.path.join(path, "repr_data.npz"),
                        X_train=x_train_repr, y_train=y_train,
                        X_test=x_test_repr, y_test=y_test)

    return train_df, test_df
