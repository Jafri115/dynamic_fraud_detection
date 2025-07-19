# src/utils/helpers.py
import os
import numpy as np
import tensorflow as tf
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


class CombinedCustomLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(CombinedCustomLoss, self).__init__()
        self.tabular_loss = 0
        self.sequence_loss = 0
        # Use BinaryCrossentropy for binary classification tasks
        self.bce_tab = tf.keras.losses.BinaryCrossentropy()
        self.bce_seq = tf.keras.losses.BinaryCrossentropy()
        self.bce_comb = tf.keras.losses.BinaryCrossentropy()

    def call(self, y_true, y_pred):
        y_tab, y_seq = y_true

        y_tab_pred, y_seq_pred ,y_combined_pred = y_pred 
        
        # Ensure predictions are squeezed to match label shape
        y_tab_pred = tf.squeeze(y_tab_pred, axis=-1) 
        y_seq_pred = tf.squeeze(y_seq_pred, axis=-1)
        y_combined_pred = tf.squeeze(y_combined_pred, axis=-1)

        self.tabular_loss = self.bce_tab(y_tab, y_tab_pred)
        self.sequence_loss = self.bce_seq(y_seq, y_seq_pred)

        loss_combined = self.bce_comb(y_seq, y_combined_pred)
        total_loss = self.tabular_loss + self.sequence_loss + loss_combined

        return total_loss
    
