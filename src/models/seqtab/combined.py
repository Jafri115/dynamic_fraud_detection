# src/models/seqtab/combined.py
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Concatenate
from tensorflow.keras import regularizers

from src.models.seqtab.transformer import TransformerTime
from src.models.seqtab.tabular_nn import TabularNNModel
from src.utils.helpers import CombinedCustomLoss


class CombinedModel(tf.keras.Model):
    def __init__(
        self, input_shape_tabular, max_len, max_code, combined_hidden_layers, dropout_rate_comb,
        dropout_rate_seq, droput_rate_tab, layer=None, tab_hidden_states=None,
        batch_size=None, n_event_code=273042, is_public_dataset=False,
        l2_lambda_comb=0.01, l2_lambda_tab=0.01, l2_lambda_seq=0.01, model_dim=256
    ):
        super(CombinedModel, self).__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.max_code = max_code
        self.n_event_code = n_event_code

        # Store initialization parameters for serialization
        self.input_shape_tabular = input_shape_tabular
        self.combined_hidden_layers = combined_hidden_layers
        self.dropout_rate_comb = dropout_rate_comb
        self.dropout_rate_seq = dropout_rate_seq
        self.droput_rate_tab = droput_rate_tab
        self.layer = layer
        self.tab_hidden_states = tab_hidden_states
        self.is_public_dataset = is_public_dataset
        self.l2_lambda_comb = l2_lambda_comb
        self.l2_lambda_tab = l2_lambda_tab
        self.l2_lambda_seq = l2_lambda_seq
        self.model_dim = model_dim
        self.bce_loss_tab = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.bce_loss_seq = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.total_loss = CombinedCustomLoss()

        self.tabular_model = TabularNNModel(
            input_dim=input_shape_tabular,
            hidden_layers=tab_hidden_states,
            dropout_rate=droput_rate_tab,
            l2_lambda=l2_lambda_tab
        )
        self.tabular_output = Dense(1, activation='sigmoid', name='tabular_output')

        self.sequential_model = TransformerTime(
            n_event_code, batch_size, max_len, layer, dropout_rate_seq,
            is_public_dataset, l2_lambda_seq, model_dim
        )
        self.sequential_output = Dense(1, activation='sigmoid', name='sequential_output')

        self.concatenate = Concatenate()
        self.dense_layers = [
            Dense(units, activation='relu', name=f'combined_layer{units}',
                  kernel_regularizer=regularizers.l2(l2_lambda_comb))
            for units in combined_hidden_layers
        ]
        self.dropout = Dropout(dropout_rate_comb)
        self.final_output = Dense(1, activation='sigmoid', name='final_output')

    def call(self, inputs, training=False, return_representation=False, **train_flags):
        tabular_rep, seq_rep = self.build_representation(inputs, training, train_flags)

        # Handle the case when train_flags is empty (e.g., during model saving)
        if not train_flags or all(not v for v in train_flags.values()):
            # Default behavior based on available representations
            if tabular_rep is not None and seq_rep is not None:
                x = self.concatenate([tabular_rep, seq_rep])
            elif tabular_rep is not None:
                x = tabular_rep
            elif seq_rep is not None:
                x = seq_rep
            else:
                raise ValueError("No valid representation found")
        elif train_flags.get('train_combined'):
            x = self.concatenate([tabular_rep, seq_rep])
        elif train_flags.get('train_tab'):
            x = tabular_rep
        elif train_flags.get('train_seq'):
            x = seq_rep
        else:
            raise ValueError("Invalid mode")

        for layer in self.dense_layers:
            x = layer(x)
            x = self.dropout(x, training=training)

        if return_representation:
            return x, (tabular_rep if tabular_rep is not None else seq_rep)

        return self.final_output(x)

    
    def build_representation(self, inputs, training, train_flags):
        # Handle the case when train_flags is empty (e.g., during model saving)
        if not train_flags or all(not v for v in train_flags.values()):
            # Default to combined mode if we have multiple inputs, otherwise tabular
            if isinstance(inputs, (list, tuple)) and len(inputs) > 1:
                tabular_input, *seq_inputs = inputs
                tabular_rep = self.tabular_model(tabular_input, training=training)
                seq_rep = self.sequential_model(seq_inputs, training=training)
                return tabular_rep, seq_rep
            else:
                # Single input, assume tabular
                return self.tabular_model(inputs, training=training), None
        
        if train_flags.get('train_combined'):
            tabular_input, *seq_inputs = inputs
            tabular_rep = self.tabular_model(tabular_input, training=training)
            seq_rep = self.sequential_model(seq_inputs, training=training)
            return tabular_rep, seq_rep

        if train_flags.get('train_tab'):
            return self.tabular_model(inputs, training=training), None

        if train_flags.get('train_seq'):
            return None, self.sequential_model(inputs, training=training)

        raise ValueError("Invalid train flags")


    def get_representation(self, inputs, training=False, train_flags=None):
        if train_flags is None:
            train_flags = {}
        return self.call(inputs, training=training, return_representation=True, **train_flags)

    def predict_from_representation(self, inputs, training=False, train_flags=None):
        if train_flags is None:
            train_flags = {}
        return self.call(inputs, training=training, return_representation=False, **train_flags)

    def get_config(self):
        return {
            "input_shape_tabular": self.input_shape_tabular,
            "max_len": self.max_len,
            "max_code": self.max_code,
            "combined_hidden_layers": self.combined_hidden_layers,
            "dropout_rate_comb": self.dropout_rate_comb,
            "dropout_rate_seq": self.dropout_rate_seq,
            "droput_rate_tab": self.droput_rate_tab,
            "layer": self.layer,
            "tab_hidden_states": self.tab_hidden_states,
            "batch_size": self.batch_size,
            "n_event_code": self.n_event_code,
            "is_public_dataset": self.is_public_dataset,
            "l2_lambda_comb": self.l2_lambda_comb,
            "l2_lambda_tab": self.l2_lambda_tab,
            "l2_lambda_seq": self.l2_lambda_seq,
            "model_dim": self.model_dim,
        }

    @classmethod
    def from_config(cls, config):
        # Filter out parameters that aren't part of the model constructor
        valid_params = {
            'input_shape_tabular', 'max_len', 'max_code', 'combined_hidden_layers',
            'dropout_rate_comb', 'dropout_rate_seq', 'droput_rate_tab', 'layer',
            'tab_hidden_states', 'batch_size', 'n_event_code', 'is_public_dataset',
            'l2_lambda_comb', 'l2_lambda_tab', 'l2_lambda_seq', 'model_dim'
        }
        
        # Filter config to only include valid model parameters
        filtered_config = {k: v for k, v in config.items() if k in valid_params}
        
        return cls(**filtered_config)
