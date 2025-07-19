# src/pipeline/predict_pipeline.py
"""Utilities for making predictions with the trained phase 1 model."""

import os
from typing import Tuple, Dict, Union
import numpy as np
import pandas as pd
import tensorflow as tf

from src.components.data_transformation import WikiCombinedTransformation
from src.utils.io import load_object
from src.models.seqtab.combined import CombinedModel
from src.models.seqtab.tabular_nn import TabularNNModel
from src.models.seqtab.transformer import TransformerTime
from src.models.seqtab.units import pad_matrix, pad_time, pad_failure_bits
from src.config.configuration import PredictionConfig


class Phase1PredictPipeline:
    """Pipeline for making predictions using the phase 1 combined model."""

    def __init__(
        self,
        model_dir=None,
        config_path=None,
        preprocessor_path=None,
        category_dict_path=None,
    ):
        import json
        
        pred_config = PredictionConfig()
        
        model_dir = model_dir or pred_config.model_dir
        config_path = config_path or pred_config.config_path
        preprocessor_path = preprocessor_path or pred_config.preprocessor_path
        category_dict_path = category_dict_path or pred_config.category_dict_path

        self.selected_indices = None

        try:
            config = json.load(open(config_path))
            self.max_len = config["max_len"]
            self.max_code = config["max_code"]
            self.input_shape_tabular = config["input_shape_tabular"]
            self.combined_hidden_layers = config["combined_hidden_layers"]
            self.dropout_rate_comb = config["dropout_rate_comb"]
            self.dropout_rate_seq = config["dropout_rate_seq"]
            self.droput_rate_tab = config["droput_rate_tab"]
            self.tab_hidden_states = config["tab_hidden_states"]
            self.n_event_code = config["n_event_code"]
            self.model_dim = config.get("model_dim", 256)
            self.layer = config.get("layer", 1)
            self.selected_indices = config.get("selected_indices")
            self.is_public_dataset = config.get("is_public_dataset", False)
        except FileNotFoundError:
            self.max_len = pred_config.default_max_len
            self.max_code = pred_config.default_max_code
            self.input_shape_tabular = pred_config.default_input_shape_tabular
            self.combined_hidden_layers = pred_config.default_combined_hidden_layers
            self.dropout_rate_comb = pred_config.default_dropout_rate_comb
            self.dropout_rate_seq = pred_config.default_dropout_rate_seq
            self.droput_rate_tab = pred_config.default_dropout_rate_tab
            self.tab_hidden_states = pred_config.default_tab_hidden_states
            self.n_event_code = pred_config.default_n_event_code
            self.model_dim = pred_config.default_model_dim
            self.layer = pred_config.default_layer
            self.is_public_dataset = pred_config.default_is_public_dataset

        try:
            self.preprocessor = load_object(preprocessor_path)
        except FileNotFoundError:
            self.preprocessor = None

        try:
            self.category_dict = load_object(category_dict_path)
        except FileNotFoundError:
            self.category_dict = {"no_cat": self.n_event_code}

        self.transformer = WikiCombinedTransformation()

        # Try to load the trained model if available, otherwise use a dummy
        self.model = None
        if os.path.exists(model_dir):
            try:
                self.model = tf.keras.models.load_model(
                    model_dir,
                    custom_objects={
                        "CombinedModel": CombinedModel,
                        "TabularNNModel": TabularNNModel,
                        "TransformerTime": TransformerTime,
                    },
                    compile=False,
                )
            except Exception as e:
                print(f"Could not load model from {model_dir}: {e}")

        if self.model is None:
            class DummyModel:
                def predict_from_representation(self_inner, inputs, training=False, train_flags=None):
                    n = inputs[0].shape[0]
                    return tf.zeros((n, 1), dtype=tf.float32)

            self.model = DummyModel()
        




    

    def _prepare_tabular(self, df: pd.DataFrame) -> np.ndarray:
        df = self.transformer.compute_time_deltas(df)
        features, _ = self.transformer.extract_tabular_features(df, label_required=False)

        if self.preprocessor is not None:
            features = self.preprocessor.transform(features)
        else:
            features = features.to_numpy()

        # Apply feature selection after preprocessing
        if self.selected_indices is not None:
            features = features[:, self.selected_indices]

        return features.astype(np.float32)

    def _prepare_sequence(self, df: pd.DataFrame) -> Tuple[np.ndarray, ...]:
        event_seq = [self.transformer.impute_sequence(s) + [[self.n_event_code - 1]]
                     for s in df["edit_sequence"]]
        time_seq = [self.transformer.impute_sequence(s) + [0] for s in df["time_delta_seq"]]
        # Use default values for failure sequences since we don't have revert data
        failure_sys = [[0] * len(self.transformer.impute_sequence(s)) + [1] for s in df["edit_sequence"]]
        failure_user = [[0] * len(self.transformer.impute_sequence(s)) + [1] for s in df["edit_sequence"]]

        seq_pad, mask, mask_final, mask_code = pad_matrix(event_seq, self.max_len, self.n_event_code, self.max_code, pad_token=self.n_event_code)
        time_pad = pad_time(time_seq, max_len=self.max_len)
        fail_sys_pad = pad_failure_bits(failure_sys, max_len=self.max_len)
        fail_user_pad = pad_failure_bits(failure_user, max_len=self.max_len)
        lengths = np.array([len(s) for s in event_seq], dtype=np.float32)
        return seq_pad, time_pad, fail_sys_pad, fail_user_pad, mask, mask_final, mask_code, lengths

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, ...]:
        x_tab = self._prepare_tabular(df)
        seq_pad, time_pad, fail_sys_pad, fail_user_pad, mask, mask_final, mask_code, lengths = self._prepare_sequence(df)
        return (
            x_tab,
            seq_pad,
            time_pad,
            fail_sys_pad,
            fail_user_pad,
            mask,
            mask_final,
            mask_code,
            lengths,
        )

    def predict(self, df: pd.DataFrame) -> Dict[str, Union[np.ndarray, list]]:
        inputs = self.preprocess(df)
        preds = self.model.predict_from_representation(inputs, training=False, train_flags={"train_combined": True})
        probabilities = preds.numpy().ravel()
        
        # Convert to percentages and add labels
        probability_percentages = probabilities * 100
        labels = ['fraud' if prob >= 0.5 else 'legitimate' for prob in probabilities]
        
        return {
            'probability_percentages': probability_percentages,
            'labels': labels,
            'raw_probabilities': probabilities
        }


if __name__ == "__main__":


    # Example usage for manual testing. This block is executed only when the
    # module is run as a script, preventing side effects on import.
    sample_data = {
        "edit_sequence": [[[1], [0], [0]]],
        "rev_time": [["2024-01-01T12:00:00Z", "2024-01-01T12:02:00Z", "2024-01-01T12:03:00Z"]],
        "total_edits": [3]
    }


    df = pd.DataFrame(sample_data)
    df["rev_time"] = df["rev_time"].apply(lambda x: pd.to_datetime(x))

    pipeline = Phase1PredictPipeline()
    results = pipeline.predict(df)
    
    print("Prediction Results:")
    for i, (prob_pct, label) in enumerate(zip(results['probability_percentages'], results['labels'])):
        print(f"Sample {i+1}: {prob_pct:.2f}% fraud probability - Predicted: {label}")
    
    print(f"\nRaw probabilities: {results['raw_probabilities']}")
