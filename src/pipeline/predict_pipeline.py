# src/pipeline/predict_pipeline.py
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, Union

from src.models.seqtab.combined import CombinedModel
from src.models.seqtab.units import pad_matrix, pad_time, pad_failure_bits
from src.components.data_transformation import WikiCombinedTransformation
from src.utils.io import load_object
from src.config.configuration import PredictionConfig

import tensorflow as tf
tf.get_logger().setLevel('ERROR')


class Phase1PredictPipeline:
    def __init__(self):
        cfg = PredictionConfig()

        # Load model config
        with open(cfg.config_path, "r") as f:
            self.model_config = json.load(f)

        self.model = CombinedModel.from_config(self.model_config)
        self.model.load_weights(cfg.model_dir)

        # Load preprocessor
        self.preprocessor = load_object(cfg.preprocessor_path)

        # Load transformer used for feature engineering
        self.transformer = WikiCombinedTransformation()

        self.n_event_code = self.model_config["n_event_code"]
        self.max_len = self.model_config["max_len"]
        self.max_code = self.model_config["max_code"]
        self.selected_indices = self.model_config.get("selected_indices")

    def _prepare_tabular_features(self, df: pd.DataFrame) -> np.ndarray:
        df = self.transformer.compute_time_deltas(df)
        x_tab, _ = self.transformer.extract_tabular_features(df, label_required=False)

        if self.preprocessor is not None:
            x_tab = self.preprocessor.transform(x_tab)

        if self.selected_indices is not None:
            x_tab = x_tab[:, self.selected_indices]

        return x_tab.astype(np.float32)

    def _prepare_sequence_features(self, df: pd.DataFrame) -> tuple:
        event_seq = [self.transformer.impute_sequence(s) + [[self.n_event_code - 1]] for s in df["edit_sequence"]]
        time_seq = [self.transformer.impute_sequence(s) + [0] for s in df["time_delta_seq"]]

        fail_sys = [[0] * len(self.transformer.impute_sequence(s)) + [1] for s in df["edit_sequence"]]
        fail_user = [[0] * len(self.transformer.impute_sequence(s)) + [1] for s in df["edit_sequence"]]

        event_seq_pad, mask, mask_final, mask_code = pad_matrix(
            event_seq, self.max_len, pad_token=self.n_event_code,
            n_event_code=self.n_event_code, maxcode=self.max_code
        )
        time_pad = pad_time(time_seq, max_len=self.max_len)
        fail_sys_pad = pad_failure_bits(fail_sys, max_len=self.max_len)
        fail_user_pad = pad_failure_bits(fail_user, max_len=self.max_len)
        lengths = np.array([len(s) for s in event_seq], dtype=np.float32)

        return (
            event_seq_pad, time_pad, fail_sys_pad, fail_user_pad,
            mask, mask_final, mask_code, lengths
        )

    def predict(self, df: pd.DataFrame) -> Dict[str, Union[np.ndarray, list]]:
        if "rev_time" not in df or "edit_sequence" not in df:
            raise ValueError("DataFrame must include 'rev_time' and 'edit_sequence' columns.")

        # Prepare tabular & sequence features
        x_tab = self._prepare_tabular_features(df)
        seq_inputs = self._prepare_sequence_features(df)

        # Predict
        preds = self.model.predict_from_representation(
            (x_tab, *seq_inputs),
            training=False,
            train_flags={"train_combined": True}
        )

        probabilities = preds.numpy().ravel()
        labels = ["fraud" if p >= 0.5 else "legitimate" for p in probabilities]

        return {
            "raw_probabilities": probabilities,
            "probability_percentages": probabilities * 100,
            "labels": labels,
            "prediction_mode": "combined"
        }
