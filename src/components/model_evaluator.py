# src/evaluation/evaluator.py

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)
import tensorflow as tf
from src.models.seqtab.combined import CombinedModel


class ModelEvaluator:
    def __init__(self, model_path, config_path, train_flags):
        self.model_path = model_path
        self.config_path = config_path
        self.train_flags = train_flags
        self.model = self.load_model()

    def load_model(self):
        # Load config
        import json
        with open(self.config_path, 'r') as f:
            config = json.load(f)

        # Rebuild model and load weights
        model = CombinedModel.from_config(config)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc_roc"),
                tf.keras.metrics.AUC(curve="PR", name="auc_pr"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ]
        )
        model.load_weights(self.model_path)
        return model

    def evaluate(self, inputs, labels, batch_size=128, return_preds=False):
        preds = []
        y_true = []

        n_samples = len(labels)
        num_batches = (n_samples + batch_size - 1) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_samples)
            batch_inputs = self._slice_inputs(inputs, start, end)
            batch_labels = labels[start:end]

            y_pred = self.model.predict_from_representation(batch_inputs, training=False, train_flags=self.train_flags)
            preds.extend(y_pred.numpy().ravel())
            y_true.extend(batch_labels)

        preds = np.array(preds)
        y_true = np.array(y_true)

        y_pred_binary = (preds >= 0.5).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred_binary),
            "precision": precision_score(y_true, y_pred_binary, zero_division=0),
            "recall": recall_score(y_true, y_pred_binary, zero_division=0),
            "f1_score": f1_score(y_true, y_pred_binary),
            "auc_roc": roc_auc_score(y_true, preds),
            "auc_pr": average_precision_score(y_true, preds),
        }

        if return_preds:
            return metrics, preds, y_true
        return metrics

    def _slice_inputs(self, inputs, start, end):
        if self.train_flags.get("train_combined"):
            tabular_input, *seq_inputs = inputs
            tab_slice = tabular_input[start:end]
            seq_slices = [s[start:end] for s in seq_inputs]
            return (tab_slice, *seq_slices)

        return inputs[start:end]

