
from __future__ import annotations

from typing import Tuple, Dict

import numpy as np
import tensorflow as tf

from src.models.ocan.ocgan import GANModel
from src.models.seqtab.combined import CombinedModel


def compute_representation(
    model: CombinedModel, inputs: Tuple[np.ndarray, ...], batch_size: int
) -> np.ndarray:
    """Return latent representations from a pre-trained CombinedModel."""

    def _to_tensor(x):
        """Convert ``x`` to a ``Tensor`` or ``RaggedTensor``.

        ``numpy`` arrays that store Python objects (``dtype=object``) cannot be
        directly converted using ``tf.convert_to_tensor``. These typically come
        from variable-length sequence features saved with ``np.save``/``np.load``.
        In that case we first convert the array to a nested Python ``list`` and
        create a ``RaggedTensor`` from it.
        """

        if isinstance(x, np.ndarray) and x.dtype == object:
            return tf.ragged.constant(x.tolist())

        try:
            return tf.convert_to_tensor(x)
        except (ValueError, TypeError):  # fallback for irregular data structures
            return tf.ragged.constant(x)

    prepared_inputs = [_to_tensor(inp) for inp in inputs]

    ds = tf.data.Dataset.from_tensor_slices(tuple(prepared_inputs)).batch(batch_size)
    reps = []
    flags = {"train_combined": True}
    for batch in ds:
        rep, _ = model.get_representation(batch, training=False, train_flags=flags)
        reps.append(rep.numpy())
    return np.concatenate(reps, axis=0)


def train_phase2(combined_model: CombinedModel,
                 train_data: Tuple[Tuple[np.ndarray, ...], np.ndarray],
                 val_data: Tuple[Tuple[np.ndarray, ...], np.ndarray],
                 params: Dict,
                 epochs: int = 5,
                 batch_size: int = 32) -> Tuple[GANModel, Dict]:
    """Train an OCAN model using representations from ``combined_model``."""
    (train_inputs, y_train) = train_data
    (val_inputs, y_val) = val_data

    x_train_repr = compute_representation(combined_model, train_inputs, batch_size)
    x_val_repr = compute_representation(combined_model, val_inputs, batch_size)

    y_train_oh = tf.one_hot(y_train.astype(int), depth=2)
    y_val_oh = tf.one_hot(y_val.astype(int), depth=2)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train_repr, y_train_oh)).shuffle(1024).batch(batch_size)
    val_tuple = (x_val_repr, y_val_oh)

    params = params.copy()
    params['dim_inp'] = x_train_repr.shape[1]

    gan_model = GANModel(params, total_samples=x_train_repr.shape[0])
    history = gan_model.fit(train_ds, val_tuple, epochs=epochs)
    return gan_model, history
