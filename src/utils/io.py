# src/utils/io.py

"""Utility functions for file IO operations."""

import os
import pickle
import numpy as np


def save_object(file_path: str, obj) -> None:
    """Serialize an object to the given file path using pickle."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_object(file_path: str):
    """Load a pickled object from the given file path."""
    with open(file_path, "rb") as f:
        return pickle.load(f)
    
    
def load_transformed_data(cfg, selected_indices=None, sample_size=100, random_state=42):
    """
    Load transformed tabular and sequence data with optional feature selection and sample size.

    Args:
        cfg: Configuration object containing paths to .npy/.npz files.
        selected_indices (list[int], optional): Indices of selected tabular features.
        sample_size (int or 'all'): Number of samples to load, or 'all' to load full dataset.

    Returns:
        Tuple of training and validation data: (inputs, labels)
    """
    rng = np.random.default_rng(random_state)

    # === Load tabular data ===
    x_tab_train_full = np.load(cfg.x_train_path)
    x_tab_val_full = np.load(cfg.x_val_path)

    # Feature selection
    if selected_indices is not None:
        x_tab_train_full = x_tab_train_full[:, selected_indices]
        x_tab_val_full = x_tab_val_full[:, selected_indices]

    # === Load sequence data ===
    train_seq = np.load(cfg.train_seq_path, allow_pickle=True)
    val_seq = np.load(cfg.val_seq_path, allow_pickle=True)

    # === Load labels ===
    y_train_full = np.load(cfg.train_label_path)
    y_val_full = np.load(cfg.val_label_path)

    # === Select sample indices ===
    if sample_size == 'all':
        train_indices = np.arange(len(y_train_full))
        val_indices = np.arange(len(y_val_full))
    else:
        sample_size = int(sample_size)

        # Balanced sampling for train
        pos_idx = np.where(y_train_full == 1)[0]
        neg_idx = np.where(y_train_full == 0)[0]
        pos_count = max(1, sample_size // 2)
        neg_count = sample_size - pos_count
        train_indices = np.concatenate([
            rng.choice(pos_idx, size=pos_count, replace=len(pos_idx) < pos_count),
            rng.choice(neg_idx, size=neg_count, replace=len(neg_idx) < neg_count)
        ])
        rng.shuffle(train_indices)

        # Random subset for validation
        val_indices = rng.choice(len(y_val_full), size=min(sample_size, len(y_val_full)), replace=False)

    # === Apply indices ===
    x_tab_train = x_tab_train_full[train_indices]
    y_train = y_train_full[train_indices]
    x_tab_val = x_tab_val_full[val_indices]
    y_val = y_val_full[val_indices]

    train_seq_data = (
        train_seq["event_seq"][train_indices],
        train_seq["rev_time"][train_indices],
        train_seq["event_failure_sys"][train_indices],
        train_seq["event_failure_user"][train_indices],
    )
    val_seq_data = (
        val_seq["event_seq"][val_indices],
        val_seq["rev_time"][val_indices],
        val_seq["event_failure_sys"][val_indices],
        val_seq["event_failure_user"][val_indices],
    )

    train_inputs = (x_tab_train, *train_seq_data)
    val_inputs = (x_tab_val, *val_seq_data)

    return (train_inputs, y_train), (val_inputs, y_val)

