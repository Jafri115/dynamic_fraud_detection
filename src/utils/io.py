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
    
    
def sample_balanced_indices(y, sample_size, rng):
    """Returns balanced indices (50/50 pos-neg) from binary labels."""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    pos_count = max(1, sample_size // 2)
    neg_count = sample_size - pos_count

    sampled_pos = rng.choice(pos_idx, size=pos_count, replace=len(pos_idx) < pos_count)
    sampled_neg = rng.choice(neg_idx, size=neg_count, replace=len(neg_idx) < neg_count)

    indices = np.concatenate([sampled_pos, sampled_neg])
    rng.shuffle(indices)
    return indices


def load_transformed_data(cfg, selected_indices=None, sample_size='all', random_state=42):
    """
    Load transformed tabular and (optional) sequence data with feature selection and sampling.

    Args:
        cfg: Configuration object with .npy and .npz paths.
        selected_indices: List of indices for feature selection.
        sample_size: 'all' or integer number of samples per split.
        random_state: Random seed.
        train_path: Optional raw train data path (if transformation is needed).
        val_path: Optional raw val data path.

    Returns:
        (train_inputs, y_train), (val_inputs, y_val)
    """
    rng = np.random.default_rng(random_state)


    # Load tabular data
    x_tab_train = np.load(cfg.x_train_path)
    x_tab_val = np.load(cfg.x_val_path)

    if selected_indices is not None:
        x_tab_train = x_tab_train[:, selected_indices]
        x_tab_val = x_tab_val[:, selected_indices]

    y_train = np.load(cfg.train_label_path)
    y_val = np.load(cfg.val_label_path)

    # Sample train/val indices
    if sample_size == 'all':
        train_indices = np.arange(len(y_train))
        val_indices = np.arange(len(y_val))
    else:
        train_indices = sample_balanced_indices(y_train, sample_size, rng)
        val_indices = rng.choice(len(y_val), size=min(sample_size, len(y_val)), replace=False)

    x_tab_train = x_tab_train[train_indices]
    x_tab_val = x_tab_val[val_indices]
    y_train = y_train[train_indices]
    y_val = y_val[val_indices]

    # Load sequence data if present
    try:
        train_seq = np.load(cfg.train_seq_path, allow_pickle=True)
        val_seq = np.load(cfg.val_seq_path, allow_pickle=True)

        train_inputs = (
            x_tab_train,
            train_seq["event_seq"][train_indices],
            train_seq["rev_time"][train_indices],
            train_seq["event_failure_sys"][train_indices],
            train_seq["event_failure_user"][train_indices],
        )
        val_inputs = (
            x_tab_val,
            val_seq["event_seq"][val_indices],
            val_seq["rev_time"][val_indices],
            val_seq["event_failure_sys"][val_indices],
            val_seq["event_failure_user"][val_indices],
        )
    except FileNotFoundError:
        print("[*] No sequence data found. Using tabular inputs only.")
        train_inputs = (x_tab_train,)
        val_inputs = (x_tab_val,)

    return (train_inputs, y_train), (val_inputs, y_val)



