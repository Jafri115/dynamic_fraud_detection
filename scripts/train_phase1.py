import os
import sys
import json
import numpy as np
import tensorflow as tf
import pandas as pd

from src.components.data_transformation import CombinedTransformConfig
from src.components.data_ingestion import WikiDataIngestionConfig, WikiDataIngestion
from src.models.seqtab.combined import CombinedModel
from src.training.combined_trainer import train_model
from src.utils.visualization import plot_training_history
from src.config.configuration import Phase1TrainingConfig
from src.utils.io import load_transformed_data

def setup_gpu():
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as err:
            print(f"Could not set memory growth for {gpu}: {err}")

def convert_to_python_type(obj):
    if isinstance(obj, dict):
        return {k: convert_to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_type(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj

def display_config(train_cfg):
    print(f"  Model Dim: {train_cfg.model_dim}")
    print(f"  Learning Rate: {train_cfg.learning_rate}")
    print(f"  Batch Size: {train_cfg.batch_size}")
    print(f"  Epochs: {train_cfg.epochs}")
    print(f"  Combined Hidden Layers: {train_cfg.combined_hidden_layers}")
    print(f"  Tab Hidden States: {train_cfg.tab_hidden_states}")

    if hasattr(train_cfg, 'tuning_metadata'):
        metadata = train_cfg.tuning_metadata
        if metadata.get('best_f1_score'):
            print(f"  Best F1 Score from tuning: {metadata['best_f1_score']:.4f}")
        if metadata.get('n_trials'):
            print(f"  Number of trials: {metadata['n_trials']}")

def prepare_data(i_cfg):
    if not (os.path.exists(i_cfg.train_data_path) and
            os.path.exists(i_cfg.val_data_path) and
            os.path.exists(i_cfg.test_data_path)):
        print("\nEnhanced VEWS data not found, creating with behavioral features...")
        data_ingestion = WikiDataIngestion()
        return data_ingestion.initiate_enhanced_data_ingestion()
    else:
        print("\nUsing existing enhanced VEWS data...")
        return i_cfg.train_data_path, i_cfg.val_data_path, i_cfg.test_data_path

def train_phase():
    print("="*70)
    print("PHASE 1 TRAINING WITH ENHANCED VEWS FEATURES")
    print("="*70)

    setup_gpu()

    cfg = CombinedTransformConfig()
    i_cfg = WikiDataIngestionConfig()
    train_cfg = Phase1TrainingConfig.from_complete_parameters()

    print("Using optimized parameters:")
    display_config(train_cfg)

    # enhanced_train_path, enhanced_val_path, _ = prepare_data(i_cfg)

    (train_inputs, y_train), (val_inputs, y_val) = load_transformed_data(
        cfg, train_cfg.selected_indices, train_cfg.sample_size, train_cfg.random_state
    )

    # Add debugging information
    print(f"\nDEBUG: Data shapes after loading:")
    for i, inp in enumerate(train_inputs):
        print(f"  train_inputs[{i}].shape: {inp.shape}")
    for i, inp in enumerate(val_inputs):
        print(f"  val_inputs[{i}].shape: {inp.shape}")
    print(f"  y_train.shape: {y_train.shape}")
    print(f"  y_val.shape: {y_val.shape}")

    category_dict = pd.read_pickle(i_cfg.category_dict_path)
    input_shape_tab = train_inputs[0].shape[1]
    has_sequence_data = len(train_inputs) > 1

    print("\nData configuration:")
    print(f"  Input shape (tabular): {train_inputs[0].shape}")
    print(f"  Has sequence data: {has_sequence_data}")
    print(f"  Training samples: {len(y_train)}")
    print(f"  Validation samples: {len(y_val)}")
    print(f"  Calculated input_shape_tab: {input_shape_tab}")
    print(f"  Category dict 'no_cat': {category_dict['no_cat']}")

    train_flags = train_cfg.train_flags if has_sequence_data else {'train_tab': True, 'train_seq': False, 'train_combined': False}
    print("  Training with combined model" if has_sequence_data else "  Training with tabular-only model")
    print(f"  Train flags: {train_flags}")

    model_config = train_cfg.to_model_config(input_shape_tab, category_dict['no_cat'] + 1)
    model_params = train_cfg.get_model_params(input_shape_tab, category_dict['no_cat'] + 1)
    
    print(f"\nDEBUG: Model configuration:")
    print(f"  input_shape_tab: {input_shape_tab}")
    print(f"  category_dict['no_cat']: {category_dict['no_cat']}")
    print(f"  n_event_code: {category_dict['no_cat'] + 1}")
    print(f"  model_params keys: {list(model_params.keys()) if isinstance(model_params, dict) else 'Not a dict'}")
    if isinstance(model_params, dict):
        print(f"  model_params input_shape_tabular: {model_params.get('input_shape_tabular', 'NOT FOUND')}")
        print(f"  model_params tab_hidden_states: {model_params.get('tab_hidden_states', 'NOT FOUND')}")
    
    model = CombinedModel(**model_params)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(train_cfg.learning_rate),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc_roc"),
            tf.keras.metrics.AUC(curve="PR", name="auc_pr"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    history = train_model(
        model,
        train_data=(train_inputs, y_train),
        val_data=(val_inputs, y_val),
        epochs=train_cfg.epochs,
        train_flags=train_flags,
        verbose=True,
    )
    history = convert_to_python_type(history)

    os.makedirs(train_cfg.artifact_dir, exist_ok=True)
    model.save(train_cfg.model_path)

    with open(train_cfg.training_log_path, "w") as f:
        json.dump(history, f, indent=4)

    with open(train_cfg.model_config_path, "w") as f:
        json.dump(model_config, f, indent=4)

    plot_training_history(history)
    print("Training complete")

if __name__ == "__main__":
    train_phase()
