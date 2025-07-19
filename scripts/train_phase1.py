# scripts/train_phase1.py
import os
import sys
import json
import numpy as np
import tensorflow as tf
import pandas as pd



from src.components.data_transformation import CombinedTransformConfig
from src.components.data_ingestion import WikiDataIngestionConfig
from src.models.seqtab.combined import CombinedModel
from src.training.combined_trainer import train_model
from src.utils.visualization import plot_training_history
from src.config.configuration import Phase1TrainingConfig
from src.utils.io import load_transformed_data

# Ensure GPU memory is allocated incrementally to help avoid OOM errors
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as err:  # RuntimeError if GPUs have already been initialized
        print(f"Could not set memory growth for {gpu}: {err}")








def convert_to_python_type(obj):
    if isinstance(obj, dict):
        return {k: convert_to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_type(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()  # convert numpy scalar to native Python
    else:
        return obj
    
def main():
    print("="*70)
    print("PHASE 1 TRAINING WITH COMPLETE PARAMETERS")
    print("="*70)
    
    cfg = CombinedTransformConfig()
    i_cfg = WikiDataIngestionConfig()
    
    # Load configuration from complete parameters JSON
    train_cfg = Phase1TrainingConfig.from_complete_parameters()
    
    print(f"Using optimized parameters:")
    print(f"  Model Dim: {train_cfg.model_dim}")
    print(f"  Learning Rate: {train_cfg.learning_rate}")
    print(f"  Batch Size: {train_cfg.batch_size}")
    print(f"  Epochs: {train_cfg.epochs}")
    print(f"  Combined Hidden Layers: {train_cfg.combined_hidden_layers}")
    print(f"  Tab Hidden States: {train_cfg.tab_hidden_states}")
    
    # Display tuning metadata if available
    if hasattr(train_cfg, 'tuning_metadata'):
        metadata = train_cfg.tuning_metadata
        if metadata.get('best_f1_score'):
            print(f"  Best F1 Score from tuning: {metadata['best_f1_score']:.4f}")
        if metadata.get('n_trials'):
            print(f"  Number of trials: {metadata['n_trials']}")

    # Load data with configuration
    (train_inputs, y_train), (val_inputs, y_val) = load_transformed_data(
        cfg, train_cfg.selected_indices, train_cfg.sample_size, train_cfg.random_state
    )

    category_dict = pd.read_pickle(i_cfg.category_dict_path)
    input_shape_tab = train_inputs[0].shape[1]

    # Generate model configuration
    model_config = train_cfg.to_model_config(input_shape_tab, category_dict['no_cat']+1)

    # Instantiate model with filtered parameters
    model_params = train_cfg.get_model_params(input_shape_tab, category_dict['no_cat']+1)
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
        train_flags=train_cfg.train_flags,
        verbose=True,
    )
    history = convert_to_python_type(history)

    # === Save artifacts ===
    os.makedirs(train_cfg.artifact_dir, exist_ok=True)

    # Save full model
    model.save(train_cfg.model_path)

    # Save training log
    with open(train_cfg.training_log_path, "w") as f:
        json.dump(history, f, indent=4)

    # Save model config
    with open(train_cfg.model_config_path, "w") as f:
        json.dump(model_config, f, indent=4)

    # Plot and save
    plot_training_history(history)

    print("Training complete")


if __name__ == "__main__":
    main()
