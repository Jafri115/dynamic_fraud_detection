import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
)

from src.components.model_evaluator import ModelEvaluator
from src.config.configuration import Phase1TrainingConfig
from src.components.data_ingestion import WikiDataIngestionConfig
from src.components.data_transformation import CombinedTransformConfig
from src.utils.io import load_transformed_data


def save_evaluation_results(metrics: dict, directory: str = "evaluation_results/model_evaluation"):
    """Saves evaluation metrics to a JSON file with a timestamp."""
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(directory, f"model_evaluation_{timestamp}.json")
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\n[*] Evaluation results saved to: {filepath}")
    return filepath


def main():
    print("=" * 70)
    print("PHASE 1 MODEL EVALUATION ON TEST SET")
    print("=" * 70)

    # Load configs
    cfg = Phase1TrainingConfig.from_complete_parameters()
    ingestion_cfg = WikiDataIngestionConfig()
    transform_cfg = CombinedTransformConfig()
    train_flags = cfg.train_flags

    # Load test data
    print("\nLoading test data...")
    df_test = pd.read_pickle(ingestion_cfg.test_data_path)
    y_true = df_test["label"].astype(int).values

    print("Transforming test data...")
    # Use the WikiCombinedTransformation class to process the test data
    from src.components.data_transformation import WikiCombinedTransformation
    from src.utils.io import load_object
    
    transformer = WikiCombinedTransformation()
    
    # Compute time deltas and extract features
    df_test = transformer.compute_time_deltas(df_test)
    x_tab, y_test = transformer.extract_tabular_features(df_test, label_required=True)
    
    # Load preprocessor and transform tabular features
    preprocessor = load_object(transform_cfg.preprocessor_path)
    x_tab = preprocessor.transform(x_tab)
    
    # Apply feature selection if specified
    if cfg.selected_indices is not None:
        x_tab = x_tab[:, cfg.selected_indices]
    
    # Extract sequence features  
    from src.models.seqtab.units import pad_matrix, pad_time, pad_failure_bits
    
    # Get sequence parameters from model config (similar to predict pipeline)
    with open(cfg.model_config_path, 'r') as f:
        model_config = json.load(f)
    
    n_event_code = model_config["n_event_code"]
    max_len = model_config["max_len"] 
    max_code = model_config["max_code"]
    
    # Prepare sequence features similar to predict pipeline
    event_seq = [transformer.impute_sequence(s) + [[n_event_code - 1]] for s in df_test["edit_sequence"]]
    time_seq = [transformer.impute_sequence(s) + [0] for s in df_test["time_delta_seq"]]
    
    fail_sys = [[0] * len(transformer.impute_sequence(s)) + [1] for s in df_test["edit_sequence"]]
    fail_user = [[0] * len(transformer.impute_sequence(s)) + [1] for s in df_test["edit_sequence"]]
    
    # Pad sequences using the same max_len and max_code from config
    event_seq_pad, mask, mask_final, mask_code = pad_matrix(
        event_seq, max_len, pad_token=n_event_code,
        n_event_code=n_event_code, maxcode=max_code
    )
    time_pad = pad_time(time_seq, max_len=max_len)
    fail_sys_pad = pad_failure_bits(fail_sys, max_len=max_len)
    fail_user_pad = pad_failure_bits(fail_user, max_len=max_len)
    lengths = np.array([len(s) for s in event_seq], dtype=np.float32)
    
    seq_inputs = (event_seq_pad, time_pad, fail_sys_pad, fail_user_pad, mask, mask_final, mask_code, lengths)
    
    # Combine all inputs for the model
    test_inputs = (x_tab.astype(np.float32), *seq_inputs)

    print(f"Loaded {len(y_test)} test samples.")

    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=cfg.model_path,
        config_path=cfg.model_config_path,
        train_flags=train_flags
    )

    # Evaluate
    print("\nRunning model evaluation...")
    metrics, y_probs, y_true_used = evaluator.evaluate(
        test_inputs, y_test, batch_size=cfg.batch_size, return_preds=True
    )

    y_probs = np.array(y_probs)
    y_pred = (y_probs >= 0.5).astype(int)

    # Full metric calculation (safeguarded with y_true alignment)
    accuracy = accuracy_score(y_true_used, y_pred)
    f1 = f1_score(y_true_used, y_pred)
    precision = precision_score(y_true_used, y_pred)
    recall = recall_score(y_true_used, y_pred)
    roc_auc = roc_auc_score(y_true_used, y_probs)
    avg_precision = average_precision_score(y_true_used, y_probs)
    
    print("\n=== Detailed Evaluation Metrics ===")
    print(f"Accuracy           : {accuracy:.4f}")
    print(f"F1 Score           : {f1:.4f}")
    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")
    print(f"ROC AUC            : {roc_auc:.4f}")
    print(f"Average Precision  : {avg_precision:.4f}")

    print("\n=== Classification Report ===")
    class_report_str = classification_report(y_true_used, y_pred, target_names=["Legit", "Vandal"])
    print(class_report_str)
    
    # Prepare results for saving
    evaluation_metrics = {
        "test_evaluation": {
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "roc_auc": float(roc_auc),
            "average_precision": float(avg_precision),
            "classification_report": classification_report(y_true_used, y_pred, target_names=["Legit", "Vandal"], output_dict=True)
        },
        "metadata": {
            "evaluation_timestamp": datetime.now().isoformat(),
            "model_path": cfg.model_path,
            "model_config_path": cfg.model_config_path,
            "num_test_samples": len(y_test)
        }
    }

    # Save the results
    save_evaluation_results(evaluation_metrics)


if __name__ == "__main__":
    main()
