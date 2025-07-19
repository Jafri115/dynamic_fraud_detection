"""Hyperparameter tuning utilities and configuration."""

import os
import json
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

import optuna
import mlflow
import mlflow.tensorflow
from optuna.integration.mlflow import MLflowCallback

from src.models.seqtab.combined import CombinedModel
from src.training.combined_trainer import train_model as train_combined_model
from src.training.ocan_trainer import train_phase2 as train_ocan_model
from src.utils.helpers import CombinedCustomLoss


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""
    
    # Study configuration
    study_name: str = "wiki-fraud-detection-tuning"
    direction: str = "maximize"  # maximize F1 score
    n_trials: int = 50
    n_startup_trials: int = 10
    
    # MLflow configuration
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "wiki-fraud-detection"
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.001
    
    # Cross-validation
    cv_folds: int = 3
    
    # Sampling strategy
    enable_smote: bool = True
    smote_sampling_strategy: float = 0.7
    undersampling_strategy: float = 0.8
    
    # Pruning
    enable_pruning: bool = True
    pruning_warmup_epochs: int = 5
    
    # Resource limits
    max_epochs: int = 20
    timeout_seconds: Optional[int] = 3600  # 1 hour per trial
    
    # Model types to tune
    tune_phase1: bool = True
    tune_phase2: bool = True
    
    # Paths
    data_cache_dir: str = "data/cache"
    tuning_results_dir: str = "tuning_results"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        os.makedirs(self.data_cache_dir, exist_ok=True)
        os.makedirs(self.tuning_results_dir, exist_ok=True)


class ModelTuner:
    """Base class for model hyperparameter tuning."""
    
    def __init__(self, config: TuningConfig):
        self.config = config
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)
        
    def create_study(self, study_name: str) -> optuna.Study:
        """Create Optuna study with MLflow integration."""
        # Setup MLflow callback - let it handle run name generation
        mlflow_callback = MLflowCallback(
            tracking_uri=self.config.mlflow_tracking_uri,
            metric_name="f1_score"
        )
        
        # Create study
        study = optuna.create_study(
            study_name=study_name,
            direction=self.config.direction,
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=self.config.n_startup_trials,
                n_ei_candidates=24,
                seed=42
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=self.config.pruning_warmup_epochs,
                n_warmup_steps=self.config.pruning_warmup_epochs,
                interval_steps=1
            ) if self.config.enable_pruning else optuna.pruners.NopPruner()
        )
        
        return study, mlflow_callback
    
    def apply_class_balancing(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply class balancing using SMOTE and undersampling."""
        if not self.config.enable_smote:
            return X, y
            
        # Reshape if needed for SMOTE
        if X.ndim > 2:
            original_shape = X.shape
            X_reshaped = X.reshape(X.shape[0], -1)
        else:
            original_shape = None
            X_reshaped = X
            
        # Calculate class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(unique, counts))
        
        # Calculate a safe sampling strategy
        if len(class_counts) < 2:
            return X, y
            
        minority_class = min(class_counts.keys(), key=lambda k: class_counts[k])
        majority_class = max(class_counts.keys(), key=lambda k: class_counts[k])
        
        # Use a more conservative sampling strategy
        minority_count = class_counts[minority_class]
        majority_count = class_counts[majority_class]
        
        # Ensure we don't try to oversample beyond a reasonable ratio
        max_minority_samples = min(
            int(majority_count * self.config.smote_sampling_strategy),
            majority_count  # Don't exceed majority class size
        )
        
        # Make sure we have enough samples to work with
        if max_minority_samples <= minority_count:
            print(f"Skipping SMOTE: target samples ({max_minority_samples}) <= current samples ({minority_count})")
            return X, y
            
        try:
            # Apply SMOTE + undersampling pipeline
            pipeline = ImbPipeline([
                ('over', SMOTE(
                    sampling_strategy={minority_class: max_minority_samples},
                    random_state=42
                )),
                ('under', RandomUnderSampler(
                    sampling_strategy=self.config.undersampling_strategy, 
                    random_state=42
                ))
            ])
            
            X_resampled, y_resampled = pipeline.fit_resample(X_reshaped, y)
            
            # Reshape back if needed
            if original_shape is not None:
                X_resampled = X_resampled.reshape(-1, *original_shape[1:])
                
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"SMOTE failed: {e}. Returning original data.")
            return X, y
    
    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute class weights for imbalanced dataset."""
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        return dict(zip(np.unique(y), class_weights))
    
    def evaluate_model(self, model, X_val: np.ndarray, y_val: np.ndarray, 
                      threshold: float = 0.5) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_val)
                if y_proba.shape[1] > 1:
                    y_proba = y_proba[:, 1]
            else:
                y_pred_raw = model.predict(X_val)
                y_proba = y_pred_raw.flatten() if y_pred_raw.ndim > 1 else y_pred_raw
                
            y_pred = (y_proba >= threshold).astype(int)
            
            # Handle edge cases
            if len(np.unique(y_val)) < 2 or len(np.unique(y_pred)) < 2:
                return {
                    'f1_score': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'roc_auc': 0.5,
                    'pr_auc': 0.0,
                    'accuracy': 0.0
                }
            
            metrics = {
                'f1_score': f1_score(y_val, y_pred, average='binary'),
                'precision': precision_score(y_val, y_pred, average='binary'),
                'recall': recall_score(y_val, y_pred, average='binary'),
                'roc_auc': roc_auc_score(y_val, y_proba),
                'pr_auc': average_precision_score(y_val, y_proba),
                'accuracy': np.mean(y_val == y_pred)
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return {
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'roc_auc': 0.5,
                'pr_auc': 0.0,
                'accuracy': 0.0
            }
    
    def optimize_threshold(self, model, X_val: np.ndarray, y_val: np.ndarray, 
                          metric: str = 'f1_score') -> float:
        """Optimize classification threshold."""
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_val)
                if y_proba.shape[1] > 1:
                    y_proba = y_proba[:, 1]
            else:
                y_pred_raw = model.predict(X_val)
                y_proba = y_pred_raw.flatten() if y_pred_raw.ndim > 1 else y_pred_raw
            
            best_threshold = 0.5
            best_score = 0.0
            
            for threshold in np.arange(0.1, 0.9, 0.1):
                y_pred = (y_proba >= threshold).astype(int)
                
                if len(np.unique(y_pred)) < 2:
                    continue
                    
                if metric == 'f1_score':
                    score = f1_score(y_val, y_pred, average='binary')
                elif metric == 'precision':
                    score = precision_score(y_val, y_pred, average='binary')
                elif metric == 'recall':
                    score = recall_score(y_val, y_pred, average='binary')
                else:
                    score = f1_score(y_val, y_pred, average='binary')
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    
            return best_threshold
            
        except Exception as e:
            print(f"Error in threshold optimization: {e}")
            return 0.5
    
    def save_best_params(self, study: optuna.Study, model_name: str):
        """Save best parameters to file."""
        best_params = study.best_params
        best_value = study.best_value
        
        results = {
            'best_params': best_params,
            'best_value': best_value,
            'n_trials': len(study.trials),
            'study_name': study.study_name
        }
        
        results_path = os.path.join(
            self.config.tuning_results_dir, 
            f"{model_name}_best_params.json"
        )
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Best parameters saved to: {results_path}")
        print(f"Best {model_name} F1 score: {best_value:.4f}")
        print(f"Best parameters: {best_params}")
        
    def create_trials_dataframe(self, study: optuna.Study):
        """Create a dataframe with trial results."""
        import pandas as pd
        
        trials_df = study.trials_dataframe()
        trials_df.to_csv(
            os.path.join(self.config.tuning_results_dir, f"{study.study_name}_trials.csv"),
            index=False
        )
        
        return trials_df
