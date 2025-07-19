"""Phase 2 (OCAN) hyperparameter tuning."""

import os
import json
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple, List, Optional
from sklearn.metrics import f1_score

import optuna
import mlflow
import mlflow.tensorflow

from src.tuning.base_tuner import ModelTuner, TuningConfig
from src.models.ocan.ocgan import GANModel
from src.training.ocan_trainer import train_phase2 as train_ocan_model


class Phase2Tuner(ModelTuner):
    """Hyperparameter tuning for Phase 2 OCAN Model."""
    
    def __init__(self, config: TuningConfig):
        super().__init__(config)
        self.model_name = "phase2_ocan"
        
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for Phase 2 OCAN model."""
        params = {
            # GAN Architecture
            'g_layers': trial.suggest_categorical('g_layers', [
                [32, 64], [64, 128], [128, 256], [32, 64, 128]
            ]),
            'd_layers': trial.suggest_categorical('d_layers', [
                [64, 32], [128, 64], [256, 128], [128, 64, 32]
            ]),
            
            # Training parameters
            'mb_size': trial.suggest_categorical('mb_size', [16, 32, 64, 128]),
            'epochs': trial.suggest_int('epochs', 5, 25),
            
            # Dropout
            'g_dropout_1': trial.suggest_float('g_dropout_1', 0.1, 0.5, step=0.1),
            'g_dropout_2': trial.suggest_float('g_dropout_2', 0.1, 0.5, step=0.1),
            'd_dropout_1': trial.suggest_float('d_dropout_1', 0.1, 0.5, step=0.1),
            'd_dropout_2': trial.suggest_float('d_dropout_2', 0.1, 0.5, step=0.1),
            
            # Learning rates
            'g_lr': trial.suggest_float('g_lr', 1e-5, 1e-2, log=True),
            'd_lr': trial.suggest_float('d_lr', 1e-5, 1e-2, log=True),
            
            # Adam parameters
            'beta1_g': trial.suggest_float('beta1_g', 0.3, 0.7, step=0.1),
            'beta2_g': trial.suggest_float('beta2_g', 0.8, 0.999, step=0.01),
            'beta1_d': trial.suggest_float('beta1_d', 0.3, 0.7, step=0.1),
            'beta2_d': trial.suggest_float('beta2_d', 0.8, 0.999, step=0.01),
            
            # Loss weights
            'lambda_pt': trial.suggest_float('lambda_pt', 0.01, 1.0, step=0.01),
            'lambda_ent': trial.suggest_float('lambda_ent', 0.01, 1.0, step=0.01),
            'lambda_fm': trial.suggest_float('lambda_fm', 0.01, 1.0, step=0.01),
            'lambda_gp': trial.suggest_float('lambda_gp', 1.0, 20.0, step=1.0),
            
            # Batch normalization
            'batch_norm_g': trial.suggest_categorical('batch_norm_g', [True, False]),
            'batch_norm_d': trial.suggest_categorical('batch_norm_d', [True, False]),
            
            # Class balancing
            'use_class_weights': trial.suggest_categorical('use_class_weights', [True, False]),
            'use_smote': trial.suggest_categorical('use_smote', [True, False]),
        }
        
        return params
    
    def create_ocan_params(self, params: Dict[str, Any], dim_inp: int) -> Dict[str, Any]:
        """Create OCAN model parameters from suggested hyperparameters."""
        ocan_params = {
            "G_D_layers": (params['g_layers'], params['d_layers']),
            "dim_inp": dim_inp,
            "mb_size": params['mb_size'],
            "g_dropouts": (params['g_dropout_1'], params['g_dropout_2']),
            "d_dropouts": (params['d_dropout_1'], params['d_dropout_2']),
            "batch_norm_g": params['batch_norm_g'],
            "batch_norm_d": params['batch_norm_d'],
            "g_lr": params['g_lr'],
            "d_lr": params['d_lr'],
            "beta1_g": params['beta1_g'],
            "beta2_g": params['beta2_g'],
            "beta1_d": params['beta1_d'],
            "beta2_d": params['beta2_d'],
            "lambda_pt": params['lambda_pt'],
            "lambda_ent": params['lambda_ent'],
            "lambda_fm": params['lambda_fm'],
            "lambda_gp": params['lambda_gp'],
        }
        
        return ocan_params
    
    def objective(self, trial: optuna.Trial, combined_model, train_data: Tuple, 
                 val_data: Tuple) -> float:
        """Objective function for Optuna optimization."""
        
        # Start MLflow run
        with mlflow.start_run(nested=True):
            # Suggest hyperparameters
            params = self.suggest_hyperparameters(trial)
            
            # Log parameters
            mlflow.log_params(params)
            
            try:
                # Apply data balancing if enabled
                train_inputs, y_train = train_data
                if params['use_smote']:
                    # For OCAN, we work with representation data
                    X_balanced, y_balanced = self.apply_class_balancing(
                        train_inputs[0], y_train
                    )
                    train_inputs = (X_balanced, *train_inputs[1:])
                    y_train = y_balanced
                
                # Update train_data
                train_data = (train_inputs, y_train)
                
                # Get representation dimension
                dim_inp = train_inputs[0].shape[1]
                
                # Create OCAN parameters
                ocan_params = self.create_ocan_params(params, dim_inp)
                
                # Train OCAN model
                gan_model, history = train_ocan_model(
                    combined_model,
                    train_data,
                    val_data,
                    ocan_params,
                    epochs=params['epochs'],
                    batch_size=params['mb_size']
                )
                
                # Get best F1 score from validation
                val_f1_scores = history.get('F1score_val', [])
                if val_f1_scores:
                    best_f1 = max(val_f1_scores)
                else:
                    # Fallback: evaluate manually
                    val_inputs, y_val = val_data
                    eval_results = gan_model.evaluate_model(val_inputs, y_val)
                    best_f1 = eval_results.get('F1_score', 0.0)
                
                # Log metrics
                mlflow.log_metric("f1_score", best_f1)
                mlflow.log_metric("final_epoch", len(val_f1_scores))
                
                # Log additional metrics if available
                for metric_name, values in history.items():
                    if values and isinstance(values, list):
                        if 'loss' in metric_name.lower():
                            mlflow.log_metric(f"final_{metric_name}", values[-1])
                        else:
                            mlflow.log_metric(f"best_{metric_name}", max(values))
                
                # Clean up
                del gan_model
                tf.keras.backend.clear_session()
                
                return best_f1
                
            except optuna.TrialPruned:
                # Log that trial was pruned
                mlflow.log_metric("pruned", 1)
                raise
            except Exception as e:
                # Log error
                mlflow.log_param("error", str(e))
                print(f"Trial {trial.number} failed: {e}")
                return 0.0
    
    def tune(self, combined_model, train_data: Tuple, val_data: Tuple) -> Dict[str, Any]:
        """Run hyperparameter tuning."""
        print(f"Starting Phase 2 hyperparameter tuning...")
        print(f"Number of trials: {self.config.n_trials}")
        
        # Create study
        study_name = f"{self.model_name}_{self.config.study_name}"
        study, mlflow_callback = self.create_study(study_name)
        
        # Optimize
        study.optimize(
            lambda trial: self.objective(trial, combined_model, train_data, val_data),
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds,
            callbacks=[mlflow_callback]
        )
        
        # Save results
        self.save_best_params(study, self.model_name)
        trials_df = self.create_trials_dataframe(study)
        
        print(f"Tuning completed! Best F1 score: {study.best_value:.4f}")
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study,
            'trials_df': trials_df
        }
    
    def train_with_best_params(self, combined_model, train_data: Tuple, 
                              val_data: Tuple, best_params_path: str = None) -> Tuple[GANModel, Dict]:
        """Train OCAN model with best parameters."""
        
        # Load best parameters
        if best_params_path is None:
            best_params_path = os.path.join(
                self.config.tuning_results_dir, 
                f"{self.model_name}_best_params.json"
            )
        
        with open(best_params_path, 'r') as f:
            results = json.load(f)
            best_params = results['best_params']
        
        print(f"Training Phase 2 model with best parameters...")
        print(f"Best parameters: {best_params}")
        
        # Apply data balancing if enabled
        train_inputs, y_train = train_data
        if best_params.get('use_smote', False):
            X_balanced, y_balanced = self.apply_class_balancing(
                train_inputs[0], y_train
            )
            train_inputs = (X_balanced, *train_inputs[1:])
            y_train = y_balanced
        
        # Update train_data
        train_data = (train_inputs, y_train)
        
        # Get representation dimension
        dim_inp = train_inputs[0].shape[1]
        
        # Create OCAN parameters
        ocan_params = self.create_ocan_params(best_params, dim_inp)
        
        # Train OCAN model
        gan_model, history = train_ocan_model(
            combined_model,
            train_data,
            val_data,
            ocan_params,
            epochs=best_params['epochs'],
            batch_size=best_params['mb_size']
        )
        
        return gan_model, history
