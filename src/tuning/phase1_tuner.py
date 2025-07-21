"""Phase 1 (Combined Model) hyperparameter tuning."""

import os
import json
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

import optuna
import mlflow
import mlflow.tensorflow

from src.tuning.base_tuner import ModelTuner, TuningConfig
from src.models.seqtab.combined import CombinedModel
from src.training.combined_trainer import train_model as train_combined_model
from src.config.configuration import Phase1TrainingConfig


class Phase1Tuner(ModelTuner):
    """Hyperparameter tuning for Phase 1 Combined Model - SPEED OPTIMIZED v2."""
    
    def __init__(self, config: TuningConfig):
        super().__init__(config)
        self.model_name = "phase1_combined"
        print(f"Phase1Tuner initialized with SPEED OPTIMIZATIONS - {tf.timestamp()}")
        
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for Phase 1 model - ULTRA SPEED OPTIMIZED."""
        params = {
            # === Model architecture ===
            'model_dim': trial.suggest_categorical('model_dim', [128]),  # 128 consistently outperformed 64
            'layer': trial.suggest_categorical('layer', [2, 3]),  # Deeper models were more effective

            # === Dropout rates ===
            'dropout_rate_seq': trial.suggest_float('dropout_rate_seq', 0.0, 0.2, step=0.1),  # Avoid 0.3 – hurt performance
            'droput_rate_tab': trial.suggest_float('droput_rate_tab', 0.0, 0.2, step=0.1),
            'dropout_rate_comb': trial.suggest_float('dropout_rate_comb', 0.0, 0.2, step=0.1),

            # === Learning rate & batch size ===
            'learning_rate': trial.suggest_float('learning_rate', 8e-4, 3e-3, log=True),  # Best range based on logs
            'batch_size': trial.suggest_categorical('batch_size', [32, 64]),  # 16 was slower and less effective

            # === Epochs (Optional: enable dynamic training length) ===
            'epochs': 10,  # Fixed for speed (you can make this tunable if needed)

            # === L2 Regularization ===
            'l2_lambda': trial.suggest_float('l2_lambda', 0.0, 1e-3, step=1e-4),  # Fine-tune regularization

            # === Architecture sizes ===
            'combined_hidden_layers': trial.suggest_categorical(
                'combined_hidden_layers',
                ['128_64', '128_64_32', '256_128']  # Removed '64_32' due to weak performance
            ),
            'tab_hidden_states': trial.suggest_categorical(
                'tab_hidden_states',
                ['64_32', '128_64']  # Dropped '32_16'
            ),

            # === Class balancing ===
            'use_class_weights': trial.suggest_categorical('use_class_weights', [False]),  # Always false – better performance
            'use_smote': trial.suggest_categorical('use_smote', [False]),  # SMOTE consistently hurt results

            # === Optimization ===
            'optimizer': 'adam',       # Fixed for now
            'beta_1': 0.9,
            'beta_2': 0.999,

            # === Early stopping parameters ===
            'early_stopping_patience': trial.suggest_categorical('early_stopping_patience', [3, 4]),
            'early_stopping_min_delta': trial.suggest_categorical('early_stopping_min_delta', [1e-3, 5e-4]),
        }



        print(f"Trial {trial.number}: model_dim={params['model_dim']}, lr={params['learning_rate']}, batch={params['batch_size']}")
        return params
    
    def create_model(self, params: Dict[str, Any], input_shape_tabular: int, 
                    n_event_code: int) -> CombinedModel:
        """Create Phase 1 model with given parameters - SPEED OPTIMIZED."""
        # Parse architecture string representations back to lists
        combined_hidden_layers = [int(x) for x in params['combined_hidden_layers'].split('_')]

        # Handle tab_hidden_states (can be missing, string or list)
        tab_states_param = params.get('tab_hidden_states', '32_16')
        if isinstance(tab_states_param, str):
            tab_hidden_states = [int(x) for x in tab_states_param.split('_')]
        elif hasattr(tab_states_param, '__iter__'):
            tab_hidden_states = list(tab_states_param)
        else:
            tab_hidden_states = [32, 16]
        
        model_config = {
            'input_shape_tabular': input_shape_tabular,
            'max_len': 25,  # Fixed from base config
            'max_code': 50,  # Fixed from base config
            'n_event_code': n_event_code,
            'model_dim': params['model_dim'],
            'layer': params['layer'],
            'dropout_rate_seq': params['dropout_rate_seq'],
            'droput_rate_tab': params['droput_rate_tab'],  # Note: typo in CombinedModel
            'dropout_rate_comb': params['dropout_rate_comb'],
            'combined_hidden_layers': combined_hidden_layers,
            'tab_hidden_states': tab_hidden_states,
            'is_public_dataset': True,
            'batch_size': params['batch_size'],
            'l2_lambda_comb': params['l2_lambda'],  # Use l2_lambda for combined layers
            'l2_lambda_tab': params['l2_lambda'],   # Use l2_lambda for tabular layers
            'l2_lambda_seq': params['l2_lambda']    # Use l2_lambda for sequence layers
        }
        
        print(f"Creating model: dim={params['model_dim']}, comb_layers={combined_hidden_layers}, tab_layers={tab_hidden_states}")
        
        model = CombinedModel(**model_config)
        
        # Setup optimizer
        if params['optimizer'] == 'adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=params['learning_rate'],
                beta_1=params['beta_1'],
                beta_2=params['beta_2']
            )
        else:  # rmsprop
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=params['learning_rate']
            )
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc_roc"),
                tf.keras.metrics.AUC(curve="PR", name="auc_pr"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )
        
        return model
    
    def objective(self, trial: optuna.Trial, train_data: Tuple, val_data: Tuple,
                input_shape_tabular: int, n_event_code: int) -> float:
        """Objective function for Optuna optimization - FULL VALIDATION SET USED."""
        
        val_data_subset = val_data  # Use full validation set

        # Start MLflow run with auto-generated name
        with mlflow.start_run(nested=True) as run:
            # Suggest hyperparameters
            params = self.suggest_hyperparameters(trial)
            
            # Log parameters to MLflow
            mlflow.log_params(params)
            
            # Set trial user attributes to help with MLflow naming
            trial.set_user_attr("trial_name", f"trial_{trial.number}")
            trial.set_user_attr("mlflow_run_name", run.info.run_name)
            trial.set_user_attr("model_dim", params['model_dim'])
            trial.set_user_attr("learning_rate", params['learning_rate'])
            trial.set_user_attr("batch_size", params['batch_size'])
            
            # Print trial information
            print(f"[*] Starting trial: {trial.number} (MLflow run: {run.info.run_name})")
            if hasattr(trial, 'study') and hasattr(trial.study, 'study_name'):
                print(f"   Study: {trial.study.study_name}")

            try:
                # Apply data balancing if enabled
                train_inputs, y_train = train_data
                if params['use_smote']:
                    X_tab_balanced, y_balanced = self.apply_class_balancing(
                        train_inputs[0], y_train
                    )
                    train_inputs = (X_tab_balanced, *train_inputs[1:])
                    y_train = y_balanced

                # Create model
                model = self.create_model(params, input_shape_tabular, n_event_code)

                # Setup class weights if enabled
                class_weights = None
                if params['use_class_weights']:
                    class_weights = self.compute_class_weights(y_train)

                # Train model
                train_flags = {"train_combined": True}

                # Custom pruning callback
                class PruningCallback(tf.keras.callbacks.Callback):
                    def __init__(self, trial, monitor='val_f1_score'):
                        self.trial = trial
                        self.monitor = monitor

                    def on_epoch_end(self, epoch, logs=None):
                        logs = logs or {}
                        current_value = logs.get(self.monitor, 0)
                        self.trial.report(current_value, epoch)
                        if self.trial.should_prune():
                            raise optuna.TrialPruned()

                # Setup callbacks
                callbacks = []
                if self.config.enable_pruning:
                    callbacks.append(PruningCallback(trial))

                # Early stopping
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_f1_score',
                    patience=params['early_stopping_patience'],
                    min_delta=params['early_stopping_min_delta'],
                    restore_best_weights=True,
                    mode='max'
                )
                callbacks.append(early_stopping)

                # Train the model
                history = train_combined_model(
                    model,
                    train_data=(train_inputs, y_train),
                    val_data=val_data_subset,
                    epochs=params['epochs'],
                    train_flags=train_flags,
                    verbose=False,
                    val_every_n_epochs=10
                )

                # Extract best F1 score
                val_f1_scores = history.get('val_f1_score', [])
                if val_f1_scores:
                    best_f1 = max(val_f1_scores)
                else:
                    val_inputs, y_val = val_data_subset
                    val_pred = model.predict(val_inputs)
                    val_pred_binary = (val_pred.flatten() > 0.5).astype(int)
                    best_f1 = f1_score(y_val, val_pred_binary, average='binary')

                # Log final metrics
                mlflow.log_metric("f1_score", best_f1)
                mlflow.log_metric("final_epoch", len(val_f1_scores))
                for metric_name, values in history.items():
                    if values and 'val_' in metric_name:
                        mlflow.log_metric(f"best_{metric_name}", max(values))

                # Clean up
                del model
                tf.keras.backend.clear_session()

                return best_f1

            except optuna.TrialPruned:
                mlflow.log_metric("pruned", 1)
                raise
            except Exception as e:
                mlflow.log_param("error", str(e))
                print(f"Trial {trial.number} failed: {e}")
                return 0.0

    
    def tune(self, train_data: Tuple, val_data: Tuple, input_shape_tabular: int, 
             n_event_code: int) -> Dict[str, Any]:
        """Run hyperparameter tuning."""
        print(f"Starting Phase 1 hyperparameter tuning...")
        print(f"Number of trials: {self.config.n_trials}")
        
        # Create study
        study_name = f"{self.model_name}_{self.config.study_name}"
        study, mlflow_callback = self.create_study(study_name)
        
        # Optimize
        study.optimize(
            lambda trial: self.objective(
                trial, train_data, val_data, input_shape_tabular, n_event_code
            ),
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds
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
    
    def train_with_best_params(self, train_data: Tuple, val_data: Tuple,
                              input_shape_tabular: int, n_event_code: int,
                              best_params_path: str = None) -> Tuple[CombinedModel, Dict]:
        """Train model with best parameters."""
        
        # Load best parameters
        if best_params_path is None:
            best_params_path = os.path.join(
                self.config.tuning_results_dir, 
                f"{self.model_name}_best_params.json"
            )
        
        with open(best_params_path, 'r') as f:
            results = json.load(f)
            best_params = results['best_params']

        # Merge best parameters with defaults used during tuning
        default_params = {
            'layer': 2,
            'dropout_rate_seq': 0.1,
            'droput_rate_tab': 0.1,
            'dropout_rate_comb': 0.1,
            'epochs': 10,
            'l2_lambda': 1e-4,
            'tab_hidden_states': '32_16',
            'use_smote': False,
            'optimizer': 'adam',
            'beta_1': 0.9,
            'beta_2': 0.999,
            'early_stopping_patience': 1,
            'early_stopping_min_delta': 0.001,
        }

        # ensure essential params are present
        for key, value in default_params.items():
            best_params.setdefault(key, value)
        
        print(f"Training Phase 1 model with best parameters...")
        print(f"Best parameters: {best_params}")
        
        # Apply data balancing if enabled
        train_inputs, y_train = train_data
        if best_params.get('use_smote', False):
            X_tab_balanced, y_balanced = self.apply_class_balancing(
                train_inputs[0], y_train
            )
            train_inputs = (X_tab_balanced, *train_inputs[1:])
            y_train = y_balanced
        
        # Create model
        model = self.create_model(best_params, input_shape_tabular, n_event_code)
        
        # Train model
        train_flags = {"train_combined": True}
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_f1_score',
            patience=best_params['early_stopping_patience'],
            min_delta=best_params['early_stopping_min_delta'],
            restore_best_weights=True,
            mode='max'
        )
        
        history = train_combined_model(
            model,
            train_data=(train_inputs, y_train),
            val_data=val_data,
            epochs=best_params['epochs'],
            train_flags=train_flags,
            verbose=True
        )
        
        return model, history
