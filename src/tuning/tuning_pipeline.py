"""Complete pipeline for hyperparameter tuning both phases."""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple, Optional

import mlflow
import mlflow.tensorflow
from sklearn.model_selection import train_test_split

from src.tuning.base_tuner import TuningConfig
from src.tuning.phase1_tuner import Phase1Tuner
from src.tuning.phase2_tuner import Phase2Tuner
from src.components.data_transformation import CombinedTransformConfig
from src.components.data_ingestion import WikiDataIngestionConfig
from src.config.configuration import Phase1TrainingConfig
from src.utils.io import load_transformed_data

class HyperparameterTuningPipeline:
    """Complete pipeline for hyperparameter tuning."""
    
    def __init__(self, tuning_config: TuningConfig = None):
        self.tuning_config = tuning_config or TuningConfig()
        self.phase1_tuner = Phase1Tuner(self.tuning_config)
        self.phase2_tuner = Phase2Tuner(self.tuning_config)
        
        # Setup MLflow
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri(self.tuning_config.mlflow_tracking_uri)
        mlflow.set_experiment(self.tuning_config.mlflow_experiment_name)
        
        # Log tuning configuration
        with mlflow.start_run(run_name="hyperparameter_tuning_pipeline"):
            mlflow.log_params({
                'n_trials': self.tuning_config.n_trials,
                'n_startup_trials': self.tuning_config.n_startup_trials,
                'enable_smote': self.tuning_config.enable_smote,
                'enable_pruning': self.tuning_config.enable_pruning,
                'max_epochs': self.tuning_config.max_epochs,
                'tune_phase1': self.tuning_config.tune_phase1,
                'tune_phase2': self.tuning_config.tune_phase2,
            })
            



    def load_data(self) -> Tuple[Tuple, Tuple]:
        """Use only 20% of the validation set for tuning (split into train/val)."""
        print("Loading 20% subset of validation data for hyperparameter tuning...")

        # Load config and data
        cfg = CombinedTransformConfig()
        train_cfg = Phase1TrainingConfig()

        # Load only validation data
        (_, _), (val_inputs, y_val) = load_transformed_data(
            cfg, train_cfg.selected_indices, train_cfg.sample_size, train_cfg.random_state
        )

        # Step 1: Subsample 20% of validation set
        val_arrays = val_inputs
        n_samples = y_val.shape[0]
        val_subset_size = int(0.2 * n_samples)

        indices = np.arange(n_samples)
        val_subset_indices, _ = train_test_split(
            indices, train_size=val_subset_size, stratify=y_val, random_state=42
        )

        inputs_subset = tuple(arr[val_subset_indices] for arr in val_arrays)
        y_subset = y_val[val_subset_indices]

        # Step 2: Split 20% subset into tuning train/val sets (e.g., 75/25 split)
        tuning_train_indices, tuning_val_indices = train_test_split(
            np.arange(val_subset_size), test_size=0.25, stratify=y_subset, random_state=42
        )

        inputs_tune = tuple(arr[tuning_train_indices] for arr in inputs_subset)
        inputs_holdout = tuple(arr[tuning_val_indices] for arr in inputs_subset)
        y_tune = y_subset[tuning_train_indices]
        y_holdout = y_subset[tuning_val_indices]

        print(f"Original validation set size: {n_samples}")
        print(f"Tuning subset size (20% of val): {val_subset_size}")
        print(f"Tuning train size: {inputs_tune[0].shape[0]}")
        print(f"Tuning val size: {inputs_holdout[0].shape[0]}")
        print(f"Tuning train label distribution: {np.bincount(y_tune)}")
        print(f"Tuning val label distribution: {np.bincount(y_holdout)}")

        return (inputs_tune, y_tune), (inputs_holdout, y_holdout)


    
    def get_data_info(self, train_data: Tuple) -> Dict[str, Any]:
        """Get data information for model creation."""
        import pandas as pd
        
        train_inputs, y_train = train_data
        
        # Get configuration
        i_cfg = WikiDataIngestionConfig()
        category_dict = pd.read_pickle(i_cfg.category_dict_path)
        
        return {
            'input_shape_tabular': train_inputs[0].shape[1],
            'n_event_code': category_dict['no_cat'] + 1,
            'n_samples': len(y_train),
            'n_features': sum(x.shape[1] if x.ndim > 1 else 1 for x in train_inputs)
        }
    
    def _convert_numpy_to_python(self, obj):
        """Convert NumPy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_python(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
    
    def run_phase1_tuning(self, train_data: Tuple, val_data: Tuple) -> Dict[str, Any]:
        """Run Phase 1 hyperparameter tuning."""
        if not self.tuning_config.tune_phase1:
            print("Phase 1 tuning is disabled.")
            return {}
            
        print("\n" + "="*60)
        print("PHASE 1 HYPERPARAMETER TUNING")
        print("="*60)
        
        # Get data info
        data_info = self.get_data_info(train_data)
        
        # Run tuning
        results = self.phase1_tuner.tune(
            train_data, val_data, 
            data_info['input_shape_tabular'], 
            data_info['n_event_code']
        )
        
        # Log results to MLflow
        with mlflow.start_run(run_name="phase1_tuning_results", nested=True):
            mlflow.log_params(results['best_params'])
            mlflow.log_metric("best_f1_score", results['best_value'])
            mlflow.log_metric("n_trials", len(results['study'].trials))
            
            # Save study object
            study_path = os.path.join(
                self.tuning_config.tuning_results_dir, 
                "phase1_study.pkl"
            )
            import pickle
            with open(study_path, 'wb') as f:
                pickle.dump(results['study'], f)
            mlflow.log_artifact(study_path)
        
        return results
    
    def run_phase2_tuning(self, train_data: Tuple, val_data: Tuple, 
                         combined_model=None) -> Dict[str, Any]:
        """Run Phase 2 hyperparameter tuning."""
        if not self.tuning_config.tune_phase2:
            print("Phase 2 tuning is disabled.")
            return {}
            
        print("\n" + "="*60)
        print("PHASE 2 HYPERPARAMETER TUNING")
        print("="*60)
        
        # Load combined model if not provided
        if combined_model is None:
            # Try to load from best Phase 1 results
            try:
                best_params_path = os.path.join(
                    self.tuning_config.tuning_results_dir, 
                    "phase1_combined_best_params.json"
                )
                data_info = self.get_data_info(train_data)
                combined_model, _ = self.phase1_tuner.train_with_best_params(
                    train_data, val_data,
                    data_info['input_shape_tabular'], 
                    data_info['n_event_code'],
                    best_params_path
                )
            except Exception as e:
                print(f"Could not load Phase 1 model: {e}")
                print("Please run Phase 1 tuning first or provide a trained combined model.")
                return {}
        
        # Run tuning
        results = self.phase2_tuner.tune(combined_model, train_data, val_data)
        
        # Log results to MLflow
        with mlflow.start_run(run_name="phase2_tuning_results", nested=True):
            mlflow.log_params(results['best_params'])
            mlflow.log_metric("best_f1_score", results['best_value'])
            mlflow.log_metric("n_trials", len(results['study'].trials))
            
            # Save study object
            study_path = os.path.join(
                self.tuning_config.tuning_results_dir, 
                "phase2_study.pkl"
            )
            import pickle
            with open(study_path, 'wb') as f:
                pickle.dump(results['study'], f)
            mlflow.log_artifact(study_path)
        
        return results
    
    def save_best_results(self, train_data: Tuple, val_data: Tuple) -> Dict[str, Any]:
        """Save complete parameter set (tuned + defaults) and metrics."""
        print("\n" + "="*60)
        print("SAVING COMPLETE PARAMETERS AND METRICS")
        print("="*60)
        
        results = {}
        
        # Save Phase 1 results
        if self.tuning_config.tune_phase1:
            print("\nSaving Phase 1 complete parameters...")
            
            # Load best parameters
            best_params_path = os.path.join(
                self.tuning_config.tuning_results_dir, 
                "phase1_combined_best_params.json"
            )
            
            if os.path.exists(best_params_path):
                with open(best_params_path, 'r') as f:
                    phase1_results = json.load(f)
                
                # Get default parameters that were used during tuning
                default_params = {
                    'layer': 2,
                    'dropout_rate_seq': 0.1,
                    'dropout_rate_tab': 0.1,
                    'dropout_rate_comb': 0.1,
                    'epochs': 10,
                    'l2_lambda': 0.0001,
                    'tab_hidden_states': '32_16',
                    'use_smote': False,
                    'optimizer': 'adam',
                    'beta_1': 0.9,
                    'beta_2': 0.999,
                    'early_stopping_patience': 1,
                    'early_stopping_min_delta': 0.001,
                }
                
                # Merge tuned parameters with defaults
                best_params = phase1_results['best_params'].copy()
                for key, value in default_params.items():
                    if key not in best_params:
                        best_params[key] = value
                
                # Create comprehensive parameter set
                complete_params = {
                    # Tuned parameters
                    'model_dim': best_params.get('model_dim', 256),
                    'learning_rate': best_params.get('learning_rate', 0.001),
                    'batch_size': best_params.get('batch_size', 32),
                    'combined_hidden_layers': best_params.get('combined_hidden_layers', '128_64'),
                    'use_class_weights': best_params.get('use_class_weights', True),
                    
                    # Default parameters used during tuning
                    'layer': best_params.get('layer', 2),
                    'dropout_rate_seq': best_params.get('dropout_rate_seq', 0.1),
                    'dropout_rate_tab': best_params.get('dropout_rate_tab', 0.1),
                    'dropout_rate_comb': best_params.get('dropout_rate_comb', 0.1),
                    'epochs': best_params.get('epochs', 10),
                    'l2_lambda': best_params.get('l2_lambda', 0.0001),
                    'tab_hidden_states': best_params.get('tab_hidden_states', '32_16'),
                    'use_smote': best_params.get('use_smote', False),
                    'optimizer': best_params.get('optimizer', 'adam'),
                    'beta_1': best_params.get('beta_1', 0.9),
                    'beta_2': best_params.get('beta_2', 0.999),
                    'early_stopping_patience': best_params.get('early_stopping_patience', 1),
                    'early_stopping_min_delta': best_params.get('early_stopping_min_delta', 0.001),
                    
                    # Additional metadata
                    'best_f1_score': phase1_results['best_value'],
                    'n_trials': phase1_results['n_trials'],
                    'study_name': phase1_results['study_name'],
                    'tuning_timestamp': phase1_results.get('timestamp', 'unknown')
                }
                
                # Save complete parameters
                complete_params_path = os.path.join(
                    self.tuning_config.tuning_results_dir, 
                    "phase1_complete_parameters.json"
                )
                with open(complete_params_path, 'w') as f:
                    json.dump(complete_params, f, indent=2)
                
                results['phase1_complete_params'] = complete_params
                results['phase1_best_f1'] = phase1_results['best_value']
                results['phase1_n_trials'] = phase1_results['n_trials']
                
                print(f"Phase 1 best F1 score: {phase1_results['best_value']:.4f}")
                print(f"Phase 1 complete parameters saved to: {complete_params_path}")
                print(f"Total parameters saved: {len(complete_params)}")
            else:
                print("Phase 1 best parameters file not found!")
        
        # Save Phase 2 results
        if self.tuning_config.tune_phase2:
            print("\nSaving Phase 2 complete parameters...")
            
            # Load best parameters
            best_params_path = os.path.join(
                self.tuning_config.tuning_results_dir, 
                "phase2_ocan_best_params.json"
            )
            
            if os.path.exists(best_params_path):
                with open(best_params_path, 'r') as f:
                    phase2_results = json.load(f)
                
                # Get default parameters for Phase 2
                default_params_phase2 = {
                    'discriminator_lr': 0.0001,
                    'generator_lr': 0.0001,
                    'gan_epochs': 50,
                    'discriminator_hidden_layers': [64, 32],
                    'generator_hidden_layers': [32, 64],
                    'noise_dim': 100,
                    'lambda_recon': 1.0,
                    'lambda_adv': 1.0,
                    'beta_1': 0.9,
                    'beta_2': 0.999,
                    'batch_size': 32,
                }
                
                # Merge tuned parameters with defaults
                best_params = phase2_results['best_params'].copy()
                for key, value in default_params_phase2.items():
                    if key not in best_params:
                        best_params[key] = value
                
                # Create complete parameter set for Phase 2
                complete_params_phase2 = {
                    **best_params,
                    'best_f1_score': phase2_results['best_value'],
                    'n_trials': phase2_results['n_trials'],
                    'study_name': phase2_results['study_name'],
                    'tuning_timestamp': phase2_results.get('timestamp', 'unknown')
                }
                
                # Save complete parameters
                complete_params_path = os.path.join(
                    self.tuning_config.tuning_results_dir, 
                    "phase2_complete_parameters.json"
                )
                with open(complete_params_path, 'w') as f:
                    json.dump(complete_params_phase2, f, indent=2)
                
                results['phase2_complete_params'] = complete_params_phase2
                results['phase2_best_f1'] = phase2_results['best_value']
                results['phase2_n_trials'] = phase2_results['n_trials']
                
                print(f"Phase 2 best F1 score: {phase2_results['best_value']:.4f}")
                print(f"Phase 2 complete parameters saved to: {complete_params_path}")
                print(f"Total parameters saved: {len(complete_params_phase2)}")
            else:
                print("Phase 2 best parameters file not found!")
        
        return results
    
    def run_complete_tuning(self) -> Dict[str, Any]:
        """Run complete hyperparameter tuning pipeline."""
        print("Starting complete hyperparameter tuning pipeline...")
        
        # Setup GPU
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                print(f"Could not set memory growth for {gpu}: {e}")
        
        # Load data
        train_data, val_data = self.load_data()
        
        # Run Phase 1 tuning
        phase1_results = self.run_phase1_tuning(train_data, val_data)
        
        # Run Phase 2 tuning
        phase2_results = self.run_phase2_tuning(train_data, val_data)
        
        # Save best results (parameters and metrics only)
        best_results = self.save_best_results(train_data, val_data)
        
        # Combine results
        results = {
            'phase1_tuning': phase1_results,
            'phase2_tuning': phase2_results,
            'best_results': best_results
        }
        
        # Generate summary report
        self.generate_summary_report(results)
        
        return results
    
    def generate_summary_report(self, results: Dict[str, Any]):
        """Generate a summary report of tuning results."""
        print("\n" + "="*60)
        print("TUNING SUMMARY REPORT")
        print("="*60)
        
        # Phase 1 results
        if 'phase1_tuning' in results and results['phase1_tuning']:
            phase1_results = results['phase1_tuning']
            print(f"\nPhase 1 Results:")
            print(f"  Best F1 Score: {phase1_results['best_value']:.4f}")
            print(f"  Number of trials: {len(phase1_results['study'].trials)}")
            print(f"  Best parameters: {phase1_results['best_params']}")
        
        # Phase 2 results
        if 'phase2_tuning' in results and results['phase2_tuning']:
            phase2_results = results['phase2_tuning']
            print(f"\nPhase 2 Results:")
            print(f"  Best F1 Score: {phase2_results['best_value']:.4f}")
            print(f"  Number of trials: {len(phase2_results['study'].trials)}")
            print(f"  Best parameters: {phase2_results['best_params']}")
        
        # Save summary
        summary_path = os.path.join(
            self.tuning_config.tuning_results_dir, 
            "tuning_summary.json"
        )
        
        summary = {
            'phase1_best_f1': results.get('phase1_tuning', {}).get('best_value', 0),
            'phase2_best_f1': results.get('phase2_tuning', {}).get('best_value', 0),
            'phase1_best_params': results.get('phase1_tuning', {}).get('best_params', {}),
            'phase2_best_params': results.get('phase2_tuning', {}).get('best_params', {}),
            'tuning_config': {
                'n_trials': self.tuning_config.n_trials,
                'enable_smote': self.tuning_config.enable_smote,
                'enable_pruning': self.tuning_config.enable_pruning,
                'max_epochs': self.tuning_config.max_epochs,
            }
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to: {summary_path}")
        print(f"MLflow tracking URI: {self.tuning_config.mlflow_tracking_uri}")
        print(f"Results directory: {self.tuning_config.tuning_results_dir}")
        
        print("\nTuning completed! Best parameters and metrics saved.")
        print("Note: Model weights are not saved - only hyperparameters and performance metrics.")
        
        # List all parameter files created
        print("\nParameter files created:")
        param_files = [
            "phase1_combined_best_params.json",
            "phase1_complete_parameters.json",
            "phase2_ocan_best_params.json", 
            "phase2_complete_parameters.json",
            "tuning_summary.json"
        ]
        
        for param_file in param_files:
            file_path = os.path.join(self.tuning_config.tuning_results_dir, param_file)
            if os.path.exists(file_path):
                print(f"  [*] {param_file}")
            else:
                print(f"  [MISSING] {param_file} (not created)")
        
        print("\nUse 'phase1_complete_parameters.json' for comprehensive parameter set including all defaults.")


def main():
    """Main function for running hyperparameter tuning."""
    # Configuration
    tuning_config = TuningConfig(
        n_trials=100,              # Optional: increase if you expect more trials to fit in 10h
        n_startup_trials=1,
        enable_smote=False,        # Based on logs: keep this off
        enable_pruning=True,       # Still useful with long runtime
        max_epochs=10,
        tune_phase1=True,
        tune_phase2=False,
        timeout_seconds=36000      # ⏱ 10 hours = 10 × 60 × 60 = 36,000 seconds
    )
    
    # Run tuning
    pipeline = HyperparameterTuningPipeline(tuning_config)
    results = pipeline.run_complete_tuning()
    
    return results


if __name__ == "__main__":
    main()
