# src/config/configuration.py

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class Phase1TrainingConfig:
    """Configuration for Phase 1 model training."""
    
    # Data configuration
    selected_indices: List[int] = None
    sample_size: int = 'all'
    random_state: int = 42
    
    # Model architecture
    max_len: int = 25
    max_code: int = 50
    combined_hidden_layers: List[int] = None
    dropout_rate_comb: float = 0.2
    dropout_rate_seq: float = 0.1
    dropout_rate_tab: float = 0.1
    layer: int = 1
    tab_hidden_states: List[int] = None
    model_dim: int = 256
    is_public_dataset: bool = True
    
    # Training configuration
    batch_size: int = 32
    epochs: int = 15
    learning_rate: float = 5e-4  # Reduced from 1e-3 for better stability
    train_flags: Dict[str, bool] = None
    
    # Paths
    artifact_dir: str = "artifacts/combined_phase1"
    model_config_path: str = "artifacts/combined_phase1/model_config.json"
    model_path: str = "artifacts/combined_phase1/model"
    training_log_path: str = "artifacts/combined_phase1/training_log.json"
    
    def __post_init__(self):
        """Set default values for mutable fields."""
        if self.selected_indices is None:
            self.selected_indices = [0,1,2,3,4,5,6,7,8]  # 9 features: indices 0-8
        
        if self.combined_hidden_layers is None:
            self.combined_hidden_layers = [128, 64]
            
        if self.tab_hidden_states is None:
            self.tab_hidden_states = [64, 32]
            
        if self.train_flags is None:
            self.train_flags = {"train_combined": True}
    
    def to_model_config(self, input_shape_tabular: int, n_event_code: int) -> Dict[str, Any]:
        """Convert to model configuration dictionary."""
        return {
            "input_shape_tabular": input_shape_tabular,
            "max_len": self.max_len,
            "max_code": self.max_code,
            "combined_hidden_layers": self.combined_hidden_layers,
            "dropout_rate_comb": self.dropout_rate_comb,
            "dropout_rate_seq": self.dropout_rate_seq,
            "droput_rate_tab": self.dropout_rate_tab,  # Note: keeping the typo for compatibility
            "layer": self.layer,
            "tab_hidden_states": self.tab_hidden_states,
            "batch_size": self.batch_size,
            "n_event_code": n_event_code,
            "model_dim": self.model_dim,
            "selected_indices": self.selected_indices,
            "is_public_dataset": self.is_public_dataset,
        }
    
    def get_model_params(self, input_shape_tabular: int, n_event_code: int) -> Dict[str, Any]:
        """Get parameters for model instantiation (excluding non-model params)."""
        model_config = self.to_model_config(input_shape_tabular, n_event_code)
        # Remove parameters that are not accepted by CombinedModel constructor
        excluded_params = ['selected_indices']
        return {k: v for k, v in model_config.items() if k not in excluded_params}

    def load_best_hyperparameters(self, best_params_path: str = "tuning_results/phase1_combined_best_params.json") -> 'Phase1TrainingConfig':
        """Load best hyperparameters from tuning results and return updated config."""
        import json
        
        if not os.path.exists(best_params_path):
            print(f"Warning: Best parameters file not found at {best_params_path}")
            print("Using default configuration.")
            return self
            
        try:
            with open(best_params_path, 'r') as f:
                results = json.load(f)
                best_params = results.get('best_params', {})
            
            # Create a new config instance to avoid modifying the original
            new_config = Phase1TrainingConfig(
                selected_indices=self.selected_indices,
                sample_size=self.sample_size,
                random_state=self.random_state,
                max_len=self.max_len,
                max_code=self.max_code,
                is_public_dataset=self.is_public_dataset,
                artifact_dir=self.artifact_dir,
                model_config_path=self.model_config_path,
                model_path=self.model_path,
                training_log_path=self.training_log_path,
                train_flags=self.train_flags
            )
            
            # Apply best hyperparameters
            if 'model_dim' in best_params:
                new_config.model_dim = best_params['model_dim']
                
            if 'learning_rate' in best_params:
                new_config.learning_rate = best_params['learning_rate']
                
            if 'batch_size' in best_params:
                new_config.batch_size = best_params['batch_size']
                
            if 'combined_hidden_layers' in best_params:
                # Convert string format "64_32" to list [64, 32]
                layers_str = best_params['combined_hidden_layers']
                if isinstance(layers_str, str):
                    new_config.combined_hidden_layers = [int(x) for x in layers_str.split('_')]
                else:
                    new_config.combined_hidden_layers = layers_str
                    
            if 'dropout_rate_comb' in best_params:
                new_config.dropout_rate_comb = best_params['dropout_rate_comb']
                
            if 'dropout_rate_seq' in best_params:
                new_config.dropout_rate_seq = best_params['dropout_rate_seq']
                
            if 'dropout_rate_tab' in best_params:
                new_config.dropout_rate_tab = best_params['dropout_rate_tab']
                
            if 'tab_hidden_states' in best_params:
                # Convert string format "32_16" to list [32, 16]
                layers_str = best_params['tab_hidden_states']
                if isinstance(layers_str, str):
                    new_config.tab_hidden_states = [int(x) for x in layers_str.split('_')]
                else:
                    new_config.tab_hidden_states = layers_str
                    
            if 'layer' in best_params:
                new_config.layer = best_params['layer']
                
            if 'epochs' in best_params:
                new_config.epochs = best_params['epochs']
            
            print(f"[OK] Loaded best hyperparameters from {best_params_path}")
            print(f"   Best F1 Score: {results.get('best_value', 'N/A'):.4f}")
            print(f"   Model Dim: {new_config.model_dim}")
            print(f"   Learning Rate: {new_config.learning_rate}")
            print(f"   Batch Size: {new_config.batch_size}")
            print(f"   Combined Hidden Layers: {new_config.combined_hidden_layers}")
            print(f"   Tabular Hidden States: {new_config.tab_hidden_states}")
            
            return new_config
            
        except Exception as e:
            print(f"Error loading best parameters: {e}")
            print("Using default configuration.")
            return self

    @classmethod
    def from_best_hyperparameters(cls, best_params_path: str = "tuning_results/phase1_combined_best_params.json") -> 'Phase1TrainingConfig':
        """Create a new Phase1TrainingConfig instance from best hyperparameters."""
        default_config = cls()
        return default_config.load_best_hyperparameters(best_params_path)

    @classmethod
    def from_complete_parameters(cls, complete_params_path: str = "tuning_results/phase1_complete_parameters.json") -> 'Phase1TrainingConfig':
        """Create a new Phase1TrainingConfig instance from complete parameters JSON."""
        import json
        
        if not os.path.exists(complete_params_path):
            print(f"Warning: Complete parameters file not found at {complete_params_path}")
            print("Using default configuration.")
            return cls()
            
        try:
            with open(complete_params_path, 'r') as f:
                params = json.load(f)
            
            # Convert string representations to appropriate types
            def convert_layer_string(layer_str):
                """Convert '64_32' to [64, 32]"""
                if isinstance(layer_str, str):
                    return [int(x) for x in layer_str.split('_')]
                return layer_str
            
            # Create new config with complete parameters
            new_config = cls(
                # Core training parameters
                model_dim=params.get('model_dim', 256),
                learning_rate=params.get('learning_rate', 5e-4),
                batch_size=params.get('batch_size', 32),
                epochs=params.get('epochs', 15),
                
                # Architecture parameters
                max_len=params.get('max_len', 25),
                max_code=params.get('max_code', 50),
                combined_hidden_layers=convert_layer_string(params.get('combined_hidden_layers', [128, 64])),
                tab_hidden_states=convert_layer_string(params.get('tab_hidden_states', [64, 32])),
                layer=params.get('layer', 1),
                
                # Regularization parameters
                dropout_rate_comb=params.get('dropout_rate_comb', 0.2),
                dropout_rate_seq=params.get('dropout_rate_seq', 0.1),
                dropout_rate_tab=params.get('dropout_rate_tab', 0.1),
                
                # Data parameters
                selected_indices=params.get('selected_indices', [0,1,2,3,4,5,6,7,8]),
                sample_size=params.get('sample_size', 'all'),
                random_state=params.get('random_state', 42),
                is_public_dataset=params.get('is_public_dataset', True),
                
                # Paths (keep defaults unless specified)
                artifact_dir=params.get('artifact_dir', "artifacts/combined_phase1"),
                model_config_path=params.get('model_config_path', "artifacts/combined_phase1/model_config.json"),
                model_path=params.get('model_path', "artifacts/combined_phase1/model"),
                training_log_path=params.get('training_log_path', "artifacts/combined_phase1/training_log.json"),
            )
            
            # Store additional metadata
            new_config.tuning_metadata = {
                'best_f1_score': params.get('best_f1_score', None),
                'n_trials': params.get('n_trials', None),
                'study_name': params.get('study_name', None),
                'tuning_timestamp': params.get('tuning_timestamp', None),
                'optimizer': params.get('optimizer', 'adam'),
                'beta_1': params.get('beta_1', 0.9),
                'beta_2': params.get('beta_2', 0.999),
                'l2_lambda': params.get('l2_lambda', 0.0001),
                'use_class_weights': params.get('use_class_weights', False),
                'use_smote': params.get('use_smote', False),
                'early_stopping_patience': params.get('early_stopping_patience', 1),
                'early_stopping_min_delta': params.get('early_stopping_min_delta', 0.001),
            }
            
            print(f"[OK] Loaded complete parameters from {complete_params_path}")
            if 'best_f1_score' in params:
                print(f"   Best F1 Score: {params['best_f1_score']:.4f}")
            print(f"   Model Dim: {new_config.model_dim}")
            print(f"   Learning Rate: {new_config.learning_rate}")
            print(f"   Batch Size: {new_config.batch_size}")
            print(f"   Epochs: {new_config.epochs}")
            print(f"   Combined Hidden Layers: {new_config.combined_hidden_layers}")
            print(f"   Tabular Hidden States: {new_config.tab_hidden_states}")
            
            return new_config
            
        except Exception as e:
            print(f"Error loading complete parameters: {e}")
            print("Using default configuration.")
            return cls()


@dataclass  
class PredictionConfig:
    """Configuration for model prediction pipeline."""
    
    # Model paths
    model_dir: str = "artifacts/combined_phase1/model"
    config_path: str = "artifacts/combined_phase1/model_config.json"
    preprocessor_path: str = "artifacts/preprocessor.pkl"
    category_dict_path: str = "data/processed/wiki/category_dict.pkl"
    
    # Default model parameters (used when config file is not found)
    default_max_len: int = 25
    default_max_code: int = 50
    default_input_shape_tabular: int = 9  # Updated to match actual feature count
    default_combined_hidden_layers: List[int] = None
    default_dropout_rate_comb: float = 0.2
    default_dropout_rate_seq: float = 0.1
    default_dropout_rate_tab: float = 0.1
    default_tab_hidden_states: List[int] = None
    default_n_event_code: int = 273041  # Should match training config
    default_model_dim: int = 256
    default_layer: int = 1
    default_is_public_dataset: bool = False
    
    def __post_init__(self):
        """Set default values for mutable fields."""
        if self.default_combined_hidden_layers is None:
            self.default_combined_hidden_layers = [64, 32]
            
        if self.default_tab_hidden_states is None:
            self.default_tab_hidden_states = [32, 16]


# Global configuration instances
PHASE1_TRAINING_CONFIG = Phase1TrainingConfig()
PREDICTION_CONFIG = PredictionConfig()