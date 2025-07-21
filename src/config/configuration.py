import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json


@dataclass
class Phase1TrainingConfig:
    """Configuration for Phase 1 model training."""

    # Data configuration
    selected_indices: Optional[List[int]] = None
    sample_size: int = 'all'
    random_state: int = 42

    # Model architecture
    max_len: int = 25
    max_code: int = 50
    combined_hidden_layers: List[int] = field(default_factory=lambda: [128, 64])
    dropout_rate_comb: float = 0.2
    dropout_rate_seq: float = 0.1
    dropout_rate_tab: float = 0.1
    layer: int = 1
    tab_hidden_states: List[int] = field(default_factory=lambda: [64, 32])
    model_dim: int = 256
    is_public_dataset: bool = True

    # Training configuration
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 5e-4
    train_flags: Dict[str, bool] = field(default_factory=lambda: {"train_combined": True})

    # Paths
    artifact_dir: str = "artifacts/combined_phase1"
    model_config_path: str = "artifacts/combined_phase1/model_config.json"
    model_path: str = "artifacts/combined_phase1/model"
    training_log_path: str = "artifacts/combined_phase1/training_log.json"

    def to_model_config(self, input_shape_tabular: int, n_event_code: int) -> Dict[str, Any]:
        return {
            "input_shape_tabular": input_shape_tabular,
            "max_len": self.max_len,
            "max_code": self.max_code,
            "combined_hidden_layers": self.combined_hidden_layers,
            "dropout_rate_comb": self.dropout_rate_comb,
            "dropout_rate_seq": self.dropout_rate_seq,
            "droput_rate_tab": self.dropout_rate_tab,
            "layer": self.layer,
            "tab_hidden_states": self.tab_hidden_states,
            "batch_size": self.batch_size,
            "n_event_code": n_event_code,
            "model_dim": self.model_dim,
            "selected_indices": self.selected_indices,
            "is_public_dataset": self.is_public_dataset,
        }

    def get_model_params(self, input_shape_tabular: int, n_event_code: int) -> Dict[str, Any]:
        model_config = self.to_model_config(input_shape_tabular, n_event_code)
        return {k: v for k, v in model_config.items() if k != 'selected_indices'}

    def load_best_hyperparameters(self, best_params_path: str = "tuning_results/phase1_combined_best_params.json") -> 'Phase1TrainingConfig':
        if not os.path.exists(best_params_path):
            print(f"Warning: Best parameters file not found at {best_params_path}")
            return self

        try:
            with open(best_params_path, 'r') as f:
                results = json.load(f)
                best_params = results.get('best_params', {})

            for key, val in best_params.items():
                if key in ['combined_hidden_layers', 'tab_hidden_states'] and isinstance(val, str):
                    setattr(self, key, [int(x) for x in val.split('_')])
                else:
                    setattr(self, key, val)

            print(f"[*] Loaded best hyperparameters from {best_params_path}")
            print(f"   Best F1 Score: {results.get('best_value', 'N/A'):.4f}")
            return self

        except Exception as e:
            print(f"Error loading best parameters: {e}")
            return self

    @classmethod
    def from_best_hyperparameters(cls, best_params_path: str = "tuning_results/phase1_combined_best_params.json") -> 'Phase1TrainingConfig':
        return cls().load_best_hyperparameters(best_params_path)

    @classmethod
    def from_complete_parameters(cls, complete_params_path: str = "tuning_results/phase1_complete_parameters.json") -> 'Phase1TrainingConfig':
        if not os.path.exists(complete_params_path):
            print(f"Warning: Complete parameters file not found at {complete_params_path}")
            return cls()

        try:
            with open(complete_params_path, 'r') as f:
                params = json.load(f)

            def convert_layer_string(layer_str):
                return [int(x) for x in layer_str.split('_')] if isinstance(layer_str, str) else layer_str

            new_config = cls(
                model_dim=params.get('model_dim', 256),
                learning_rate=params.get('learning_rate', 5e-4),
                batch_size=params.get('batch_size', 32),
                epochs=params.get('epochs', 15),
                max_len=params.get('max_len', 25),
                max_code=params.get('max_code', 50),
                combined_hidden_layers=convert_layer_string(params.get('combined_hidden_layers', [128, 64])),
                tab_hidden_states=convert_layer_string(params.get('tab_hidden_states', [64, 32])),
                layer=params.get('layer', 1),
                dropout_rate_comb=params.get('dropout_rate_comb', 0.2),
                dropout_rate_seq=params.get('dropout_rate_seq', 0.1),
                dropout_rate_tab=params.get('dropout_rate_tab', 0.1),
                selected_indices=params.get('selected_indices', [i for i in range(19) if i not in [7, 8, 16]]),
                sample_size=params.get('sample_size', 'all'),
                random_state=params.get('random_state', 42),
                is_public_dataset=params.get('is_public_dataset', True),
                artifact_dir=params.get('artifact_dir', "artifacts/combined_phase1"),
                model_config_path=params.get('model_config_path', "artifacts/combined_phase1/model_config.json"),
                model_path=params.get('model_path', "artifacts/combined_phase1/model"),
                training_log_path=params.get('training_log_path', "artifacts/combined_phase1/training_log.json"),
            )

            new_config.tuning_metadata = {
                'best_f1_score': params.get('best_f1_score'),
                'n_trials': params.get('n_trials'),
                'study_name': params.get('study_name'),
                'tuning_timestamp': params.get('tuning_timestamp'),
                'optimizer': params.get('optimizer', 'adam'),
                'beta_1': params.get('beta_1', 0.9),
                'beta_2': params.get('beta_2', 0.999),
                'l2_lambda': params.get('l2_lambda', 0.0001),
                'use_class_weights': params.get('use_class_weights', False),
                'use_smote': params.get('use_smote', False),
                'early_stopping_patience': params.get('early_stopping_patience', 1),
                'early_stopping_min_delta': params.get('early_stopping_min_delta', 0.001),
            }

            print(f"[*] Loaded complete parameters from {complete_params_path}")
            if 'best_f1_score' in params:
                print(f"   Best F1 Score: {params['best_f1_score']:.4f}")
            return new_config

        except Exception as e:
            print(f"Error loading complete parameters: {e}")
            return cls()


@dataclass
class PredictionConfig:
    """Configuration for model prediction pipeline."""

    model_dir: str = "artifacts/combined_phase1/model"
    config_path: str = "artifacts/combined_phase1/model_config.json"
    preprocessor_path: str = "artifacts/preprocessor.pkl"
    category_dict_path: str = "data/processed/wiki/category_dict.pkl"

    default_max_len: int = 25
    default_max_code: int = 50
    default_input_shape_tabular: int = 9
    default_combined_hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    default_dropout_rate_comb: float = 0.2
    default_dropout_rate_seq: float = 0.1
    default_dropout_rate_tab: float = 0.1
    default_tab_hidden_states: List[int] = field(default_factory=lambda: [32, 16])
    default_n_event_code: int = 273041
    default_model_dim: int = 256
    default_layer: int = 1
    default_is_public_dataset: bool = False


PHASE1_TRAINING_CONFIG = Phase1TrainingConfig()
PREDICTION_CONFIG = PredictionConfig()