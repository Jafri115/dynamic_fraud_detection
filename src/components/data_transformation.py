import json
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from tqdm import tqdm

from src.exception import CustomException
from src.logger import logging
from src.utils.io import save_object
from src.config.configuration import PREDICTION_CONFIG

@dataclass
class CombinedTransformConfig:
    base_dir: str = os.path.join("data", "processed", "combined")
    train_seq_path: str = os.path.join(base_dir, "wiki_train_sequence.npz")
    val_seq_path: str = os.path.join(base_dir, "wiki_val_sequence.npz")
    train_label_path: str = os.path.join(base_dir, "wiki_train_labels.npy")
    val_label_path: str = os.path.join(base_dir, "wiki_val_labels.npy")
    x_train_path: str = os.path.join(base_dir, "x_train.npy")
    x_val_path: str = os.path.join(base_dir, "x_val.npy")
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")
    feature_names_path: str = os.path.join(base_dir, "feature_names.json")
    default_n_event_code: int = PREDICTION_CONFIG.default_n_event_code


class WikiCombinedTransformation:
    def __init__(self):
        self.config = CombinedTransformConfig()

    def compute_time_deltas(self, df):
        if 'time_delta_seq' not in df.columns:
            tqdm.pandas(desc="Computing time deltas")
            df['time_delta_seq'] = df['rev_time'].progress_apply(
                lambda times: [(pd.to_datetime(times[i]) - pd.to_datetime(times[0])).total_seconds() / 60
                               for i in range(len(times))]
            )
        return df

    def impute_sequence(self, seq):
        return seq if len(seq) > 0 else [[-1]]

    def process_sequence_features(self, df, seq_path, label_path, n_event_code):
        try:
            os.makedirs(os.path.dirname(seq_path), exist_ok=True)
            os.makedirs(os.path.dirname(label_path), exist_ok=True)

            tqdm.pandas(desc="Processing sequences")
            event_seq = [self.impute_sequence(s) + [[n_event_code - 1]] for s in tqdm(df["edit_sequence"], desc="Event Seqs")]
            rev_time = [self.impute_sequence(s) + [0] for s in tqdm(df["time_delta_seq"], desc="Rev Times")]
            event_failure_sys = [[0] * len(self.impute_sequence(s)) + [1] for s in tqdm(df["edit_sequence"], desc="Sys Failures")]
            event_failure_user = [[0] * len(self.impute_sequence(s)) + [1] for s in tqdm(df["edit_sequence"], desc="User Failures")]

            labels = df["label"].values

            np.savez(seq_path,
                     event_seq=np.array(event_seq, dtype=object),
                     rev_time=np.array(rev_time, dtype=object),
                     event_failure_sys=np.array(event_failure_sys, dtype=object),
                     event_failure_user=np.array(event_failure_user, dtype=object))
            np.save(label_path, labels)

        except Exception as e:
            raise CustomException(f"Error while saving sequence data: {str(e)}", sys)

    def extract_tabular_features(self, df, label_required=True):
        feature_cols = [
            'total_edits', 'unique_pages', 'unique_categories', 'meta_edit_ratio',
            'sessions_count', 'avg_session_length', 'max_session_length',
            'night_edit_ratio', 'weekend_edit_ratio', 'fast_edit_ratio_3min',
            'fast_edit_ratio_15min', 'page_diversity_ratio', 'category_diversity_ratio',
            'first_edit_meta', 'meta_burst_score', 'category_switch_score',
            'reedit_score', 'time_regularity_score', 'session_intensity_variance',  'meta_burst_score', 'meta_edit_count',
            'avg_session_duration',
            'max_session_duration',
            'edit_span_hours',
            'edits_per_hour',
        ]

        available_features = [col for col in feature_cols if col in df.columns]
        if not available_features:
            raise CustomException("No usable features found in dataframe", sys)

        features = df[available_features]
        labels = df['label'].astype(int) if label_required and 'label' in df.columns else None
        return features, labels

    def get_preprocessor(self, feature_columns):
        try:
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, feature_columns)
            ])
            return preprocessor
        except Exception as e:
            logging.error("Error while creating preprocessing object")
            raise CustomException(e, sys)

    def transform(self, train_path, val_path):
        try:
            train_df = pd.read_pickle(train_path)
            val_df = pd.read_pickle(val_path)
            logging.info("Loaded combined wiki data with enhanced features")

            has_edit_sequence = 'edit_sequence' in train_df.columns and 'rev_time' in train_df.columns

            if has_edit_sequence:
                train_df = self.compute_time_deltas(train_df)
                val_df = self.compute_time_deltas(val_df)
                self.process_sequence_features(train_df, self.config.train_seq_path, self.config.train_label_path, self.config.default_n_event_code)
                self.process_sequence_features(val_df, self.config.val_seq_path, self.config.val_label_path, self.config.default_n_event_code)
                logging.info("Sequence features processed successfully")
            else:
                logging.info("Skipping sequence processing - enhanced features only")
                os.makedirs(os.path.dirname(self.config.train_seq_path), exist_ok=True)
                os.makedirs(os.path.dirname(self.config.val_seq_path), exist_ok=True)
                np.save(self.config.train_label_path, train_df["label"].values)
                np.save(self.config.val_label_path, val_df["label"].values)

            x_train, y_train = self.extract_tabular_features(train_df)
            x_val, y_val = self.extract_tabular_features(val_df)

            preprocessor = self.get_preprocessor(list(x_train.columns))
            x_train_scaled = preprocessor.fit_transform(x_train)
            x_val_scaled = preprocessor.transform(x_val)

            feature_index_map = {feature: idx for idx, feature in enumerate(x_train.columns)}
            with open(self.config.feature_names_path, "w") as f:
                json.dump(feature_index_map, f, indent=2)

            os.makedirs(self.config.base_dir, exist_ok=True)
            np.save(self.config.x_train_path, x_train_scaled)
            np.save(self.config.x_val_path, x_val_scaled)
            save_object(self.config.preprocessor_path, preprocessor)

            logging.info(f"Combined transformation complete with {x_train_scaled.shape[1]} features")
            return self.config

        except Exception as e:
            logging.error("Error during combined transformation")
            raise CustomException(e, sys)


if __name__ == "__main__":
    transformer = WikiCombinedTransformation()
    transformer.transform(
        train_path="data/processed/wiki/user_edits_train.pkl",
        val_path="data/processed/wiki/user_edits_val.pkl"
    )