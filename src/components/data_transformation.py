# src/components/data_transformation_combined.py

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils.io import save_object
from src.config.configuration import PREDICTION_CONFIG

@dataclass
class CombinedTransformConfig:
    base_dir: str = os.path.join("data", "processed", "combined")
    train_seq_path: str = os.path.join("data", "processed" , "combined" ,"wiki_train_sequence.npz")
    val_seq_path: str = os.path.join("data", "processed" , "combined" , "wiki_val_sequence.npz")
    train_label_path: str = os.path.join("data", "processed" , "combined" , "wiki_train_labels.npy")
    val_label_path: str = os.path.join("data", "processed" , "combined" , "wiki_val_labels.npy")
    x_train_path: str = os.path.join("data", "processed" , "combined" , "x_train.npy")
    x_val_path: str = os.path.join("data", "processed" , "combined" , "x_val.npy")
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")
    default_n_event_code: int = PREDICTION_CONFIG.default_n_event_code  # Import from configuration


class WikiCombinedTransformation:
    def __init__(self):
        self.config = CombinedTransformConfig()

    def compute_time_deltas(self, df):
        df['time_delta_seq'] = df['rev_time'].apply(
            lambda times: [(pd.to_datetime(times[i]) - pd.to_datetime(times[0])).total_seconds() / 60
                           for i in range(len(times))]
        )
        return df

    def impute_sequence(self, seq):
        return seq if len(seq) > 0 else [[-1]]

    def process_sequence_features(self, df, seq_path, label_path, n_event_code):
        try:
            # Ensure directories exist
            os.makedirs(os.path.dirname(seq_path), exist_ok=True)
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            
            event_seq = [self.impute_sequence(s) + [[n_event_code - 1]] for s in df["edit_sequence"]]
            rev_time = [self.impute_sequence(s) + [0] for s in df["time_delta_seq"]]
            # Use default values for system and user failure sequences
            event_failure_sys = [[0] * len(self.impute_sequence(s)) + [1] for s in df["edit_sequence"]]
            event_failure_user = [[0] * len(self.impute_sequence(s)) + [1] for s in df["edit_sequence"]]

            labels = df["label"].values

            np.savez(
                seq_path,
                event_seq=np.array(event_seq, dtype=object),
                rev_time=np.array(rev_time, dtype=object),
                event_failure_sys=np.array(event_failure_sys, dtype=object),
                event_failure_user=np.array(event_failure_user, dtype=object),
            )
            np.save(label_path, labels)

        except Exception as e:
            raise CustomException(f"Error while saving sequence data: {str(e)}", sys)



    def extract_tabular_features(self, df, label_required=True):
        df['num_edit_sessions'] = df['edit_sequence'].apply(len)

        df['avg_categories_per_edit'] = df['edit_sequence'].apply(
            lambda x: np.mean([len(c) if isinstance(c, list) else 0 for c in x])
        )
        

        # New features added below:
        df['unique_pages'] = df['unique_pages'] if 'unique_pages' in df.columns else df['edit_sequence'].apply(lambda x: len(set([item for sublist in x for item in sublist])))


        # Edit frequency (mean time delta between edits in seconds)
        df['edit_frequency'] = df['rev_time'].apply(
            lambda times: np.mean(pd.to_datetime(times).to_series().diff().dt.total_seconds()) 
            if len(times) > 1 else 0
        )

        # Night edits (18:00 to 06:00)
        df['night_edits'] = df['rev_time'].apply(
            lambda times: sum((pd.to_datetime(times).hour < 6) | (pd.to_datetime(times).hour >= 18))
        )

        # Day edits (06:00 to 18:00)
        df['day_edits'] = df['rev_time'].apply(
            lambda times: sum((pd.to_datetime(times).hour >= 6) & (pd.to_datetime(times).hour < 18))
        )

        # Weekend edits (Saturday and Sunday)
        df['weekend_edits'] = df['rev_time'].apply(
            lambda times: sum(pd.to_datetime(times).weekday >= 5)
        )

        # Category diversity (how many distinct page categories user edited)
        df['page_category_diversity'] = df['edit_sequence'].apply(
            lambda seq: len(set([cat for sublist in seq for cat in sublist if isinstance(cat, int)]))
        )

        # Final feature set
        features = df[[
            'total_edits',
            'num_edit_sessions',
            'avg_categories_per_edit',
            'unique_pages',
            'edit_frequency',
            'night_edits',
            'day_edits',
            'weekend_edits',
            'page_category_diversity',
        ]]

        if label_required and 'label' in df.columns:
            return features, df['label'].astype(int)

        return features, None


    def get_preprocessor(self):
        """Create preprocessing pipeline for tabular features."""
        try:
            numeric_features = [
                'total_edits',
                'num_edit_sessions',
                'avg_categories_per_edit',
                'unique_pages',
                'edit_frequency',
                'night_edits',
                'day_edits',
                'weekend_edits',
                'page_category_diversity',
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ]
            )

            preprocessor = ColumnTransformer(
                [('num_pipeline', num_pipeline, numeric_features)]
            )
            return preprocessor
        except Exception as e:
            logging.error("Error while creating preprocessing object")
            raise CustomException(e, sys)

    def transform(self, train_path, val_path):
        try:
            train_df = pd.read_pickle(train_path)
            val_df = pd.read_pickle(val_path)

            logging.info("Loaded combined wiki data")

            # --- Sequence ---
            train_df = self.compute_time_deltas(train_df)
            val_df = self.compute_time_deltas(val_df)
            self.process_sequence_features(train_df, self.config.train_seq_path, self.config.train_label_path, n_event_code=self.config.default_n_event_code)
            self.process_sequence_features(val_df, self.config.val_seq_path, self.config.val_label_path, n_event_code=self.config.default_n_event_code)

            # --- Tabular ---
            x_train, y_train = self.extract_tabular_features(train_df)
            x_val, y_val = self.extract_tabular_features(val_df)
            preprocessor = self.get_preprocessor()
            x_train_scaled = preprocessor.fit_transform(x_train)
            x_val_scaled = preprocessor.transform(x_val)

            os.makedirs(self.config.base_dir, exist_ok=True)

            np.save(self.config.x_train_path, x_train_scaled)
            np.save(self.config.x_val_path, x_val_scaled)

            save_object(self.config.preprocessor_path, preprocessor)

            logging.info("Combined transformation complete.")
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
