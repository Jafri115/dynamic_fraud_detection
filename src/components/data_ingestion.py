import os
import sys
import re
import pickle
from dataclasses import dataclass
from typing import Dict, List
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.exception import CustomException
from src.logger import logging


@dataclass
class WikiDataIngestionConfig:
    raw_data_dir: str = os.path.join("data", "raw")
    processed_dir: str = os.path.join("data", "processed", "wiki")
    train_data_path: str = os.path.join(processed_dir, "user_edits_train.pkl")
    val_data_path: str = os.path.join(processed_dir, "user_edits_val.pkl")
    test_data_path: str = os.path.join(processed_dir, "user_edits_test.pkl")
    category_dict_path: str = os.path.join(processed_dir, "category_dict.pkl")

    vews_dir: str = os.path.join("data", "processed", "vews")
    user_logs_path: str = os.path.join(vews_dir, "user_logs.pkl")
    edit_pairs_path: str = os.path.join(vews_dir, "edit_pairs.pkl")

    page_type_dict_path: str = os.path.join(processed_dir, "page_type_dict.pkl")
    vews_features_metadata_path: str = os.path.join(processed_dir, "vews_features_metadata.pkl")


class WikiDataIngestion:
    def __init__(self):
        self.config = WikiDataIngestionConfig()

    def classify_page_type(self, title: str) -> str:
        if pd.isna(title): return 'unknown'
        meta_prefixes = [
            'Talk:', 'User:', 'User_talk:', 'Wikipedia:', 'Wikipedia_talk:',
            'File:', 'File_talk:', 'Template:', 'Template_talk:',
            'Category:', 'Category_talk:', 'Portal:', 'Portal_talk:',
            'Help:', 'Help_talk:', 'MediaWiki:', 'MediaWiki_talk:', 'Special:'
        ]
        return 'meta' if any(title.startswith(p) for p in meta_prefixes) else 'normal'

    def extract_vews_behavioral_features(self, user_data: pd.DataFrame) -> Dict:
        timestamps = pd.to_datetime(user_data['rev_time'], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce')

        edit_seq = user_data['edit_sequence']
        page_types = [self.classify_page_type(t) for t in user_data.get('page_title', [])]

        total_edits = len(edit_seq)
        unique_pages = len(set(user_data.get('page_title', [])))
        all_categories = [cat for sub in edit_seq for cat in sub]
        unique_categories = len(set(all_categories))

        meta_count = sum(pt == 'meta' for pt in page_types)
        meta_ratio = meta_count / max(total_edits, 1)

        sessions, curr_session = [], [0]
        for i in range(1, len(timestamps)):
            if (timestamps[i] - timestamps[i - 1]).total_seconds() / 60 <= 30:
                curr_session.append(i)
            else:
                sessions.append(curr_session)
                curr_session = [i]
        sessions.append(curr_session)

        session_lengths = [len(s) for s in sessions]
        durations = [
            (timestamps[s[-1]] - timestamps[s[0]]).total_seconds() / 60 if len(s) > 1 else 0
            for s in sessions
        ]

        time_diffs = [
            (timestamps[i] - timestamps[i - 1]).total_seconds()
            for i in range(1, len(timestamps))
        ]

        feature_dict = {
            'total_edits': total_edits,
            'unique_pages': unique_pages,
            'unique_categories': unique_categories,
            'meta_edit_count': meta_count,
            'meta_edit_ratio': meta_ratio,
            'sessions_count': len(sessions),
            'avg_session_length': np.mean(session_lengths),
            'max_session_length': max(session_lengths, default=0),
            'avg_session_duration': np.mean(durations),
            'max_session_duration': max(durations, default=0),
            'night_edit_ratio': sum(1 for ts in timestamps if ts.hour < 6 or ts.hour >= 22) / max(total_edits, 1),
            'weekend_edit_ratio': sum(1 for ts in timestamps if ts.weekday() >= 5) / max(total_edits, 1),
            'fast_edit_ratio_3min': sum(1 for d in time_diffs if d < 180) / max(len(time_diffs), 1),
            'fast_edit_ratio_15min': sum(1 for d in time_diffs if d < 900) / max(len(time_diffs), 1),
            'page_diversity_ratio': unique_pages / max(total_edits, 1),
            'category_diversity_ratio': unique_categories / max(total_edits, 1),
            'edit_span_hours': (timestamps.max() - timestamps.min()).total_seconds() / 3600 if len(timestamps) > 1 else 0,
            'edits_per_hour': total_edits / max((timestamps.max() - timestamps.min()).total_seconds() / 3600, 1),
            'first_edit_meta': int(page_types[0] == 'meta') if page_types else 0
        }

        return feature_dict

    def load_raw_data(self):
        files = [f for f in os.listdir(self.config.raw_data_dir) if f.endswith(".csv") and f.startswith(('benign_', 'vandal_'))]
        edit_df = pd.concat([pd.read_csv(os.path.join(self.config.raw_data_dir, f)) for f in files], ignore_index=True)
        pages_df = pd.read_csv(os.path.join(self.config.raw_data_dir, "pages.csv"))
        users_df = pd.read_csv(os.path.join(self.config.raw_data_dir, "users.csv"), sep="\t", names=["username", "userid", "1", "2", "label"])
        return edit_df, pages_df, users_df

    def save_category_dict(self, pages_df):
        pages_df['pagecategories_list'] = pages_df['pagecategories'].apply(lambda x: list(eval(x)))
        all_categories = {cat.replace('Category:', '') for cats in pages_df['pagecategories_list'] for cat in cats}
        category_dict = {cat: idx for idx, cat in enumerate(sorted(all_categories))}
        category_dict['no_cat'] = len(category_dict)

        os.makedirs(self.config.processed_dir, exist_ok=True)
        with open(self.config.category_dict_path, "wb") as f:
            pickle.dump(category_dict, f)
        return category_dict

    def process_user_edits(self, edit_df, pages_df, users_df, category_dict):
        pages_df['pagecategories_list'] = pages_df['pagecategories'].apply(lambda x: list(eval(x)))

        def map_to_ids(cats):
            return [category_dict.get(cat.replace('Category:', ''), category_dict['no_cat']) for cat in cats] or [category_dict['no_cat']]

        pages_df['edit_sequence'] = pages_df['pagecategories_list'].apply(map_to_ids)
        pages_df['page_type'] = pages_df['pagetitle'].apply(self.classify_page_type)

        data = (edit_df
                .join(pages_df.set_index("pagetitle"), on="pagetitle")
                .join(users_df.set_index("username"), on="username")
                .reset_index(drop=True))

        data["isReverted"] = data["isReverted"].astype(int)
        data["label"] = data["label"].map({"benign": 0, "vandal": 1})
        data.sort_values(by=["username", "revtime"], inplace=True)

        user_data = []
        grouped = data.groupby(["username", "label"])
        for (username, label), group in tqdm(grouped, total=len(grouped), desc="Extracting VEWS features"):
            user_df = pd.DataFrame({
                'edit_sequence': group["edit_sequence"].tolist(),
                'rev_time': group["revtime"].tolist(),
                'page_title': group["pagetitle"].tolist()
            })

            features = self.extract_vews_behavioral_features(user_df)

            # Add raw lists with original names (no 'raw_' prefix)
            features.update({
                'user_name': username,
                'label': label,
                'edit_sequence': group["edit_sequence"].tolist(),
                'rev_time': group["revtime"].tolist(),
                'page_title': group["pagetitle"].tolist()
            })

            user_data.append(features)

        return pd.DataFrame(user_data)



    def save_processed_data(self, df):
        os.makedirs(self.config.processed_dir, exist_ok=True)
        os.makedirs(self.config.vews_dir, exist_ok=True)

        train_val, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
        train, val = train_test_split(train_val, test_size=0.25, random_state=42, stratify=train_val["label"])

        train.to_pickle(self.config.train_data_path)
        val.to_pickle(self.config.val_data_path)
        test.to_pickle(self.config.test_data_path)

        feature_cols = [col for col in df.columns if col not in ['user_name', 'label']]
        metadata = {'feature_names': feature_cols, 'n_features': len(feature_cols)}

        with open(self.config.vews_features_metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        return self.config.train_data_path, self.config.val_data_path, self.config.test_data_path

    def initiate_data_ingestion(self):
        """
        Entry point for VEWS-enhanced data ingestion. If processed files exist, reuse them.
        Otherwise, process and save enhanced data.
        """
        try:
            logging.info("Starting enhanced data ingestion process.")
            edit_df, pages_df, users_df = self.load_raw_data()
            logging.info("Loaded raw data")
            category_dict = self.save_category_dict(pages_df)
            logging.info("Saved category dictionary")
            logging.info("Starting user-level feature extraction")
            user_edits = self.process_user_edits(edit_df, pages_df, users_df, category_dict)
            return self.save_processed_data(user_edits)

        except Exception as e:
            logging.exception("Error during data ingestion")
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestion = WikiDataIngestion()
    paths = ingestion.initiate_data_ingestion()
    logging.info(f"Ingestion complete. Train: {paths[0]}, Validation: {paths[1]}, Test: {paths[2]}")
