# REFACTORED: src/components/data_ingestion.py
import os
import sys
import pandas as pd
import pickle
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
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

class WikiDataIngestion:
    def __init__(self):
        self.config = WikiDataIngestionConfig()

    def load_raw_data(self):
        logging.info("Loading raw wiki data from CSV files")
        edit_files = [
            f for f in os.listdir(self.config.raw_data_dir)
            if f.endswith(".csv") and (f.startswith("benign_") or f.startswith("vandal_"))
        ]
        dfs = [pd.read_csv(os.path.join(self.config.raw_data_dir, f)) for f in edit_files]
        edit_df = pd.concat(dfs, ignore_index=True)

        pages_df = pd.read_csv(os.path.join(self.config.raw_data_dir, "pages.csv"))
        users_df = pd.read_csv(
            os.path.join(self.config.raw_data_dir, "users.csv"),
            sep="\t",
            names=["username", "userid", "1", "2", "label"]
        )

        return edit_df, pages_df, users_df

    def save_category_dict(self, pages_df):
        logging.info("Generating and saving category dictionary")
        pages_df['pagecategories_list'] = pages_df['pagecategories'].apply(lambda x: list(eval(x)))
        all_categories = {cat for cats in pages_df['pagecategories_list'] for cat in cats}
        
        # Remove 'Category:' prefix from category names
        category_dict = {}
        for idx, cat in enumerate(all_categories):
            clean_cat = cat.replace('Category:', '') if cat.startswith('Category:') else cat
            category_dict[clean_cat] = idx
        
        category_dict['no_cat'] = len(category_dict)

        os.makedirs(self.config.processed_dir, exist_ok=True)
        with open(self.config.category_dict_path, "wb") as f:
            pickle.dump(category_dict, f)

        logging.info(f"category_dict.pkl saved at: {self.config.category_dict_path}")
        return category_dict

    def process_user_edits(self, edit_df, pages_df, users_df, category_dict):
        logging.info("Structuring wiki data for user sequence tracking")

        pages_df['pagecategories_list'] = pages_df['pagecategories'].apply(lambda x: list(eval(x)))
        
        # Function to clean category names and map to IDs
        def map_categories_to_ids(cats):
            result = []
            for cat in cats:
                clean_cat = cat.replace('Category:', '') if cat.startswith('Category:') else cat
                cat_id = category_dict.get(clean_cat, category_dict['no_cat'])
                result.append(cat_id)
            return result if result else [category_dict['no_cat']]
        
        pages_df['edit_sequence'] = pages_df['pagecategories_list'].apply(map_categories_to_ids)

        merged_data = (
            edit_df.join(pages_df.set_index("pagetitle"), on="pagetitle")
                .join(users_df.set_index("username"), on="username")
                .reset_index(drop=True)
        )

        merged_data["isReverted"] = merged_data["isReverted"].astype(int)
        merged_data["label"] = merged_data["label"].map({"benign": 0, "vandal": 1})
        merged_data.sort_values(by=["username", "revtime"], inplace=True)

        user_edits = (
            merged_data.groupby(["username", "label"])
                    .agg({
                        "edit_sequence": list,
                        "revtime": list,
                    })
                    .reset_index()
                    .rename(columns={
                        "username": "user_name",
                        "revtime": "rev_time",
                    })
        )


        user_edits["total_edits"] = merged_data.groupby(["username", "label"]).size().values

        return user_edits

    def save_processed_data(self, user_edits):
        logging.info("Splitting and saving train/val/test wiki data")

        os.makedirs(self.config.processed_dir, exist_ok=True)

        train_val_df, test_df = train_test_split(
            user_edits, test_size=0.2, random_state=42, stratify=user_edits["label"]
        )

        train_df, val_df = train_test_split(
            train_val_df, test_size=0.25, random_state=42, stratify=train_val_df["label"]
        )  # 0.25 x 0.8 = 0.2

        train_df.to_pickle(self.config.train_data_path)
        val_df.to_pickle(self.config.val_data_path)
        test_df.to_pickle(self.config.test_data_path)

        logging.info(f"Train data saved at: {self.config.train_data_path}")
        logging.info(f"Validation data saved at: {self.config.val_data_path}")
        logging.info(f"Test data saved at: {self.config.test_data_path}")

        return self.config.train_data_path, self.config.val_data_path, self.config.test_data_path

    def initiate_data_ingestion(self):
        try:
            edit_df, pages_df, users_df = self.load_raw_data()
            category_dict = self.save_category_dict(pages_df)
            user_edits = self.process_user_edits(edit_df, pages_df, users_df, category_dict)
            return self.save_processed_data(user_edits)

        except Exception as e:
            logging.exception(f"Exception occurred in WikiDataIngestion: {str(e)}")
            raise CustomException(e, sys)

# Optional usage
if __name__ == "__main__":
    ingestion = WikiDataIngestion()
    train_path, val_path, test_path = ingestion.initiate_data_ingestion()
    logging.info(f"Ingestion complete. Train: {train_path}, Validation: {val_path}, Test: {test_path}")
