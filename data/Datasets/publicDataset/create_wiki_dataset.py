# create_wiki_dataset.py
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(directory='data/Datasets/publicDataset/wiki/vews_dataset_v1.1'):

    # Load all required files from the specified directory
    edit_files = [f for f in os.listdir(directory) if f.endswith('.csv') and (f.startswith('benign_') or f.startswith('vandal_'))]
    dfs = [pd.read_csv(os.path.join(directory, filename)) for filename in edit_files]
    edit_df = pd.concat(dfs, ignore_index=True)
    
    pages_df = pd.read_csv(os.path.join(directory, 'pages.csv'))
    users_df = pd.read_csv(os.path.join(directory, 'users.csv'), sep='\t', names=['username', 'userid', '1', '2', 'label'])
    
    return edit_df[:100], pages_df[:100], users_df[:100]

def process_data(edit_df, pages_df, users_df):

    # Process page categories
    pages_df['pagecategories_list'] = pages_df['pagecategories'].apply(lambda x: list(eval(x)))
    all_categories = {cat for cats in pages_df['pagecategories_list'] for cat in cats}
    category_dict = {cat: idx for idx, cat in enumerate(all_categories)}

    # Map categories
    pages_df['encoded_categories'] = pages_df['pagecategories_list'].apply(lambda cats: [category_dict.get(cat, -1) for cat in cats])

    # Merge datasets
    merged_data = (
        edit_df.join(pages_df.set_index('pagetitle'), on='pagetitle')
        .join(users_df.set_index('username'), on='username')[['username', 'revtime', 'encoded_categories', 'label', 'isReverted']]
    )
    
    # Process columns for the ML model
    merged_data['isReverted'] = merged_data['isReverted'].astype(int)
    merged_data['label'] = merged_data['label'].map({'benign': 0, 'vandal': 1})
    user_edits = (
        merged_data.groupby(['username', 'label'])
        .agg({'encoded_categories': list, 'revtime': list, 'isReverted': list})
        .reset_index()
    )
    user_edits['total_edits'] = merged_data.groupby(['username', 'label']).size().values

    return user_edits

def save_train_test_data(user_edits, save_dir='data/Datasets/publicDataset/wiki/vews_dataset_v1.1'):

    train_df, test_df = train_test_split(user_edits, test_size=0.2, random_state=42, stratify=user_edits['label'])
    selected_cols = ['reverted_ratio']  # Columns to normalize
    scaler = MinMaxScaler()

    train_df[selected_cols] = scaler.fit_transform(train_df[selected_cols])
    test_df[selected_cols] = scaler.transform(test_df[selected_cols])

    # Save to pickle files
    train_df.to_pickle(os.path.join(save_dir, 'user_edits_train.pkl'))
    test_df.to_pickle(os.path.join(save_dir, 'user_edits_test.pkl'))

# Execution
edit_df, pages_df, users_df = load_data()
user_edits = process_data(edit_df, pages_df, users_df)
save_train_test_data(user_edits)
