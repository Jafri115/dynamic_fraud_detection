import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(directory='data/Datasets/publicDataset/wiki/vews_dataset_v1.1'):

    # Load all required files from the specified directory
    edit_files = [f for f in os.listdir(directory) if f.endswith('.csv') and (f.startswith('benign_') or f.startswith('vandal_'))]
    dfs = [pd.read_csv(os.path.join(directory, f)) for f in edit_files]
    edit_df = pd.concat(dfs, ignore_index=True)
    
    pages_df = pd.read_csv(os.path.join(directory, 'pages.csv'))
    users_df = pd.read_csv(os.path.join(directory, 'users.csv'), sep='\t', names=['username', 'userid', '1', '2', 'label'])
    
    return edit_df, pages_df, users_df

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
        .join(users_df.set_index('username'), on='username')[
            [
                'username',
                'revtime',
                'encoded_categories',
                'label',
                'isReverted',
                'pagetitle',
                'cluebotRevert',
                'stiki_score',
            ]
        ]
    )
    
    # Process columns for the ML model
    merged_data['isReverted'] = merged_data['isReverted'].astype(int)
    merged_data['label'] = merged_data['label'].map({'benign': 0, 'vandal': 1})
    merged_data['revtime'] = pd.to_datetime(merged_data['revtime'])

    grouped = merged_data.groupby(['username', 'label'])

    user_edits = grouped.agg(
        encoded_categories=('encoded_categories', list),
        revtime=('revtime', list),
        isReverted=('isReverted', list),
        pagetitle=('pagetitle', list),
        cluebotRevert=('cluebotRevert', list),
        stiki_score=('stiki_score', list),
    ).reset_index()

    user_edits['total_edits'] = user_edits['revtime'].apply(len)
    user_edits['unique_pages'] = user_edits['pagetitle'].apply(lambda x: len(set(x)))
    user_edits['avg_stiki_score'] = user_edits['stiki_score'].apply(np.mean)
    user_edits['cluebot_revert_count'] = user_edits['cluebotRevert'].apply(sum)

    def edit_freq(times):
        times = sorted(times)
        if len(times) <= 1:
            return 0.0
        diffs = [(t2 - t1).total_seconds() for t1, t2 in zip(times[:-1], times[1:])]
        return np.mean(diffs)

    user_edits['edit_frequency'] = user_edits['revtime'].apply(edit_freq)
    user_edits['night_edits'] = user_edits['revtime'].apply(
        lambda ts: sum(1 for t in ts if t.hour < 6 or t.hour >= 18)
    )
    user_edits['day_edits'] = user_edits['revtime'].apply(
        lambda ts: sum(1 for t in ts if 6 <= t.hour < 18)
    )
    user_edits['weekend_edits'] = user_edits['revtime'].apply(
        lambda ts: sum(1 for t in ts if t.weekday() >= 5)
    )
    user_edits['page_category_diversity'] = user_edits['encoded_categories'].apply(
        lambda seq: len({c for cats in seq for c in cats if c != -1})
    )
    user_edits['reverted_ratio'] = user_edits['isReverted'].apply(
        lambda x: sum(x) / len(x) if len(x) > 0 else 0
    )

    return user_edits

def save_train_test_data(user_edits, save_dir='data/Datasets/publicDataset/wiki/vews_dataset_v1.1'):

    train_df, test_df = train_test_split(
        user_edits, test_size=0.2, random_state=42, stratify=user_edits['label']
    )
    selected_cols = [
        'total_edits',
        'unique_pages',
        'avg_stiki_score',
        'cluebot_revert_count',
        'edit_frequency',
        'night_edits',
        'day_edits',
        'weekend_edits',
        'page_category_diversity',
        'reverted_ratio',
    ]
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
