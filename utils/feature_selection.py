from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd

import numpy as np

def forward_selection_random_forest_f1(X, y, n_features_to_select=50):
    """
    Perform forward feature selection using RandomForest based on F1 score.

    Parameters:
    - X: pandas DataFrame of predictor variables.
    - y: pandas Series or array-like of the target variable.
    - n_features_to_select: int, the number of features to select.

    Returns:
    - selected_features: list of features selected by forward selection.
    """
    all_features = list(X.columns)
    selected_features = []
    while len(selected_features) < n_features_to_select:
        scores_with_candidates = []
        
        for feature in all_features:
            if feature not in selected_features:
                print(f"Trying feature: {feature}")
                # Try adding the feature and fitting the model
                temp_features = selected_features + [feature]
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                
                # Calculate F1 score using cross-validation
                score = np.mean(cross_val_score(model, X[temp_features], y, cv=3, scoring='f1'))
                
                scores_with_candidates.append((score, feature))

                
        # Find the feature whose addition improved the score the most
        scores_with_candidates.sort(reverse=True)  # Sort to get the highest score first
        best_score, best_feature = scores_with_candidates[0]
        
        selected_features.append(best_feature)
        print(f"Selected features: {selected_features}")
        all_features.remove(best_feature)

    return selected_features


# Define the model architecture


train_tabular_df = pd.read_pickle('data/processed_data/tabular_data/_tab__342__20240218-133320_tabular_data_train.pkl')
train_tabular_df = train_tabular_df
test_tabular_df = pd.read_pickle('data/processed_data/tabular_data/_tab__342__20240218-133320_tabular_data_test.pkl')
train_tabular_df = train_tabular_df
X_train = train_tabular_df.drop(['ORDER_ID','ID','CUSTOMER_ID'], axis=1)
y_train = train_tabular_df['ORDER_ID'].astype(float)

X_test = test_tabular_df.drop(['ORDER_ID','ID','CUSTOMER_ID'], axis=1)
y_test = test_tabular_df['ORDER_ID'].astype(float)
# Example usage

n_features_to_select = 80  # Adjust based on your preference

X_train_selected_features = forward_selection_random_forest_f1(X_train, y_train, n_features_to_select)
#