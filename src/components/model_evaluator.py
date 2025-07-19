from src.pipeline.predict_pipeline import Phase1PredictPipeline
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
import traceback

# Load test data
df_test = pd.read_pickle("data/processed/wiki/user_edits_test.pkl")
df_test["rev_time"] = df_test["rev_time"].apply(lambda x: pd.to_datetime(x))
y_true = df_test["label"].astype(int).values

# Initialize pipeline
pipeline = Phase1PredictPipeline()

# Predict in batches to avoid GPU OOM
batch_size = 64
y_probs = []
y_preds = []

print(f"Starting evaluation on {len(df_test)} samples...")

for i in range(0, len(df_test), batch_size):
    try:
        batch_df = df_test.iloc[i:i+batch_size].copy()
        results = pipeline.predict(batch_df)
        batch_probs = results["raw_probabilities"]
        y_probs.extend(batch_probs)
        y_preds.extend([1 if p >= 0.5 else 0 for p in batch_probs])
        
        # Print progress
        if (i // batch_size) % 10 == 0:
            print(f"Processed {i + len(batch_df)} samples...")
    except Exception as e:
        print(f"Error processing batch {i}-{i+batch_size}: {e}")
        traceback.print_exc()
        break

# Evaluation
if y_probs:
    print("=== Evaluation Metrics ===")
    print(f"Accuracy: {accuracy_score(y_true[:len(y_probs)], y_preds):.4f}")
    print(f"F1 Score: {f1_score(y_true[:len(y_probs)], y_preds):.4f}")
    print(f"ROC AUC:  {roc_auc_score(y_true[:len(y_probs)], y_probs):.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_true[:len(y_probs)], y_preds, target_names=["Legit", "Fraud"]))
else:
    print("No predictions were made due to errors.")
