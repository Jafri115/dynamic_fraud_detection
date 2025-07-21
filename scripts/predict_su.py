import pandas as pd
from src.pipeline.predict_pipeline import Phase1PredictPipeline

# Sample user edit data with tabular + sequence info
sample_data = {
    "edit_sequence": [
        [
        [260874, 83025],
        [90047],
        [258110]
        ]
    ],
    "rev_time": [
        [
        "2025-07-20T13:15:00.000Z",
        "2025-07-20T13:18:00.000Z",
        "2025-07-20T13:18:00.000Z"
        ]
    ],
    "total_edits": [3],
    "unique_pages": [2],
    "unique_categories": [1],
    "meta_edit_ratio": [0],
    "sessions_count": [1],
    "avg_session_length": [3],
    "max_session_length": [3],
    "night_edit_ratio": [0.0],
    "weekend_edit_ratio": [0.0],
    "fast_edit_ratio_3min": [0.66],
    "fast_edit_ratio_15min": [1.0],
    "page_diversity_ratio": [0.5],
    "category_diversity_ratio": [0.5],
    "first_edit_meta": [0],
    "meta_burst_score": [0.0],
    "category_switch_score": [0.0],
    "reedit_score": [0.0],
    "time_regularity_score": [0.5],
    "session_intensity_variance": [0.0]
}

# Create DataFrame
df = pd.DataFrame(sample_data)

# Convert rev_time strings to datetime
df["rev_time"] = df["rev_time"].apply(lambda x: pd.to_datetime(x))

# Load prediction pipeline
pipeline = Phase1PredictPipeline()

# Get prediction
results = pipeline.predict(df)

# Display results
print("Prediction Results:")
for i, (prob_pct, label) in enumerate(zip(results['probability_percentages'], results['labels'])):
    print(f"Sample {i+1}: {prob_pct:.2f}% fraud probability - Predicted: {label}")

print(f"\nRaw probabilities: {results['raw_probabilities']}")
