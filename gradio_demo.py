import sys
import numpy as np
import tensorflow as tf
import gradio as gr


def load_trained_model(path: str):
    """Load a trained Keras model from the given path."""
    return tf.keras.models.load_model(path)


def get_model():
    """Return the loaded model based on the command-line argument."""
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "path_to_saved_model.h5"  # Edit this path if passing via CLI is not used
    return load_trained_model(model_path)


model = get_model()


def predict(
    total_edits,
    unique_pages,
    avg_stiki_score,
    cluebot_revert_count,
    edit_frequency,
    night_edits,
    day_edits,
    weekend_edits,
    page_category_diversity,
):
    """Return vandal/benign prediction and probability."""
    features = np.array(
        [
            [
                total_edits,
                unique_pages,
                avg_stiki_score,
                cluebot_revert_count,
                edit_frequency,
                night_edits,
                day_edits,
                weekend_edits,
                page_category_diversity,
            ]
        ],
        dtype=np.float32,
    )
    prob = float(model.predict(features)[0][0])
    label = "vandal" if prob >= 0.5 else "benign"
    return label, prob


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Total Edits"),
        gr.Number(label="Unique Pages"),
        gr.Number(label="Average STiki Score"),
        gr.Number(label="ClueBot Revert Count"),
        gr.Number(label="Edit Frequency"),
        gr.Number(label="Night Edits"),
        gr.Number(label="Day Edits"),
        gr.Number(label="Weekend Edits"),
        gr.Number(label="Page Category Diversity"),
    ],
    outputs=[gr.Textbox(label="Prediction"), gr.Textbox(label="Probability")],
    title="Wikipedia Vandalism Prediction",
    description="Provide tabular features from Wikipedia edits to classify a user as vandal or benign.",
)


if __name__ == "__main__":
    demo.launch()
