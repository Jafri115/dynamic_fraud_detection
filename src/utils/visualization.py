from typing import Dict, List
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


def plot_training_history(history: Dict[str, List[float]], out_dir: str = "plots") -> None:
    """Plot training history using Plotly and save as PNG or HTML.

    Parameters
    ----------
    history : Dict[str, List[float]]
        Dictionary with keys like "loss", "val_loss", etc.
    out_dir : str, optional
        Output directory for saving plots, default is "plots".
    export_html : bool
        Whether to export plots as interactive HTML instead of static PNG.
    """
    os.makedirs(out_dir, exist_ok=True)

    num_epochs = len(next((v for v in history.values() if v), []))
    epochs = list(range(1, num_epochs + 1))

    # Transform to long format
    data = []
    for metric, values in history.items():
        if not values:
            continue
        label = "val" if metric.startswith("val_") else "train"
        base_metric = metric.replace("val_", "")
        for epoch, value in zip(epochs, values):
            data.append({
                "Epoch": epoch,
                "Value": value,
                "Metric": base_metric,
                "Set": label
            })

    df = pd.DataFrame(data)

    # Plot each metric using Plotly
    for metric_name in df["Metric"].unique():
        df_metric = df[df["Metric"] == metric_name]

        fig = go.Figure()

        for set_name in df_metric["Set"].unique():
            subset = df_metric[df_metric["Set"] == set_name]
            fig.add_trace(go.Scatter(
                x=subset["Epoch"],
                y=subset["Value"],
                mode="lines+markers",
                name=set_name.capitalize()
            ))

        fig.update_layout(
            title=metric_name.title(),
            xaxis_title="Epoch",
            yaxis_title=metric_name,
            template="plotly_white",
            width=700,
            height=400
        )

        filename = os.path.join(out_dir, f"{metric_name}")

        fig.write_image(f"{filename}.png", scale=2)  # Needs kaleido installed


# --------------------------
# [TEST] Dummy Test Case
# --------------------------
if __name__ == "__main__":
    dummy_history = {
        "loss": [1.0, 0.8, 0.6, 0.5, 0.4],
        "val_loss": [1.1, 0.85, 0.65, 0.55, 0.5],
        "accuracy": [0.6, 0.7, 0.75, 0.8, 0.85],
        "val_accuracy": [0.58, 0.68, 0.72, 0.78, 0.8],
    }

    plot_training_history(dummy_history)
