import json
import os
from src.utils.visualization import plot_training_history


def main(history_path="training_history.json"):
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"{history_path} not found")
    with open(history_path, "r") as f:
        history = json.load(f)
    plot_training_history(history)


if __name__ == "__main__":
    main()
