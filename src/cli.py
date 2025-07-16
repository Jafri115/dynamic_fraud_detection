"""Simple CLI entry points for running training phases."""
import argparse

from src.training.phase1 import train_representation_learning
from src.training.phase2 import train_ocan_model


def main():
    parser = argparse.ArgumentParser(description="SeqTab-OCAN training CLI")
    parser.add_argument("phase", choices=["phase1", "phase2"], help="Training phase to run")
    args = parser.parse_args()

    if args.phase == "phase1":
        print("Running Phase 1 training...")
        # Placeholder: call the underlying training function with defaults
        # The user should adapt parameters as needed
        train_representation_learning  # noqa: B018
    else:
        print("Running Phase 2 training...")
        train_ocan_model  # noqa: B018


if __name__ == "__main__":
    main()
