# scripts/train_phase2.py
import os
import sys
import json
import numpy as np
import tensorflow as tf


gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:  
        print(f"Unable to set memory growth for {gpu}: {e}")


from src.components.data_transformation import CombinedTransformConfig
from src.models.seqtab.combined import CombinedModel
from src.training.ocan_trainer import train_phase2


os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")


def load_data(cfg: CombinedTransformConfig):
    x_tab_train = np.load(cfg.x_train_path)
    x_tab_val = np.load(cfg.x_val_path)

    train_seq = np.load(cfg.train_seq_path, allow_pickle=True)
    val_seq = np.load(cfg.val_seq_path, allow_pickle=True)

    train_inputs = (
        x_tab_train,
        train_seq["event_seq"],
        train_seq["rev_time"],
        train_seq["event_failure_sys"],
        train_seq["event_failure_user"],
    )
    val_inputs = (
        x_tab_val,
        val_seq["event_seq"],
        val_seq["rev_time"],
        val_seq["event_failure_sys"],
        val_seq["event_failure_user"],
    )
    y_train = np.load(cfg.train_label_path)
    y_val = np.load(cfg.val_label_path)

    return (train_inputs, y_train), (val_inputs, y_val)


def main():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:  # pragma: no cover - best effort
            print(f"Unable to set memory growth for {gpu}: {e}")

    cfg = CombinedTransformConfig()
    (train_data, y_train), (val_data, y_val) = load_data(cfg)

    combined_model = tf.keras.models.load_model(os.path.join("saved_models", "combined_phase1"), compile=False)

    params = {
        "G_D_layers": ([32, 64], [64, 32]),
        "mb_size": 16,
        "g_dropouts": (0.1, 0.1),
        "d_dropouts": (0.1, 0.1),
        "batch_norm_g": True,
        "batch_norm_d": True,
        "g_lr": 1e-3,
        "d_lr": 1e-3,
        "beta1_g": 0.5,
        "beta2_g": 0.9,
        "beta1_d": 0.5,
        "beta2_d": 0.9,
        "lambda_pt": 0.1,
        "lambda_ent": 0.1,
        "lambda_fm": 0.1,
        "lambda_gp": 10.0,
    }

    gan_model, history = train_phase2(
        combined_model,
        (train_data, y_train),
        (val_data, y_val),
        params,
        epochs=5,
        batch_size=16,
    )

    os.makedirs("saved_models", exist_ok=True)
    gan_model.generator.save(os.path.join("saved_models", "ocan_generator"))
    gan_model.discriminator.save(os.path.join("saved_models", "ocan_discriminator"))

    with open("phase2_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
