import numpy as np
import tensorflow as tf
import time
from sklearn.metrics import average_precision_score, f1_score
from tqdm import tqdm

from src.utils.helpers import CombinedCustomLoss
from src.models.seqtab.units import pad_matrix, pad_time, pad_failure_bits

def batch_generator(data, batch_size, max_len, n_event_code, train_flags, num_batches, maxcode):
    x, y = data

    if train_flags.get('train_combined'):
        x_tabular, event_seq_code, time_step, event_failure_sys, event_failure_user = x
        y_tab = y_seq = y

        pos_idx = np.where(y_seq == 1)[0]
        neg_idx = np.where(y_seq == 0)[0]
        rng = np.random.default_rng()

        for _ in range(num_batches):
            if len(pos_idx) and len(neg_idx):
                pos_count = max(1, batch_size // 2)
                neg_count = batch_size - pos_count
                batch_indices = np.concatenate([
                    rng.choice(pos_idx, size=pos_count, replace=len(pos_idx) < pos_count),
                    rng.choice(neg_idx, size=neg_count, replace=len(neg_idx) < neg_count)
                ])
                rng.shuffle(batch_indices)
            else:
                start_idx = _ * batch_size
                end_idx = min(start_idx + batch_size, y_seq.shape[0])
                batch_indices = np.arange(start_idx, end_idx)

            event_seq_code_sliced = event_seq_code[batch_indices]
            time_step_sliced = time_step[batch_indices]
            event_seq_code_sliced, mask, mask_final, mask_code = pad_matrix(
                event_seq_code_sliced, max_len, n_event_code, maxcode, pad_token=n_event_code
            )
            time_step_sliced = pad_time(time_step_sliced, max_len=max_len)
            event_failure_sys_sliced = pad_failure_bits(event_failure_sys[batch_indices], max_len=max_len)
            event_failure_user_sliced = pad_failure_bits(event_failure_user[batch_indices], max_len=max_len)
            lengths = np.array([len(seq) for seq in event_seq_code_sliced])

            yield (
                x_tabular[batch_indices],
                tf.convert_to_tensor(event_seq_code_sliced, dtype=tf.float32),
                tf.convert_to_tensor(time_step_sliced, dtype=tf.float32),
                tf.convert_to_tensor(event_failure_sys_sliced, dtype=tf.float32),
                tf.convert_to_tensor(event_failure_user_sliced, dtype=tf.float32),
                tf.convert_to_tensor(mask, dtype=tf.float32),
                tf.convert_to_tensor(mask_final, dtype=tf.float32),
                tf.convert_to_tensor(mask_code, dtype=tf.float32),
                tf.convert_to_tensor(lengths, dtype=tf.float32),
            ), (
                tf.convert_to_tensor(y_tab[batch_indices].astype(float)),
                tf.convert_to_tensor(y_seq[batch_indices].astype(float))
            )

def train_model(model, train_data, val_data, epochs, train_flags, verbose=False, val_every_n_epochs=1):
    history = {key: [] for key in [
        'total_loss', 'tabular_loss', 'sequential_loss', 'accuracy',
        'f1_score', 'precision', 'recall', 'auc_pr', 'avg_precision', 'auc_roc',
        'val_total_loss', 'val_tabular_loss', 'val_sequential_loss',
        'val_accuracy', 'val_f1_score', 'val_precision', 'val_recall',
        'val_auc_pr', 'val_avg_precision', 'val_auc_roc']}

    loss_fn = CombinedCustomLoss()
    optimizer = model.optimizer
    best_val_f1 = 0
    patience = 3
    patience_counter = 0

    num_samples = train_data[0][0].shape[0] if train_flags.get('train_combined') else train_data[0].shape[0]
    num_batches = (num_samples + model.batch_size - 1) // model.batch_size

    if verbose:
        print(f"Starting training for {epochs} epochs with {num_samples} samples and batch size {model.batch_size} ({num_batches} batches per epoch)")

    for epoch in range(epochs):
        start_time = time.time()
        avg_precision = 0
        epoch_total_loss = 0
        epoch_tab_loss = 0
        epoch_seq_loss = 0
        y_true_epoch = []
        y_pred_epoch = []

        print(f"\nEpoch {epoch + 1}/{epochs}")
        for batch_idx, (inputs, labels) in enumerate(
            tqdm(batch_generator(train_data, model.batch_size, model.max_len, model.n_event_code, train_flags, num_batches, model.max_code),
                 total=num_batches, desc="Training Batches"), 1):

            with tf.GradientTape() as tape:
                tabular_input, *seq_inputs = inputs
                y_tab_true, y_seq_true = labels

                tab_repr = model.tabular_model(tabular_input, training=True)
                y_tab_pred = model.tabular_output(tab_repr)

                seq_repr = model.sequential_model(tuple(seq_inputs), training=True)
                y_seq_pred = model.sequential_output(seq_repr)

                # Use original inputs for combined prediction, not representations
                y_combined_pred = model(inputs, training=True, **train_flags)

                total_loss = loss_fn((y_tab_true, y_seq_true), (y_tab_pred, y_seq_pred, y_combined_pred))
                epoch_total_loss += total_loss.numpy()
                epoch_tab_loss += loss_fn.tabular_loss.numpy()
                epoch_seq_loss += loss_fn.sequence_loss.numpy()
                y_true_epoch.extend(y_seq_true.numpy().ravel())
                y_pred_epoch.extend(y_combined_pred.numpy().ravel())

            gradients = tape.gradient(total_loss, model.trainable_variables)
            gradients = tf.clip_by_global_norm(gradients, clip_norm=1.0)[0]
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            model.compiled_metrics.update_state(y_seq_true, y_combined_pred)
            avg_precision += average_precision_score(y_seq_true.numpy(), y_combined_pred.numpy())

        metrics_result = {m.name: m.result().numpy() for m in model.metrics}
        metrics_result['avg_precision'] = avg_precision / num_batches
        metrics_result['f1_score'] = f1_score(y_true_epoch, np.array(y_pred_epoch) > 0.5)
        metrics_result['total_loss'] = epoch_total_loss / num_batches
        metrics_result['tabular_loss'] = epoch_tab_loss / num_batches
        metrics_result['sequential_loss'] = epoch_seq_loss / num_batches

        print(" - ".join([f"{k}: {metrics_result[k]:.4f}" for k in metrics_result if k in history]))

        for key in metrics_result:
            history[key].append(metrics_result[key])

        for metric in model.metrics:
            metric.reset_states()

        # ----- Validation Pass -----
        if val_data and ((epoch + 1) % val_every_n_epochs == 0 or (epoch + 1 == epochs)):
            val_samples = val_data[0][0].shape[0] if train_flags.get('train_combined') else val_data[0].shape[0]
            val_batches = (val_samples + model.batch_size - 1) // model.batch_size
            val_avg_precision = 0
            val_total_loss = 0
            val_tab_loss = 0
            val_seq_loss = 0
            val_y_true = []
            val_y_pred = []

            for val_inputs, val_labels in tqdm(
                batch_generator(val_data, model.batch_size, model.max_len, model.n_event_code, train_flags, val_batches, model.max_code),
                total=val_batches, desc="Validation Batches"):

                tabular_input, *seq_inputs = val_inputs
                y_tab_true, y_seq_true = val_labels

                tab_repr = model.tabular_model(tabular_input, training=False)
                y_tab_pred = model.tabular_output(tab_repr)

                seq_repr = model.sequential_model(tuple(seq_inputs), training=False)
                y_seq_pred = model.sequential_output(seq_repr)

                # Use original inputs for combined prediction, not representations
                y_combined_pred = model(val_inputs, training=False, **train_flags)

                total_loss = loss_fn((y_tab_true, y_seq_true), (y_tab_pred, y_seq_pred, y_combined_pred))
                val_total_loss += total_loss.numpy()
                val_tab_loss += loss_fn.tabular_loss.numpy()
                val_seq_loss += loss_fn.sequence_loss.numpy()
                val_y_true.extend(y_seq_true.numpy().ravel())
                val_y_pred.extend(y_combined_pred.numpy().ravel())

                model.compiled_metrics.update_state(y_seq_true, y_combined_pred)
                val_avg_precision += average_precision_score(y_seq_true.numpy(), y_combined_pred.numpy())

            val_metrics = {m.name: m.result().numpy() for m in model.metrics}
            val_metrics['avg_precision'] = val_avg_precision / val_batches
            val_metrics['f1_score'] = f1_score(val_y_true, np.array(val_y_pred) > 0.5)
            val_metrics['total_loss'] = val_total_loss / val_batches
            val_metrics['tabular_loss'] = val_tab_loss / val_batches
            val_metrics['sequential_loss'] = val_seq_loss / val_batches

            for key, value in val_metrics.items():
                history[f'val_{key}'].append(value)

            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            for metric in model.metrics:
                metric.reset_states()

    return history
