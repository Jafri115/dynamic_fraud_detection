
import warnings
import os
import json
import ast
import numpy as np
from optuna.integration import KerasPruningCallback
import tensorflow as tf
import mlflow
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.metrics import Precision, Recall
import matplotlib.pyplot as plt
from models.HITANET.units import adjust_input

tf.config.run_functions_eagerly(False)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import mlflow.tensorflow
import mlflow.pyfunc

warnings.filterwarnings("ignore")

def getBenignData(data, rep_layer_name='hidden_rep'):
    ben_data = data[data['label'] == 0]
    X_ben = np.vstack(ben_data[rep_layer_name].to_numpy())
    y_ben = np.vstack(ben_data['label'].to_numpy())
    return X_ben, y_ben

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

def print_box_with_header(header, extra_width=100):
    box_width = extra_width
    box_width = max(box_width, len(header) + 4)
    print("+" + "-" * (box_width - 2) + "+")
    header_padding = (box_width - len(header) - 2) // 2
    print("|" + " " * header_padding + header + " " * header_padding + (" " if len(header) % 2 != box_width % 2 else "") + "|")
    print("+" + "-" * (box_width - 2) + "+")

def transform_inputs(merged_df, n_event_code, max_len, selected_cols, train_flags, output_label_name):
    y_seq = merged_df['SEQ_' + output_label_name].to_numpy()
    y_tab = merged_df['TAB_' + output_label_name].to_numpy()

    if train_flags.get('train_combined') or train_flags.get('train_seq'):
        event_seq_code, time_step, event_failure_sys, event_failure_user = adjust_input(
            merged_df['EVENT_SEQUENCE'].to_numpy(), merged_df['TIME_DIFF_ORDER'].to_numpy(),
            merged_df['EVENT_FAILURE_SYSTEM_BIT'].to_numpy(), merged_df['EVENT_FAILURE_USER_BIT'].to_numpy(),
            max_len, n_event_code)

        if train_flags.get('train_combined'):
            x_tabular = merged_df[selected_cols].to_numpy()
            return (x_tabular, event_seq_code, time_step, event_failure_sys, event_failure_user), (y_tab, y_seq)
        elif train_flags.get('train_seq'):
            return (event_seq_code, time_step, event_failure_sys, event_failure_user), (y_seq)

    if train_flags.get('train_tab'):
        x_tabular = merged_df[selected_cols].to_numpy()
        return (x_tabular), (y_tab)

    raise ValueError("No valid training flag found in 'train_flags'.")

def transform_inputs_ae(merged_df, n_event_code, max_len, output_label_name):
    y_seq = merged_df['SEQ_' + output_label_name].to_numpy()
    event_seq_code, time_step, event_failure_sys, event_failure_user = adjust_input(
        merged_df['EVENT_SEQUENCE'].to_numpy(), merged_df['TIME_DIFF_ORDER'].to_numpy(),
        merged_df['EVENT_FAILURE_SYSTEM_BIT'].to_numpy(), merged_df['EVENT_FAILURE_USER_BIT'].to_numpy(),
        max_len, n_event_code)
    return (event_seq_code, time_step, event_failure_sys, event_failure_user), (y_seq)

def prepare_data_for_training(merged_train_df, output_label_name, resample_dict):
    y_seq = merged_train_df['SEQ_' + output_label_name].to_numpy()
    y_tab = merged_train_df['TAB_' + output_label_name].to_numpy()

    if resample_dict.get('rus'):
        resampler = RandomUnderSampler()
    elif resample_dict.get('smote'):
        resampler = SMOTE()
    elif resample_dict.get('ns'):
        return merged_train_df, (y_tab, y_seq)

    X_resampled, _ = resampler.fit_resample(merged_train_df, y_seq)
    return X_resampled, (X_resampled['TAB_' + output_label_name].astype(float).to_numpy(), X_resampled['SEQ_' + output_label_name].to_numpy())

def read_parameter_grid_for_dataset(filename, dataset_name):
    with open(filename, 'r') as file:
        all_grids = json.load(file)
        parameter_grid = all_grids.get(dataset_name)
        if parameter_grid is None:
            raise ValueError(f"No parameter grid found for dataset: {dataset_name}")
        return parameter_grid

def get_git_commit_hash() -> str:
    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
    except:
        return 'NA'
    return sha

def get_best_params_in_experiment(experiment_name, metric_name="f1_score", filter_string_ap=""):
    mlflow.set_experiment(experiment_name)
    if filter_string_ap == '':
        filter_string = f"status = 'FINISHED'"
    else:
        filter_string = f"status = 'FINISHED' and {filter_string_ap}"

    cols = ['tags.mlflow.runName',
            'tags.git_commit_hash',
            'tags.time_elapsed',
            'metrics.tp',
            'metrics.tn',
            'metrics.precision',
            'metrics.f1_score',
            'metrics.avg_prec',
            'metrics.fn',
            'metrics.auc_roc',
            'metrics.accuracy',
            'metrics.auc_pr',
            'metrics.recall',
            'metrics.fp',
            'params.tab_hidden_states',
            'params.epochs',
            'params.layer',
            'params.learning_rate',
            'params.train_label_1_percentage',
            'params.train_size',
            'params.num_of_tabular_columns',
            'params.dropout_rate_comb',
            'params.dropout_rate_seq',
            'params.droput_rate_tab',
            'params.train_flags_name',
            'params.resample_tech',
            'params.val_label_1_percentage',
            'params.combined_hidden_layers',
            'params.eval_size',
            'params.max_len',
            'params.weight_decay',
            'params.l2_lambda_comb',
            'params.l2_lambda_tab',
            'params.l2_lambda_seq',
            'params.model_dim']
    runs = mlflow.search_runs(mlflow.get_experiment_by_name(experiment_name).experiment_id, filter_string)

    best_params_exp = runs.loc[runs['metrics.' + metric_name].idxmax()][cols]
    best_params_exp = best_params_exp.to_dict()
    best_params_exp = {"ph1_" + key: value for key, value in best_params_exp.items()}

    best_param = {
        "combined_hidden_layers": best_params_exp['ph1_params.combined_hidden_layers'],
        "tab_hidden_states": best_params_exp['ph1_params.tab_hidden_states'],
        "max_len": best_params_exp['ph1_params.max_len'],
        "dropout_rate_comb": best_params_exp['ph1_params.dropout_rate_comb'],
        "dropout_rate_seq": best_params_exp['ph1_params.dropout_rate_seq'],
        "droput_rate_tab": best_params_exp['ph1_params.droput_rate_tab'],
        "l2_lambda_comb": best_params_exp['ph1_params.l2_lambda_comb'],
        "l2_lambda_tab": best_params_exp['ph1_params.l2_lambda_tab'],
        "l2_lambda_seq": best_params_exp['ph1_params.l2_lambda_seq'],
        "weight_decay": best_params_exp['ph1_params.weight_decay'],
        "learning_rate": best_params_exp['ph1_params.learning_rate'],
        "layer": best_params_exp['ph1_params.layer'],
        "num_of_tabular_columns": best_params_exp['ph1_params.num_of_tabular_columns'],
        "model_dim": best_params_exp['ph1_params.model_dim']
    }

    return convert_values(best_param)

def get_best_params_in_experiment_ae(experiment_name, metric_name="test_accuracy", filter_string_ap=""):
    mlflow.set_experiment(experiment_name)
    if filter_string_ap == '':
        filter_string = f"status = 'FINISHED'"
    else:
        filter_string = f"status = 'FINISHED' and {filter_string_ap}"

    cols = ['tags.mlflow.runName',
            'metrics.test_loss',
            'metrics.std_loss',
            'metrics.test_accuracy',
            'metrics.std_accuracy',
            'params.epochs',
            'params.layer',
            'params.learning_rate',
            'params.train_label_1_percentage',
            'params.train_size',
            'params.dropout_rate_seq',
            'params.val_label_1_percentage',
            'params.eval_size',
            'params.max_len',
            'params.l2_lambda_seq',
            'params.model_dim',
            'params.latent_dim']
    runs = mlflow.search_runs(mlflow.get_experiment_by_name(experiment_name).experiment_id, filter_string)

    best_params_exp = runs.loc[runs['metrics.' + metric_name].idxmax()][cols]
    best_params_exp = best_params_exp.to_dict()
    best_params_exp = {"ph1_" + key: value for key, value in best_params_exp.items()}

    best_param = {
        "max_len": best_params_exp['ph1_params.max_len'],
        "dropout_rate_seq": best_params_exp['ph1_params.dropout_rate_seq'],
        "l2_lambda_seq": best_params_exp['ph1_params.l2_lambda_seq'],
        "learning_rate": best_params_exp['ph1_params.learning_rate'],
        "layer": best_params_exp['ph1_params.layer'],
        "model_dim": best_params_exp['ph1_params.model_dim'],
        "latent_dim": best_params_exp['ph1_params.latent_dim']
    }

    return convert_values(best_param)

def dense_to_sparse3(dense_tensor, mask=None):
    if mask is not None:
        indices = tf.where(mask)
        values = tf.gather_nd(dense_tensor, indices)
    else:
        indices = tf.where(tf.ones_like(dense_tensor))
        values = tf.gather_nd(dense_tensor, indices)
    shape = tf.shape(dense_tensor, out_type=tf.int64)
    return tf.SparseTensor(indices, values, shape)

class EditDistanceMetricWithMask(tf.keras.metrics.Metric):
    def __init__(self, name='edit_distance', padded_value=None):
        super(EditDistanceMetricWithMask, self).__init__(name=name)
        self.total_distance = self.add_weight(name='total_distance', initializer='zeros')
        self.padded_value = padded_value
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), dtype=tf.int64)
        valid_mask = tf.not_equal(y_true, self.padded_value)
        y_true_sparse = dense_to_sparse3(tf.cast(y_true, dtype=tf.int64), mask=valid_mask)
        y_pred_sparse = dense_to_sparse3(y_pred, mask=valid_mask)
        distance = tf.edit_distance(y_pred_sparse, y_true_sparse, normalize=True)
        self.total_distance.assign_add(tf.reduce_sum(distance))
        self.count.assign_add(tf.cast(tf.size(distance), dtype=tf.float32))

    def result(self):
        return self.total_distance / self.count

    def reset_states(self):
        self.total_distance.assign(0)
        self.count.assign(0)

def get_run_by_runname_ae(experiment_name, run_name):
    mlflow.set_experiment(experiment_name)
    filter_string = f"tags.mlflow.runName = '{run_name}'"
    cols = ['tags.mlflow.runName',
            'tags.git_commit_hash',
            'tags.time_elapsed',
            'metrics.test_loss',
            'metrics.test_accuracy',
            'metrics.test_distance',
            'params.epochs',
            'params.layer',
            'params.learning_rate',
            'params.train_label_1_percentage',
            'params.train_size',
            'params.dropout_rate_seq',
            'params.val_label_1_percentage',
            'params.eval_size',
            'params.max_len',
            'params.model_dim',
            'params.l2_lambda_seq']
    runs = mlflow.search_runs(mlflow.get_experiment_by_name(experiment_name).experiment_id, filter_string)
    if not runs.empty:
        row = runs[cols].iloc[0]
        row_dict = row.to_dict()
        row_dict = {"ph1_" + key: value for key, value in row_dict.items()}
        return convert_values(row_dict)
    else:
        return None

def get_run_by_runname(experiment_name, run_name):
    mlflow.set_experiment(experiment_name)
    filter_string = f"tags.mlflow.runName = '{run_name}'"
    cols = ['tags.mlflow.runName',
            'tags.git_commit_hash',
            'tags.time_elapsed',
            'metrics.tp',
            'metrics.tn',
            'metrics.precision',
            'metrics.f1_score',
            'metrics.avg_prec',
            'metrics.fn',
            'metrics.auc_roc',
            'metrics.accuracy',
            'metrics.auc_pr',
            'metrics.recall',
            'metrics.fp',
            'params.tab_hidden_states',
            'params.epochs',
            'params.layer',
            'params.learning_rate',
            'params.train_label_1_percentage',
            'params.train_size',
            'params.num_of_tabular_columns',
            'params.dropout_rate_comb',
            'params.dropout_rate_seq',
            'params.droput_rate_tab',
            'params.train_flags_name',
            'params.resample_tech',
            'params.val_label_1_percentage',
            'params.combined_hidden_layers',
            'params.eval_size',
            'params.max_len',
            'params.weight_decay',
            'params.l2_lambda_comb',
            'params.l2_lambda_tab',
            'params.l2_lambda_seq']
    runs = mlflow.search_runs(mlflow.get_experiment_by_name(experiment_name).experiment_id, filter_string)
    if not runs.empty:
        row = runs[cols].iloc[0]
        row_dict = row.to_dict()
        row_dict = {"ph1_" + key: value for key, value in row_dict.items()}
        return convert_values(row_dict)
    else:
        return None

def convert_values(data):
    for key, value in data.items():
        if type(value) == float or type(value) == int:
            data[key] = (value)
        elif value.startswith('[') and value.endswith(']'):
            data[key] = ast.literal_eval(value)
        elif '.' in value or 'e' in value.lower():
            try:
                data[key] = float(value)
            except ValueError:
                print(f"Value {value} at key {key} could not be converted to float.")
        else:
            try:
                data[key] = int(value)
            except ValueError:
                print(f"Value {value} at key {key} could not be converted to int.")
    return data

def plot_combined_metrics_ae(history, experiment_name, run_name):
    metrics = ['loss', 'accuracy']
    for metric in metrics:
        val_metric = f'val_{metric}'
        train_metric = f'train_{metric}'
        if val_metric in history:
            plt.figure(figsize=(10, 6))
            plt.plot(history[train_metric], label=f'Train {metric}')
            plt.plot(history[val_metric], label=f'Validation {metric}')
            plt.title(f'{metric.capitalize()} over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
            basepath = 'plots/' + experiment_name
            if not os.path.exists(basepath):
                os.mkdir(basepath)
            dir_name = basepath + '/' + run_name
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            plt.savefig(dir_name + f'/{metric}_combined.png')
            plt.close()

def plot_metrics_phase2(history, experiment_name, run_name):
    metrics = ['G_loss', 'D_loss', 'roc_auc', 'F1score', 'avg_prec', 'accuracy', 'recall', 'precision']
    for metric in metrics:
        val_metric = f'{metric}_val'
        train_metric = f'{metric}_train'
        if val_metric in history:
            plt.figure(figsize=(10, 6))
            plt.plot(history[train_metric], label=f'Train {train_metric}')
            if metric != 'G_loss' or metric != 'D_loss':
                plt.plot(history[val_metric], label=f'Validation {val_metric}')
            plt.title(f'{metric.capitalize()} over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
            basepath = 'plots/' + experiment_name
            if not os.path.exists(basepath):
                os.mkdir(basepath)
            dir_name = basepath + '/' + run_name
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            plt.savefig(dir_name + f'/{metric}_combined.png')
            plt.close()

def evaluate_model(model, val_data, batch_size, train_flags):
    evaluation = model.evaluate_model(val_data, batch_size, train_flags)
    metrices = []
    for _, value in evaluation.items():
        metrices.append(value)
    return metrices

def evaluate_model_ae(model, val_data, batch_size):
    evaluation = model.evaluate_model(val_data, batch_size)
    metrices = []
    for _, value in evaluation.items():
        metrices.append(value)
    return metrices

class BlockingTimeSeriesSplit:
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)
        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class CombinedCustomLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(CombinedCustomLoss, self).__init__()
        self.tabular_loss = 0
        self.sequence_loss = 0
        self.bce_tab = tf.keras.losses.CategoricalCrossentropy()
        self.bce_tab = tf.keras.losses.CategoricalCrossentropy()
        self.bce_comb = tf.keras.losses.CategoricalCrossentropy()

    def call(self, y_true, y_pred):
        y_tab, y_seq = y_true

        y_tab_pred, y_seq_pred ,y_combined_pred = y_pred 
        # y_tab_pred = tf.squeeze(y_tab_pred, axis=-1) 
        # y_seq_pred = tf.squeeze(y_seq_pred, axis=-1)
        # y_combined_pred = tf.squeeze(y_combined_pred, axis=-1)

        self.tabular_loss = self.bce_tab(y_tab, y_tab_pred)
        self.sequence_loss = self.bce_tab(y_seq, y_seq_pred)

        loss_combined = self.bce_comb(y_seq, y_combined_pred)
        total_loss = loss_combined

        return total_loss


class CustomMaskedSparseCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, padded_value, name="custom_masked_sparse_categorical_accuracy"):
        super(CustomMaskedSparseCategoricalAccuracy, self).__init__(name=name)
        self.padded_value = padded_value
        self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    def update_state(self, y_true, y_pred, sample_weight=None):

        mask = tf.math.not_equal(y_true, self.padded_value)

        if sample_weight is None:
            sample_weight = tf.cast(mask, dtype=tf.float32)
        else:
            sample_weight = tf.cast(mask, dtype=tf.float32) * sample_weight

        self.accuracy.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return self.accuracy.result()

    def reset_states(self):
        self.accuracy.reset_states()





