# src/models/ocan/ocgan.py
from __future__ import annotations

import os
import pickle
import sys
from typing import Tuple, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def sample_Z(m: int, n: int) -> tf.Tensor:
    """Return a normal noise tensor."""
    return tf.random.normal(shape=(m, n))


def check_nan(value: tf.Tensor, name: str) -> None:
    if tf.reduce_any(tf.math.is_nan(value)):
        tf.print(f"NaN detected in {name}", output_stream=sys.stderr)


def gradient_penalty(discriminator: tf.keras.Model,
                      real: tf.Tensor,
                      fake: tf.Tensor,
                      labels: tf.Tensor) -> tf.Tensor:
    """WGAN-GP style gradient penalty."""
    alpha = tf.random.uniform(shape=[real.shape[0], 1], minval=0.0, maxval=1.0)
    interpolated = real * alpha + fake * (1 - alpha)
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        prob, _, _ = discriminator(interpolated, training=True)
    grad = tape.gradient(prob, interpolated)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=1))
    return tf.reduce_mean((slopes - 1.0) ** 2)


def discriminator_loss(prob_real: tf.Tensor,
                        logit_real: tf.Tensor,
                        prob_fake: tf.Tensor,
                        logit_fake: tf.Tensor,
                        labels: tf.Tensor,
                        fake_labels: tf.Tensor) -> tf.Tensor:
    """Cross entropy discriminator loss."""
    real_loss = tf.keras.losses.categorical_crossentropy(labels, prob_real)
    fake_loss = tf.keras.losses.categorical_crossentropy(fake_labels, prob_fake)
    return tf.reduce_mean(real_loss + fake_loss)


def generator_loss(h_tar_gen: tf.Tensor,
                   logit_tar: tf.Tensor,
                   prob_tar_gen: tf.Tensor,
                   h_real: tf.Tensor,
                   h_gen: tf.Tensor,
                   prob_gen: tf.Tensor,
                   labels: tf.Tensor,
                   lambda_pt: float = 0.0,
                   lambda_ent: float = 0.0,
                   lambda_fm: float = 0.0) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return total generator loss and its components."""
    adv_loss = tf.keras.losses.categorical_crossentropy(labels, prob_gen)
    adv_loss = tf.reduce_mean(adv_loss)

    pt_loss = tf.reduce_mean(tf.square(h_tar_gen))
    ent_loss = -tf.reduce_mean(tf.keras.losses.categorical_crossentropy(prob_tar_gen, prob_tar_gen))
    fm_loss = tf.reduce_mean(tf.square(tf.reduce_mean(h_real, axis=0) - tf.reduce_mean(h_gen, axis=0)))

    total = adv_loss + lambda_pt * pt_loss + lambda_ent * ent_loss + lambda_fm * fm_loss
    return total, pt_loss, ent_loss, fm_loss


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class Discriminator(tf.keras.Model):
    def __init__(self,
                 dims: Tuple[int, ...],
                 input_dim: int,
                 dropout_rates: Tuple[float, ...],
                 batch_norm: bool = True):
        super().__init__()
        assert len(dropout_rates) == len(dims)
        self.layers_list = []

        self.layers_list.append(
            Dense(dims[0], input_shape=(input_dim,), activation=None,
                  kernel_initializer=initializers.HeNormal(seed=42))
        )
        self.layers_list[-1].name = 'input_dense'
        self.layers_list.append(LeakyReLU(alpha=0.2, name='input_leaky_relu'))
        if batch_norm:
            self.layers_list.append(BatchNormalization(name='input_batch_norm'))
        self.layers_list.append(Dropout(dropout_rates[0], name='input_dropout'))

        for i, dim in enumerate(dims[1:], 1):
            dense = Dense(dim, activation=None,
                          kernel_initializer=initializers.GlorotNormal(seed=42),
                          name=f'hidden_dense_{i}')
            self.layers_list.append(dense)
            self.layers_list.append(LeakyReLU(alpha=0.2, name=f'hidden_leaky_relu_{i}'))
            if batch_norm:
                self.layers_list.append(BatchNormalization(name=f'hidden_batch_norm_{i}'))
            self.layers_list.append(Dropout(dropout_rates[i], name=f'hidden_dropout_{i}'))

        self.output_layer = Dense(2, kernel_initializer=initializers.GlorotNormal(seed=42), name='final_layer_dense')
        self.softmax = tf.nn.softmax
        self.target_layer_name = f'hidden_dense_{len(dims) - 1}'

    def call(self, x: tf.Tensor, training: bool = True):
        final_hidden = None
        for layer in self.layers_list:
            if isinstance(layer, (Dropout, BatchNormalization)):
                x = layer(x, training=training)
            else:
                x = layer(x)
            if layer.name == self.target_layer_name:
                final_hidden = x
        logits = self.output_layer(x)
        prob = self.softmax(logits)
        return prob, logits, final_hidden


class DiscriminatorTarget(Discriminator):
    pass


class Generator(tf.keras.Model):
    def __init__(self,
                 dims: Tuple[int, ...],
                 input_dim: int,
                 dropout_rates: Tuple[float, ...],
                 batch_norm: bool = True):
        super().__init__()
        assert len(dropout_rates) == len(dims)
        self.layers_list = []
        for i, dim in enumerate(dims):
            self.layers_list.append(
                Dense(dim, activation='relu', kernel_initializer=initializers.HeNormal(seed=42))
            )
            if batch_norm:
                self.layers_list.append(BatchNormalization())
            self.layers_list.append(Dropout(dropout_rates[i]))
        self.layers_list.append(
            Dense(input_dim, activation='tanh', kernel_initializer=initializers.GlorotNormal(seed=42))
        )

    def call(self, z: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = z
        for layer in self.layers_list:
            if isinstance(layer, Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
                tf.debugging.check_numerics(x, 'generator_output')
        return x


class GANModel(tf.keras.Model):
    """OCAN training wrapper."""

    def __init__(self, params: Dict, total_samples: int, checkpoint_path: str | None = None):
        super().__init__()
        self.Z_dim = params['G_D_layers'][0][0]
        self.total_samples = total_samples
        self.mb_size = params['mb_size']
        self.generator = Generator(params['G_D_layers'][0], params['dim_inp'], params['g_dropouts'], batch_norm=params['batch_norm_g'])
        self.discriminator = Discriminator(params['G_D_layers'][1], params['dim_inp'], params['d_dropouts'], batch_norm=params['batch_norm_d'])
        self.discriminator_tar = DiscriminatorTarget(params['G_D_layers'][1], params['dim_inp'], params['d_dropouts'], batch_norm=params['batch_norm_d'])
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=params['g_lr'], beta_1=params['beta1_g'], beta_2=params['beta2_g'])
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=params['d_lr'], beta_1=params['beta1_d'], beta_2=params['beta2_d'])
        self.lambda_pt = params['lambda_pt']
        self.lambda_ent = params['lambda_ent']
        self.lambda_fm = params['lambda_fm']
        self.lambda_gp = params['lambda_gp']

        if checkpoint_path:
            self.ckpt_manager = tf.train.CheckpointManager(
                tf.train.Checkpoint(
                    generator_optimizer=self.generator_optimizer,
                    discriminator_optimizer=self.discriminator_optimizer,
                    generator=self.generator,
                    discriminator=self.discriminator,
                ),
                checkpoint_path,
                max_to_keep=3,
            )
        else:
            self.ckpt_manager = None

    def call(self, inputs: tf.Tensor, training: bool = False):
        prob, logits, features = self.discriminator(inputs, training=False)
        return prob, logits, features

    def train_step(self, real_samples: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        y_fake = tf.one_hot(tf.ones(labels.shape[0], dtype=tf.int32), depth=2)
        with tf.GradientTape() as disc_tape:
            generated = self.generator(sample_Z(labels.shape[0], self.Z_dim), training=False)
            prob_real, logit_real, _ = self.discriminator(real_samples, training=True)
            prob_fake, logit_fake, _ = self.discriminator(generated, training=True)
            disc_loss = discriminator_loss(prob_real, logit_real, prob_fake, logit_fake, labels, y_fake)
            gp = gradient_penalty(self.discriminator, tf.cast(real_samples, tf.float32), tf.cast(generated, tf.float32), labels)
            disc_loss += self.lambda_gp * gp
        grads_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(grads_disc, self.discriminator.trainable_variables))

        with tf.GradientTape() as gen_tape:
            generated = self.generator(sample_Z(labels.shape[0], self.Z_dim), training=True)
            _, logit_real, h_real = self.discriminator(real_samples, training=False)
            prob_gen, logit_gen, h_gen = self.discriminator(generated, training=False)
            _, logit_tar, _ = self.discriminator_tar(real_samples, training=False)
            prob_tar_gen, _, h_tar_gen = self.discriminator_tar(generated, training=False)
            gen_loss, pt_loss, ent_loss, fm_loss = generator_loss(
                h_tar_gen, logit_tar, prob_tar_gen, h_real, h_gen, prob_gen, labels,
                lambda_pt=self.lambda_pt, lambda_ent=self.lambda_ent, lambda_fm=self.lambda_fm)
        grads_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grads_gen, self.generator.trainable_variables))
        return disc_loss, gen_loss, pt_loss, ent_loss, fm_loss

    def fit(self, train_ds: tf.data.Dataset, val_data: Tuple[np.ndarray, np.ndarray], epochs: int = 100, initial_epoch: int = 0):
        history = {
            'D_loss_train': [],
            'G_loss_train': [],
            'pt_loss': [],
            'G_ent_loss': [],
            'fm_loss': [],
            'F1score_val': [],
        }
        x_val, y_val = val_data
        for epoch in range(initial_epoch, epochs):
            d_losses, g_losses, pt_losses, ent_losses, fm_losses = [], [], [], [], []
            for x_batch, y_batch in train_ds:
                d_loss, g_loss, pt_l, ent_l, fm_l = self.train_step(x_batch, y_batch)
                d_losses.append(d_loss)
                g_losses.append(g_loss)
                pt_losses.append(pt_l)
                ent_losses.append(ent_l)
                fm_losses.append(fm_l)
            history['D_loss_train'].append(np.mean(d_losses))
            history['G_loss_train'].append(np.mean(g_losses))
            history['pt_loss'].append(np.mean(pt_losses))
            history['G_ent_loss'].append(np.mean(ent_losses))
            history['fm_loss'].append(np.mean(fm_losses))
            eval_res = self.evaluate_model(x_val, y_val)
            history['F1score_val'].append(eval_res['F1_score'])
            print(
                f"Epoch {epoch + 1} - D_loss: {history['D_loss_train'][-1]:.4f} "
                f"G_loss: {history['G_loss_train'][-1]:.4f} F1: {eval_res['F1_score']:.4f}"
            )
        return history

    def evaluate_model(self, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        probs, _, _ = self(x_test, training=False)
        if np.isnan(probs).any():
            probs = np.nan_to_num(probs, nan=0.5)
        predictions = tf.argmax(probs, axis=1).numpy()
        labels = tf.argmax(y_test, axis=1).numpy()
        precision_metric = tf.keras.metrics.Precision()
        recall_metric = tf.keras.metrics.Recall()
        precision_metric.update_state(labels, predictions)
        recall_metric.update_state(labels, predictions)
        precision_v = precision_metric.result().numpy()
        recall_v = recall_metric.result().numpy()
        f1_v = 2 * precision_v * recall_v / (precision_v + recall_v + 1e-8)
        auc_metric = tf.keras.metrics.AUC()
        auc_metric.update_state(labels, probs[:, 1])
        auc_v = auc_metric.result().numpy()
        ap_metric = tf.keras.metrics.AUC(curve='PR')
        ap_metric.update_state(labels, probs[:, 1])
        ap_v = ap_metric.result().numpy()
        acc_v = np.mean(labels == predictions)
        return {
            'roc_auc_score': auc_v,
            'F1_score': f1_v,
            'average_precision_score': ap_v,
            'accuracy': acc_v,
            'recall': recall_v,
            'precision': precision_v,
        }

    def save_prob(self, x_test: np.ndarray, y_test: np.ndarray, run_name: str, dataset_name: str) -> None:
        probs, _, _ = self(x_test, training=False)
        df = {
            'features': list(x_test),
            'probs': probs[:, 1],
            'labels': list(tf.argmax(y_test, axis=1).numpy())
        }
        base_path = os.path.join('data', 'processed_data', dataset_name, 'OCAN_results')
        os.makedirs(os.path.join(base_path, run_name), exist_ok=True)
        file_path = os.path.join(base_path, run_name, f"combined_test_df_{x_test.shape[0]}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(df, f)