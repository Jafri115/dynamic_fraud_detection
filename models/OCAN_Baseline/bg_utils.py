# bg_utils.py
import sys
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KernelDensity
import matplotlib as plt
from tensorflow.keras.losses import CategoricalCrossentropy
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.offline import plot as plot_offline

def softmax_cross_entropy_with_logits_wrapper(labels, logits):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

def one_hot(x, depth):
    x_one_hot = np.zeros((len(x), depth), dtype=np.int32)
    x = x.astype(int)
    for i in range(x_one_hot.shape[0]):
        x_one_hot[i, x[i]] = 1
    return x_one_hot

def pull_away_loss(g):
    Nor = tf.norm(g, axis=1)
    Nor_mat = tf.tile(tf.expand_dims(Nor, axis=1), [1, tf.shape(g)[1]])
    X = tf.divide(g, Nor_mat)
    X_X = tf.square(tf.matmul(X, tf.transpose(X)))
    mask = tf.subtract(tf.ones_like(X_X), tf.linalg.diag(tf.ones([tf.shape(X_X)[0]])))
    pt_loss = tf.divide(tf.reduce_sum(tf.multiply(X_X, mask)),
                        tf.multiply(tf.cast(tf.shape(X_X)[0], tf.float32), tf.cast(tf.shape(X_X)[0]-1, tf.float32)))
    return pt_loss

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def gradient_penalty(discriminator, real_samples, fake_samples, labels):
    epsilon = tf.random.uniform([real_samples.shape[0], 1], 0.0, 1.0)
    real_samples = tf.cast(real_samples, tf.float32)
    fake_samples = tf.cast(fake_samples, tf.float32)
    epsilon = tf.cast(epsilon, tf.float32)
    interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        prob_interpolated, _, _ = discriminator(interpolated, training=True)
        grad_interpolated = tape.gradient(prob_interpolated, [interpolated])[0]
    epsilon = 1e-8  # Adding epsilon to avoid sqrt(0) leading to NaN
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad_interpolated), axis=1) + epsilon)
    gradient_penalty = tf.reduce_mean(tf.square(grad_norm - 1.0))

    return gradient_penalty

def discriminator_loss(D_prob_real,D_logit_real,D_prob_gen, D_logit_gen, y_real, y_fake):

    real_loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_real, logits=D_logit_real)) 
    fake_loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_fake, logits=D_logit_gen)) 
    ent_real_loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(D_prob_real, tf.math.log(tf.clip_by_value(D_prob_real, 1e-12, 1.0))), axis=1))

    return real_loss + fake_loss  + 1.85 * ent_real_loss

def discriminator_loss_reg(D_logit_real, D_logit_gen, y_real, y_fake):
    
    # Compute the loss for real and fake outputs
    real_loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_real, logits=D_logit_real)) 
    fake_loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_fake, logits=D_logit_gen)) 

    return real_loss + fake_loss  

def generator_loss_reg(D_logit_gen,y_fake):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_fake, logits=D_logit_gen)) 


def generator_loss(D_h2_tar_gen, D_logit_tar, D_prob_tar_gen, D_h2_real, D_h2_gen, y_tar,D_prob_gen, lambda_pt=0.5, lambda_ent=0.9, lambda_fm=1.5):
    # Pull-away loss
    pt_loss = pull_away_loss(D_h2_tar_gen)
    
    # Threshold calculation
    tar_thrld = (tf.reduce_max(D_prob_tar_gen[:, -1]) + tf.reduce_min(D_prob_tar_gen[:, -1])) / 2
    
    # Entropy loss calculation
    indicator = tf.sign(D_prob_tar_gen[:, -1] - tar_thrld)
    condition = tf.greater(tf.zeros_like(indicator), indicator)
    mask_tar = tf.where(condition, tf.zeros_like(indicator), indicator)
    epsilon = 1e-12
    G_ent_loss = tf.reduce_mean(tf.math.log(tf.maximum(D_prob_tar_gen[:, -1], epsilon)) * mask_tar) # for prop above threshold
    
    # Feature matching loss
    mean_feature_of_generated = tf.reduce_mean(D_h2_gen, axis=0) # shape should be (1, dim_h2)
    mean_feature_of_real = tf.reduce_mean(D_h2_real, axis=0)
    fm_loss = tf.reduce_mean(tf.square(mean_feature_of_generated - mean_feature_of_real))

    # Total generator loss
    G_loss = lambda_pt * pt_loss + lambda_ent * G_ent_loss + lambda_fm * fm_loss

    return G_loss ,pt_loss , G_ent_loss ,fm_loss


def check_nan(tensor, message):
    if tf.reduce_any(tf.math.is_nan(tensor)):
        tf.print(f"NaN detected in {message}", output_stream=sys.stderr)