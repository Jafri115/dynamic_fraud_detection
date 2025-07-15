# oc_gan_1.py
from datetime import datetime
import os
import pickle
import sys
import time

import pandas as pd
from models.OCAN_Baseline.bg_utils import check_nan, discriminator_loss, generator_loss, gradient_penalty,  sample_Z
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score,precision_score, recall_score ,average_precision_score,accuracy_score
from sklearn.metrics import classification_report, precision_recall_fscore_support

from tensorflow.python.keras import layers, initializers
from tensorflow.python.keras.layers import LeakyReLU,Dropout
from tensorflow.python.keras.initializers import GlorotNormal,HeNormal
from keras.layers import BatchNormalization, Dense,LeakyReLU

class Discriminator(tf.keras.Model):
    def __init__(self, D_dims, input_dim, dropout_rates, batchNorm=True):
        super(Discriminator, self).__init__()
        self.D_dims = D_dims
        self.layers_list = []
        self.batchNorm = batchNorm
        self.dropout_rates = dropout_rates
        assert len(dropout_rates) == len(D_dims), "Length of dropout_rates must match D_dims"

        # Input layer
        input_layer = Dense(D_dims[0], input_shape=(input_dim,), activation=None, kernel_initializer=initializers.HeNormal(seed=42), name='input_dense')
        self.layers_list.append(input_layer)
        self.layers_list.append(LeakyReLU(alpha=0.2, name='input_leaky_relu'))
        if batchNorm:
            self.layers_list.append(BatchNormalization(name='input_batch_norm'))
        self.layers_list.append(Dropout(dropout_rates[0], name='input_dropout'))

        # Hidden layers
        for i, dim in enumerate(D_dims[1:], 1):
            dense_layer = Dense(dim, activation=None, kernel_initializer=initializers.GlorotNormal(seed=42), name=f'hidden_dense_{i}')
            self.layers_list.append(dense_layer)
            self.layers_list.append(LeakyReLU(alpha=0.2, name=f'hidden_leaky_relu_{i}'))
            if batchNorm:
                self.layers_list.append(BatchNormalization(name=f'hidden_batch_norm_{i}'))
            self.layers_list.append(Dropout(dropout_rates[i], name=f'hidden_dropout_{i}'))

        # Output layer
        self.output_layer = Dense(2, kernel_initializer=initializers.GlorotNormal(seed=42), name='final_layer_dense')
        self.softmax = tf.nn.softmax

    def call(self, x, training=True):
        final_hidden_output = None  
        target_layer_name = f'hidden_dense_{len(self.D_dims) - 1}'

        for layer in self.layers_list:
            x = layer(x, training=training) if isinstance(layer, (Dropout, BatchNormalization)) else layer(x)
            if layer.name == target_layer_name:
                final_hidden_output = x  

        logits = self.output_layer(x)
        prob = self.softmax(logits)
        
        return prob, logits, final_hidden_output
    


class Discriminator_tar(tf.keras.Model):
    def __init__(self, D_dims, input_dim, dropout_rates, batchNorm=True):
        super(Discriminator_tar, self).__init__()
        self.D_dims = D_dims
        self.layers_list = []
        self.batchNorm = batchNorm
        self.dropout_rates = dropout_rates
        assert len(dropout_rates) == len(D_dims), "Length of dropout_rates must match D_dims"

        # Input layer
        input_layer = Dense(D_dims[0], input_shape=(input_dim,), activation=None, kernel_initializer=initializers.HeNormal(seed=42), name='input_dense')
        self.layers_list.append(input_layer)
        self.layers_list.append(LeakyReLU(alpha=0.2, name='input_leaky_relu'))
        if batchNorm:
            self.layers_list.append(BatchNormalization(name='input_batch_norm'))
        self.layers_list.append(Dropout(dropout_rates[0], name='input_dropout'))

        # Hidden layers
        for i, dim in enumerate(D_dims[1:], 1):
            dense_layer = Dense(dim, activation=None, kernel_initializer=initializers.GlorotNormal(seed=42), name=f'hidden_dense_{i}')
            self.layers_list.append(dense_layer)
            self.layers_list.append(LeakyReLU(alpha=0.2, name=f'hidden_leaky_relu_{i}'))
            if batchNorm:
                self.layers_list.append(BatchNormalization(name=f'hidden_batch_norm_{i}'))
            self.layers_list.append(Dropout(dropout_rates[i], name=f'hidden_dropout_{i}'))

        # Output layer
        self.output_layer = Dense(2, kernel_initializer=initializers.GlorotNormal(seed=42), name='final_layer_dense')
        self.softmax = tf.nn.softmax

    def call(self, x, training=True):
        final_hidden_output = None  # Variable to store the output of the last hidden dense layer
        target_layer_name = f'hidden_dense_{len(self.D_dims) - 1}'

        for layer in self.layers_list:
            x = layer(x, training=training) if isinstance(layer, (Dropout, BatchNormalization)) else layer(x)
            if layer.name == target_layer_name:
                final_hidden_output = x  # Capture the output of the last hidden dense layer

        logits = self.output_layer(x)
        prob = self.softmax(logits)
        
        return prob, logits, final_hidden_output
    
    def train_pretrained_discriminator(self, real_data, labels, optimizer):
        with tf.GradientTape() as tape:
            _ , logits, _ = self(real_data, training=True)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        gradients = tape.gradient(loss, self.trainable_variables)
        # optimizer.build(self.trainable_variables)
        optimizer.apply_gradients(zip(gradients,self.trainable_variables))
        return loss
    
    def compute_metrics(self, labels, logits):
        predictions = tf.argmax(logits, axis=1)
        true_classes = tf.argmax(labels, axis=1)
        f1 = f1_score(true_classes.numpy(), predictions.numpy(), average='macro')
        precision = precision_score(true_classes.numpy(), predictions.numpy(), average='macro')
        recall = recall_score(true_classes.numpy(), predictions.numpy(), average='macro')
        return f1, precision, recall

    def validate(self, validation_data):
        val_loss = 0
        all_f1_scores = []
        all_precisions = []
        all_recalls = []

        for x_val, y_val in validation_data:
            _, logits, _ = self(x_val, training=False)
            val_loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_val, logits=logits))
            f1, precision, recall = self.compute_metrics(y_val, logits)
            all_f1_scores.append(f1)
            all_precisions.append(precision)
            all_recalls.append(recall)

        avg_f1 = np.mean(all_f1_scores)
        avg_precision = np.mean(all_precisions)
        avg_recall = np.mean(all_recalls)
        return val_loss / len(validation_data), avg_f1, avg_precision, avg_recall

    
    def fit_pretrained_disc(self, train_data, validation_data, epochs=100, optimizer=None, patience=10):
        best_val_loss = float('inf')
        best_f1 = 0
        best_epoch = 0
        wait = 0  # Counter for how many epochs to wait after last improvement in validation loss
        best_weights = None  # Store the best model weights
        history = {
            "train_loss": [],
            "train_f1": [],
            "train_precision": [],
            "train_recall": [],
            "val_loss": [],
            "val_f1": [],
            "val_precision": [],
            "val_recall": []
        }

        for epoch in range(epochs):
            start_time = time.time()
            losses = []
            f1_scores = []
            precisions = []
            recalls = []

            for x_batch, y_batch in train_data:
                _, logits, _ = self(x_batch, training=True)
                loss = self.train_pretrained_discriminator(x_batch, y_batch, optimizer)
                losses.append(loss.numpy())
                f1, precision, recall = self.compute_metrics(y_batch, logits)
                f1_scores.append(f1)
                precisions.append(precision)
                recalls.append(recall)

            # Calculate average metrics for the epoch
            avg_train_loss = np.mean(losses)
            avg_f1 = np.mean(f1_scores)
            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)

            # Evaluate on validation data
            val_loss, val_f1, val_precision, val_recall = self.validate(validation_data)

            # Store metrics in history
            history['train_loss'].append(avg_train_loss)
            history['train_f1'].append(avg_f1)
            history['train_precision'].append(avg_precision)
            history['train_recall'].append(avg_recall)
            history['val_loss'].append(val_loss.numpy())
            history['val_f1'].append(val_f1)
            history['val_precision'].append(val_precision)
            history['val_recall'].append(val_recall)

            # Log the metrics for this epoch
            print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Train F1: {avg_f1:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")
            print(f"Validation Loss: {val_loss.numpy():.4f}, Val F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, Time Taken: {(time.time() - start_time)/60:.2f} min")

            # Early stopping logic based on validation F1 score
            if val_loss == 0 and avg_train_loss == 0:
                print(f"Stopping early. Validation F1 score reached 1.0 at epoch {epoch + 1}")
                break
            # elif val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     best_f1 = val_f1
            #     best_epoch = epoch
            #     wait = 0
            #     best_weights = self.get_weights()  # Save the best model weights
            # else:
            #     wait += 1
            #     if wait >= patience:
            #         print(f"Stopping early due to no improvement in validation loss for {patience} epochs.")
            #         break

        # Restore the model weights with the best observed validation loss
        if best_weights:
            self.set_weights(best_weights)
            print(f"Restored model weights from the end of the best epoch: {best_epoch + 1}")

        return history


# Generator definition
class Generator(tf.keras.Model):
    def __init__(self, G_dims, input_dim, dropout_rates,batchNorm=True):
        super(Generator, self).__init__()
        self.layers_list = []
        self.batchNorm = batchNorm

        # Ensure dropout rates are provided for each layer
        assert len(dropout_rates) == len(G_dims), "Length of dropout_rates must match G_dims"

        # Initialize layers
        for i, dim in enumerate(G_dims):
            self.layers_list.append(Dense(dim, activation='relu', kernel_initializer=HeNormal(seed=42)))
            if batchNorm:
                self.layers_list.append(BatchNormalization())
            self.layers_list.append(Dropout(dropout_rates[i]))

        # Output layer
        self.layers_list.append(Dense(input_dim, activation='tanh', kernel_initializer=GlorotNormal(seed=42)))

    def call(self, z, training=True):
        for layer in self.layers_list:
            if isinstance(layer, Dropout):
                z = layer(z, training=training)
            else:
                z = layer(z)
                tf.debugging.check_numerics(z, 'Check numerics: z')
        return z

class GANModel(tf.keras.Model):
    def __init__(self,param_dict,total_samples,checkpoint_path= None):
        super(GANModel, self).__init__()
        self.Z_dim = param_dict['G_D_layers'][0][0]
        self.total_samples = total_samples
        self.mb_size = param_dict['mb_size']
        self.G_dropout = param_dict['g_dropouts']
        self.D_dropout = param_dict['d_dropouts']
        self.generator  = Generator( G_dims = param_dict['G_D_layers'][0],input_dim=param_dict['dim_inp'],dropout_rates=param_dict['g_dropouts'],batchNorm=param_dict['batch_norm_g'])
        self.discriminator  = Discriminator( D_dims = param_dict['G_D_layers'][1],input_dim=param_dict['dim_inp'],dropout_rates=param_dict['d_dropouts'],batchNorm=param_dict['batch_norm_d'])
        self.discriminator_tar  = Discriminator_tar(D_dims=param_dict['G_D_layers'][1], input_dim=param_dict['dim_inp'],dropout_rates=param_dict['d_dropouts'],batchNorm=param_dict['batch_norm_d'])
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=param_dict['g_lr'], beta_1=param_dict['beta1_g'], beta_2=param_dict['beta2_g'])
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=param_dict['d_lr'], beta_1=param_dict['beta1_d'], beta_2=param_dict['beta2_d'])
        self.discriminator_tar_optimizer = tf.keras.optimizers.Adam(learning_rate=param_dict['d_lr'], beta_1=param_dict['beta1_d'], beta_2=param_dict['beta2_d'])
        self.lambda_pt = param_dict['lambda_pt']
        self.lambda_ent = param_dict['lambda_ent']
        self.lambda_fm = param_dict['lambda_fm']
        self.lambda_gp = param_dict['lambda_gp']
        print('pt_loss lambda:',self.lambda_pt,'G_ent_loss lambda:',self.lambda_ent,'fm_loss lambda:',self.lambda_fm,'gp_loss lambda:',self.lambda_gp)

        # Setup checkpointing
        if checkpoint_path is not None:
            self.checkpoint_dir = checkpoint_path
            self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
            self.checkpoint = tf.train.Checkpoint(
                generator_optimizer=self.generator_optimizer,
                discriminator_optimizer=self.discriminator_optimizer,
                generator=self.generator,
                discriminator=self.discriminator
            )
            self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)
        else:
            self.checkpoint = None
            self.ckpt_manager = None

    
    def call(self, inputs, training=False):

        if training:
            pass
        else:
            D_prob_real, D_logit_real, D_features_real = self.discriminator(inputs, training=False)
            return D_prob_real, D_logit_real, D_features_real
        
    def train_step(self, real_samples, labels):
        y_fake_mb = tf.one_hot(tf.ones(labels.shape[0], dtype=tf.int32), depth=2)
        # Update the discriminator
        with tf.GradientTape() as disc_tape:
            generated_samples = self.generator(sample_Z(labels.shape[0], self.Z_dim ), training=False)
            D_prob_real, D_logit_real, _ = self.discriminator(real_samples, training=True)
            D_prob_gen, D_logit_gen, _ = self.discriminator(generated_samples, training=True)

            disc_loss = discriminator_loss(D_prob_real,D_logit_real,D_prob_gen, D_logit_gen, labels, y_fake_mb)
            real_samples = tf.cast(real_samples, tf.float32)
            generated_samples = tf.cast(generated_samples, tf.float32)

            gp = gradient_penalty(self.discriminator, real_samples, generated_samples, labels)

            check_nan(gp, "gradient penalty")
            check_nan(disc_loss, "discriminator loss")
            disc_loss += self.lambda_gp * gp
                    # Check for NaN in loss
            if tf.reduce_any(tf.math.is_nan(disc_loss)):
                tf.print("NaN detected in loss", output_stream=sys.stderr)
                disc_loss = tf.where(tf.math.is_nan(disc_loss), tf.zeros_like(disc_loss), disc_loss)

        gradients_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_disc, self.discriminator.trainable_variables))

        # Update the generator
        with tf.GradientTape() as gen_tape:
            generated_samples = self.generator(sample_Z(labels.shape[0], self.Z_dim ), training=True)
            
            D_prob_real, D_logit_real, D_h2_real = self.discriminator(real_samples, training=False)
            D_prob_gen, D_logit_gen, D_h2_gen = self.discriminator(generated_samples, training=False)
            
            _, D_logit_tar, _ = self.discriminator_tar(real_samples, training=False)
            D_prob_tar_gen, _, D_h2_tar_gen  = self.discriminator_tar(generated_samples, training=False)

            gen_loss,pt_loss , G_ent_loss ,fm_loss = generator_loss(D_h2_tar_gen, D_logit_tar, D_prob_tar_gen, D_h2_real, D_h2_gen,D_prob_gen, labels,lambda_pt = self.lambda_pt,lambda_ent=  self.lambda_pt,lambda_fm=  self.lambda_pt)

        gradients_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        # gradients_gen = [tf.clip_by_value(g, -0.01, 0.01) for g in gradients_gen]
        self.generator_optimizer.apply_gradients(zip(gradients_gen, self.generator.trainable_variables))
        return disc_loss, gen_loss,pt_loss , G_ent_loss ,fm_loss


    def fit(self, train_data, test_data, epochs=100,initial_epoch=0,patience= 10):

        history = {
            'D_loss_train': [],
            'G_loss_train': [],
            'pt_loss':[],
            'G_ent_loss':[],
            'fm_loss':[],
            'roc_auc_val': [],
            'F1score_val': [],
            'avg_prec_val': [],
            'accuracy_val': [],
            'recall_val': [],
            'precision_val': [],
            'roc_auc_train': [],
            'F1score_train': [],
            'avg_prec_train': [],
            'accuracy_train': [],
            'recall_train': [],
            'precision_train': [],
        }
        best_f1_score = 0
        best_epoch = 0
        wait = 0
        time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_name = os.path.join('plots/TNSE_plots', time_now)
        # Create directory if it doesn't exist
        # if not os.path.exists(folder_name):
        #     os.makedirs(folder_name)
        # print(f"Saving plots to {folder_name}")
        for epoch in range(initial_epoch, epochs):
            batch_index = 0
            real_samples = None
            epoch_D_losses = []
            epoch_G_losses = []
            epoch_pt_losses = []
            epoch_G_ent_losses = []
            epoch_fm_losses = []
            for x_batch, y_batch in train_data.shuffle(1024):
                epoch_D_loss, epoch_G_loss,pt_loss , G_ent_loss ,fm_loss = self.train_step(x_batch, y_batch)  # Train model
                batch_index += 1
                real_samples = x_batch
                epoch_D_losses.append(epoch_D_loss)
                epoch_G_losses.append(epoch_G_loss)
                epoch_pt_losses.append(pt_loss)
                epoch_G_ent_losses.append(G_ent_loss)
                epoch_fm_losses.append(fm_loss)

            epoch_D_loss = np.mean(epoch_D_losses)
            epoch_G_loss = np.mean(epoch_G_losses)
            pt_loss = np.mean(epoch_pt_losses)
            G_ent_loss = np.mean(epoch_G_ent_losses)
            fm_loss = np.mean(epoch_fm_losses)
            evaluation_result_val = self.evaluate_model(*test_data)


            if evaluation_result_val['F1_score'] > best_f1_score and  self.checkpoint is not None:
                best_f1_score = evaluation_result_val['F1_score']
                best_epoch = epoch
                # Save the model
                path = self.ckpt_manager.save()
                print(f"Model improved and saved to {path} at epoch {epoch + 1}, F1 Score: {best_f1_score:.4f}")
            else:
                wait += 1
                if epoch >= 50 and evaluation_result_val['F1_score'] < 0.5:
                    print(f"Stopping early. Best validation f1 score was {best_f1_score:.4f} at epoch {best_epoch + 1}")
                    break

            print(f"Epoch {epoch + 1} -Discriminator Loss: {epoch_D_loss}, Generator Loss: {epoch_G_loss},pt_loss : {pt_loss} ,g_ent_loss : {G_ent_loss} ,fm_loss : {fm_loss}  F1 Score:{evaluation_result_val['F1_score']:.4f}, Precision: {evaluation_result_val['precision']:.4f}, Recall: {evaluation_result_val['recall']:.4f}, AUC: {evaluation_result_val['roc_auc_score']:.4f}, Avg Precision: {evaluation_result_val['average_precision_score']:.4f}, Accuracy: {evaluation_result_val['accuracy']:.4f}")

            # Update history dict

            history['D_loss_train'].append(epoch_D_loss)
            history['G_loss_train'].append(epoch_G_loss)
            history['pt_loss'].append(pt_loss)
            history['G_ent_loss'].append(G_ent_loss)
            history['fm_loss'].append(fm_loss)
            history['roc_auc_val'].append(evaluation_result_val['roc_auc_score'])
            history['F1score_val'].append(evaluation_result_val['F1_score'])
            history['avg_prec_val'].append(evaluation_result_val['average_precision_score'])
            history['accuracy_val'].append(evaluation_result_val['accuracy'])
            history['recall_val'].append(evaluation_result_val['recall'])
            history['precision_val'].append(evaluation_result_val['precision'])

            # if (epoch % 5 == 0):  # After the 5th epoch, plot all test real data and generated samples
            #     test_x, test_y = test_data
            #     generated_samples = self.generator(sample_Z(test_y.shape[0], self.Z_dim), training=False)
            #     plot_tsne_3d_with_plotly(test_x,test_y, generated_samples, epoch+1,save_dir=folder_name)
        

        return history

    def evaluate_model(self, x_test, y_test):
        probs, _, _ = self(x_test, training=False)
        if np.isnan(probs).any():
            print("Warning: NaN values detected in model outputs")
            probs = np.nan_to_num(probs, nan=0.5)  # Set NaN probabilities to 0.5

        predictions = tf.argmax(probs, axis=1).numpy()
        labels = tf.argmax(y_test, axis=1).numpy()

        precision, recall, F1score, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
        # Correct way to calculate ROC AUC score using the probabilities for the positive class
        roc_Auc_score = roc_auc_score(labels, probs[:, 1])  # assuming that the positive class is at index 1
        accuracy = accuracy_score(labels, predictions)
        ap_score = average_precision_score(labels, probs[:, 1])  # also using probabilities for the positive class

        TruePositives = np.sum(np.logical_and(labels == 1, predictions == 1))
        TrueNegatives = np.sum(np.logical_and(labels == 0, predictions == 0))
        FalsePositives = np.sum(np.logical_and(labels == 0, predictions == 1))
        FalseNegatives = np.sum(np.logical_and(labels == 1, predictions == 0))

        # Return metrics as a dictionary
        return {
            'roc_auc_score': roc_Auc_score,
            'F1_score': F1score,
            'average_precision_score': ap_score,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'TruePositives': TruePositives,
            'TrueNegatives': TrueNegatives,
            'FalsePositives': FalsePositives,
            'FalseNegatives': FalseNegatives
        }

        
    def print_classification_report(self, x_test, y_test):
            probs, _ ,_= self.discriminator(x_test, training=False)
            predictions = tf.argmax(probs, axis=1).numpy()

            labels = tf.argmax(y_test, axis=1).numpy()
            print("Classification Report:")
            print(classification_report(labels, predictions))


    def save_prob(self, x_test, y_test, run_name, dataset_name):
        probs, _, _ = self(x_test, training=False)
        
        df = pd.DataFrame({
            'features': list(x_test), 
            'probs': probs[:, 1], 
            'labels': list(tf.argmax(y_test, axis=1).numpy())
        })
        
        # Prepare directory and file path for saving
        base_path = 'data/processed_data/' + dataset_name + '/OCAN_results/'
        hidden_rep_shape = x_test.shape[0]
        dir_name = os.path.join(base_path, run_name)
        
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        
        file_path_test = os.path.join(dir_name, f"combined_test_df_{hidden_rep_shape}.pkl")
        
        print(f"Saving test dataframe to {file_path_test}")
        df.to_pickle(file_path_test)


