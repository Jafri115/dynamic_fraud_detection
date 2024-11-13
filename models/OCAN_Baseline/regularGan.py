from datetime import datetime
import os
import sys
import time
from models.OCAN_Baseline.bg_utils import check_nan,  discriminator_loss_reg,  generator_loss_reg, gradient_penalty,  sample_Z
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score,precision_score, recall_score ,average_precision_score,accuracy_score
from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_recall_fscore_support
import scipy.linalg
from tensorflow.keras import layers, initializers
from tensorflow.keras.layers import Dense, Activation, Dropout,BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.initializers import GlorotNormal,HeNormal

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

class regular_GANModel(tf.keras.Model):
    def __init__(self,param_dict,total_samples,checkpoint_path= None):
        super(regular_GANModel, self).__init__()
        self.Z_dim = param_dict['G_D_layers'][0][0]
        self.total_samples = total_samples
        self.mb_size = 512
        self.G_dropout = param_dict['g_dropouts']
        self.D_dropout = param_dict['d_dropouts']
        self.generator  = Generator( G_dims = param_dict['G_D_layers'][0],input_dim=param_dict['dim_inp'],dropout_rates=param_dict['g_dropouts'],batchNorm=param_dict['batch_norm_g'])
        self.discriminator  = Discriminator( D_dims = param_dict['G_D_layers'][1],input_dim=param_dict['dim_inp'],dropout_rates=param_dict['d_dropouts'],batchNorm=param_dict['batch_norm_d'])
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
        
    def train_step(self, real_samples, labels,train_disc=True,train_gen=True):
        y_fake_mb = tf.one_hot(tf.ones(labels.shape[0], dtype=tf.int32), depth=2)
        # Update the discriminator
        if train_disc:
            with tf.GradientTape() as disc_tape:
                generated_samples = self.generator(sample_Z(labels.shape[0], self.Z_dim ), training=False)
                D_prob_real, D_logit_real, _ = self.discriminator(real_samples, training=True)
                D_prob_gen, D_logit_gen, _ = self.discriminator(generated_samples, training=True)

                disc_loss = discriminator_loss_reg(D_logit_real, D_logit_gen, labels, y_fake_mb)

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
        else:
            generated_samples = self.generator(sample_Z(labels.shape[0], self.Z_dim ), training=False)
            D_prob_real, D_logit_real, _ = self.discriminator(real_samples, training=False)
            D_prob_gen, D_logit_gen, _ = self.discriminator(generated_samples, training=False)
            disc_loss = discriminator_loss_reg(D_logit_real, D_logit_gen, labels, y_fake_mb)

        # Update the generator
        if train_gen:
            with tf.GradientTape() as gen_tape:
                generated_samples = self.generator(sample_Z(labels.shape[0], self.Z_dim ), training=True)
                D_prob_gen, D_logit_gen, _ = self.discriminator(generated_samples, training=False)
                gen_loss = generator_loss_reg(D_logit_gen, y_fake_mb)

            gradients_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            # gradients_gen = [tf.clip_by_value(g, -0.01, 0.01) for g in gradients_gen]
            self.generator_optimizer.apply_gradients(zip(gradients_gen, self.generator.trainable_variables))
        return disc_loss, gen_loss


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
            'mean_distance': [],
            'FID': []
        }
        best_f1_score = 0
        best_epoch = 0
        wait = 0
        time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_name = os.path.join('plots/TNSE_plots', time_now)
        # Create directory if it doesn't exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        print(f"Saving plots to {folder_name}")
        for epoch in range(initial_epoch, epochs):
            batch_index = 0

            epoch_D_losses = []
            epoch_G_losses = []
            if epoch % 3 == 0 and epoch < 15:
                for x_batch, y_batch in train_data:
                    epoch_D_loss, epoch_G_loss = self.train_step(x_batch, y_batch,True,True)  # Train model
                    batch_index += 1
                    real_samples = x_batch
                    epoch_D_losses.append(epoch_D_loss)
                    epoch_G_losses.append(epoch_G_loss)
            else:
                for x_batch, y_batch in train_data:
                    epoch_D_loss, epoch_G_loss = self.train_step(x_batch, y_batch,False,True)  # Train model
                    batch_index += 1
                    real_samples = x_batch
                    epoch_D_losses.append(epoch_D_loss)
                    epoch_G_losses.append(epoch_G_loss)

            epoch_D_loss = np.mean(epoch_D_losses)
            epoch_G_loss = np.mean(epoch_G_losses)

            disc_metrics = self.evaluate_discriminator(test_data[0])
            gen_metrics = self.evaluate_generator(test_data[0], test_data[0].shape[0])
            print(f"Epoch {epoch + 1} -Discriminator Loss: {epoch_D_loss}, Generator Loss: {epoch_G_loss},- Discriminator F1 Score: {disc_metrics['f1_score']:.4f}, Precision: {disc_metrics['precision']:.4f}, Recall: {disc_metrics['recall']:.4f}, Generator Fooled Rate: {gen_metrics['fooled_rate']:.4f}")

            # Update history dict

            history['D_loss_train'].append(epoch_D_loss)
            history['G_loss_train'].append(epoch_G_loss)
            # history['roc_auc_val'].append(evaluation_result_val['roc_auc_score'])
            # history['F1score_val'].append(evaluation_result_val['F1_score'])
            # history['avg_prec_val'].append(evaluation_result_val['average_precision_score'])
            # history['accuracy_val'].append(evaluation_result_val['accuracy'])
            # history['recall_val'].append(evaluation_result_val['recall'])
            # history['precision_val'].append(evaluation_result_val['precision'])
            # history['mean_distance'].append(evaluation_result_val['mean_distance'])
            # history['FID'].append(evaluation_result_val['FID'])

            # if (epoch % 5 == 0):  # After the 5th epoch, plot all test real data and generated samples
            #     test_x, test_y = test_data
            #     generated_samples = self.generator(sample_Z(test_y.shape[0], self.Z_dim), training=False)
            #     plot_tsne_3d_with_plotly(test_x,test_y, generated_samples, epoch+1,save_dir=folder_name)
        

        return history
    def evaluate_generator(self, real_data, num_samples):
        # Generate fake data
        # noise = sample_Z(num_samples, self.generator.Z_dim)
        fake_data = fake_data =  self.generator(sample_Z(num_samples, self.Z_dim ), training=False)

        # Discriminator's response to fake data
        prob, fake_logits, final_hidden_output = self.discriminator(fake_data, training=False)

        # The ideal case is that discriminator is fooled to believe data is real (close to 0)
        fooled_rate = tf.reduce_mean(tf.nn.sigmoid(fake_logits))  # Higher is better, indicates more samples fooled the discriminator

        return {
            'fooled_rate': fooled_rate.numpy()  # Assuming TensorFlow for computation
        }
    def evaluate_discriminator(self, real_data, fake_data=None):
        if fake_data is None:
            # Generate fake data if not provided
            fake_data =  self.generator(sample_Z(real_data.shape[0], self.Z_dim ), training=False)
        
        # Obtain logits for real and fake data
        prob, real_logits, final_hidden_output = self.discriminator(real_data, training=False)
        prob, fake_logits, final_hidden_output = self.discriminator(fake_data, training=False)

        # Prepare labels (real=0, fake=1)
        real_labels = tf.one_hot(tf.zeros_like(real_logits[:, 0], dtype=tf.int32), depth=2)
        fake_labels = tf.one_hot(tf.ones_like(fake_logits[:, 0], dtype=tf.int32), depth=2)

        # Concatenate logits and labels
        logits = tf.concat([real_logits, fake_logits], axis=0)
        labels = tf.concat([real_labels, fake_labels], axis=0)

        # Apply a sigmoid to the logits to get binary predictions
        predictions = tf.round(tf.nn.sigmoid(logits))

        # Calculate metrics
        precision, recall, f1_score, _ = precision_recall_fscore_support(labels.numpy(), predictions.numpy(), average='macro')
        
        return {
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall
        }
    def evaluate_model(self, x_test, y_test):
            
            generated_samples = self.generator(sample_Z(y_test.shape[0], self.Z_dim ), training=False)
            real_features, D_logit_real, _ = self.discriminator(x_test, training=False)
            generated_features, D_logit_gen, _ = self.discriminator(generated_samples, training=False)


            # Calculate the average Euclidean distance
            dists = cdist(real_features.numpy(), generated_features.numpy(), 'euclidean')
            mean_distance = np.mean(np.min(dists, axis=0))  # Mean minimum distance from each generated to nearest real

            # Calculate statistics like FID by considering the layers as features (adapted for non-image data)
            mu_real, sigma_real = real_features.numpy().mean(axis=0), np.cov(real_features.numpy(), rowvar=False)
            mu_gen, sigma_gen = generated_features.numpy().mean(axis=0), np.cov(generated_features.numpy(), rowvar=False)

            # Using a simple FID-like calculation
            fid_distance = np.sum((mu_real - mu_gen) ** 2) + np.trace(sigma_real + sigma_gen - 2 * scipy.linalg.sqrtm(np.dot(sigma_real, sigma_gen)))


            # Evaluate with a classifier if labels are provided
            # if y_test is not None:
            #     predictions = np.argmax(generated_features, axis=1)
            #     labels = np.argmax(y_test, axis=1)

            #     precision, recall, f1_score, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
            #     roc_auc = roc_auc_score(labels, predictions)
            #     accuracy = accuracy_score(labels, predictions)
            #     ap_score = average_precision_score(labels, generated_features[:, 1])

            #     metrics = {
            #         'mean_distance': mean_distance,
            #         'FID': np.real(fid_distance),
            #         'roc_auc_score': roc_auc,
            #         'f1_score': f1_score,
            #         'average_precision_score': ap_score,
            #         'accuracy': accuracy,
            #         'recall': recall,
            #         'precision': precision
            #     }
            # else:
            metrics = {
                'mean_distance': mean_distance,
                'FID': np.real(fid_distance)
            }

            return metrics
        
    def print_classification_report(self, x_test, y_test):
            probs, _ ,_= self.discriminator(x_test, training=False)
            predictions = tf.argmax(probs, axis=1).numpy()

            labels = tf.argmax(y_test, axis=1).numpy()
            print("Classification Report:")
            print(classification_report(labels, predictions))

