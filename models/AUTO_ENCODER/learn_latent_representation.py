# learn_latent_representation.py
import os
import datetime
import pytz
import json
import numpy as np
import sys
import tensorflow as tf
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping

en_ae = int(sys.argv[1]) # 1 for training, 2 for wiki 

class Autoencoder(Model):
    def __init__(self, input_size, latent_space_size, encoder_layer_sizes, decoder_layer_sizes):
        super(Autoencoder, self).__init__()
        
        # Initialize the encoder as a Sequential model
        self.encoder = Sequential()
        self.encoder.add(Input(shape=(input_size,)))
        for size in encoder_layer_sizes:
            self.encoder.add(Dense(size, activation='relu', kernel_initializer='he_normal'))
            self.encoder.add(Dropout(0.2))
        self.encoder.add(Dense(latent_space_size, activation='relu', kernel_initializer='he_normal'))  # Latent space layer

        # Initialize the decoder as a Sequential model
        self.decoder = Sequential()
        self.decoder.add(Input(shape=(latent_space_size,)))
        for size in decoder_layer_sizes:
            self.decoder.add(Dense(size, activation='relu', kernel_initializer='he_normal'))
        self.decoder.add(Dense(input_size, activation='sigmoid', kernel_initializer='glorot_normal'))  # Output layer

    def call(self, inputs):
        # Encoding and Decoding
        latent_space = self.encoder(inputs)
        outputs = self.decoder(latent_space)
        return outputs

    def get_encoder(self):
        return self.encoder
    
def save_model(model, data_type, param_dict, X_train, base_path='./saved_models'):
    os.makedirs(base_path, exist_ok=True)
        
    # Calculate metrics
    X_pred = model.predict(X_train)

    encoder = autoencoder.get_encoder()

    mse = mean_squared_error(X_train,X_pred)

    # Create a timestamp
    germany_tz = pytz.timezone('Europe/Berlin')
    germany_time = datetime.datetime.now(germany_tz)
    timestamp = germany_time.strftime('%Y%m%d-%H%M%S')
    
    # Create the data type specific path
    data_type_path = os.path.join(base_path, data_type)
    data_time_path = os.path.join(data_type_path, f"{mse:.2f}__{timestamp}")
    os.makedirs(data_type_path, exist_ok=True)

    model_filename = f"D_{data_type}__{timestamp}"
    metrics_filename = f"D_{data_type}__{timestamp}__metrics.json"
    model_filepath = os.path.join(data_time_path, model_filename)
    metrics_filepath = os.path.join(data_time_path, metrics_filename)

    # Save the model
    encoder.save(model_filepath,save_format="tf")

    # Save the metrics in a JSON file
    metrics = {
        'mse': mse,

    }
    metrics.update(param_dict)

    with open(metrics_filepath, 'w') as f:
        json.dump(metrics, f)
    
    return model_filepath, metrics_filepath

def save_representation(model,X_train,y_train,X_test, y_test ,data_type, input_dim ,laten_dim, base_path='./data/processed_data/tab_data_representations'):
    # Ensure the base path exists
    os.makedirs(base_path, exist_ok=True)
    X_ben_train= model.predict(X_train)
    X_test= model.predict(X_test)

    # Create a timestamp
    germany_tz = pytz.timezone('Europe/Berlin')
    germany_time = datetime.datetime.now(germany_tz)
    timestamp = germany_time.strftime('%Y%m%d-%H%M%S')

    file_path = os.path.join(base_path, f"{data_type}__{input_dim}__{laten_dim}__{timestamp}")
    os.makedirs(file_path, exist_ok=True)


    repre_filename =  f"{data_type}__{input_dim}__{laten_dim}__{timestamp}"

    model_filepath = os.path.join(file_path, repre_filename)

    np.savez(model_filepath, X_ben_train=X_ben_train, y_ben_train=y_train, X_test=X_test, y_test=y_test)
    
    return model_filepath

# Usage example
if __name__ == "__main__":
    data = np.load('/home/wjafri/fraud_detection_thesis/data/processed_data/processed_tabular_data.npz')
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    latent_space_size = 300
    encoder_layer_sizes = [128, 256]  # Encoder layer sizes
    decoder_layer_sizes = [256, 128]  # Decoder layer sizes
    input_size = 196

    if en_ae == 1 :

        param_dict = {
            'input_size': input_size,
            'latent_space_size': latent_space_size,
            'encoder_layer_sizes': encoder_layer_sizes,
            'decoder_layer_sizes': decoder_layer_sizes,
        }
        # Setup TensorBoard
        log_dir = os.path.join("logs", "Autoencoder", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


        autoencoder = Autoencoder(input_size, latent_space_size, encoder_layer_sizes, decoder_layer_sizes)

        # Compile and train the autoencoder
        learning_rate = 0.0005
        optimizer = Adam(learning_rate=learning_rate)
        autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')

        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)

        autoencoder.fit(X_train, X_train, epochs=50, batch_size=128, shuffle=True, validation_split=0.2,callbacks=[tensorboard_callback])
        data_type = 'dataset_tab_data'

        model_filepath = save_model(autoencoder, data_type,param_dict, X_train=X_train)
        print(f"Model saved to {model_filepath}")

        # Create the encoder model from the trained autoencoder
        encoder = autoencoder.get_encoder()
        # Create a mask for label 1 samples
        label_1_mask = (y_train == 1)

        # Select samples with label 1
        X_train_label_1 = X_train[label_1_mask]

        # Remove label 1 samples from X_train and y_train
        X_train = X_train[~label_1_mask]
        y_train = y_train[~label_1_mask]

        # Extract the latent representation
        latent_representations = encoder.predict(X_train)
        data_type = 'dataset_tab'
        # Save the latent representations
        latent_representations_filepath = save_representation(encoder,X_train,y_train,X_test, y_test ,data_type, input_size, latent_space_size)
        print(f"Latent representations saved to {latent_representations_filepath}")

    
    elif en_ae == 2 :
        
        saved_model_dir = 'saved_models/dataset_tab_data/0.52__20231112-152315/D_dataset_tab_data__20231112-152315'

        # load model
        encoder = tf.keras.models.load_model(saved_model_dir)
        data_type = 'dataset_tab_data_benign'
        # Save the latent representations
        latent_representations_filepath = save_representation(encoder,X_train,y_train,X_test, y_test ,data_type, input_size, latent_space_size)
        print(f"Latent representations saved to {latent_representations_filepath}")