# combined_rep_integ_model.py
# Standard library imports
import warnings

# Third-party imports
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout
import time
from sklearn.metrics import average_precision_score
from models.HITANET.transformer import TransformerTime
from models.NN_REPRESENTATION.TABULAR_MODEL_NN import ClassificationModel
from models.HITANET.units import pad_matrix_new,pad_time,pad_failure_bits
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Concatenate
from utils.utils import  CombinedCustomLoss

# Set TensorFlow and MLflow configurations
tf.config.run_functions_eagerly(False)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# Suppress warnings
warnings.filterwarnings("ignore")

class CombinedModel(tf.keras.Model):
    def __init__(self, input_shape_tabular, max_len , combined_hidden_layers,dropout_rate_comb ,  dropout_rate_seq , droput_rate_tab,  layer=None, tab_hidden_states=None, batch_size=None,n_event_code=150,is_public_dataset=False,l2_lambda_comb=0.01,l2_lambda_tab=0.01,l2_lambda_seq=0.01,model_dim =256):
        super(CombinedModel, self).__init__()

        self.batch_size = batch_size
        self.max_len = max_len
        self.n_event_code = n_event_code
        self.bce_loss_tab = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.bce_loss_seq = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.total_loss = CombinedCustomLoss()

        # Tabular branch
        self.tabular_model = ClassificationModel(input_size=input_shape_tabular, hidden_layer_sizes=tab_hidden_states, dropout_rate=droput_rate_tab, l2_lambda=l2_lambda_tab,name='tabular_model')

        self.tabular_output = Dense(1, activation='sigmoid', name='tabular_output')  # Last dense layer for tabular data
        
        # Sequential branch
        self.sequential_model = TransformerTime(n_event_code, batch_size ,max_len, layer, dropout_rate_seq,is_public_dataset,l2_lambda_seq,model_dim,name='sequential_model')

        self.sequential_output = Dense(1, activation='sigmoid', name='sequential_output')  # Last dense layer for sequential data
        
        # Concatenate layer
        self.concatenate = Concatenate()
        
        # Final layers
        self.dense_layers = [Dense(units, activation='relu',name='combined_layer'+ str(units) ,kernel_regularizer=regularizers.l2(l2_lambda_comb)) for units in combined_hidden_layers]
        self.dropout = Dropout(dropout_rate_comb)
        self.final_output = Dense(1, activation='sigmoid',name='final_output')
    # @tf.function
    def call(self, inputs, training=False,return_representation=False,return_concated = False,train_tab=False, train_seq=False, train_combined=True):
        if train_tab:
            tabular_input = inputs
            combined_input = self.tabular_model(tabular_input)
            x = combined_input
            for dense_layer in self.dense_layers:
                x = dense_layer(x)
                x = self.dropout(x, training=training)
        elif train_seq:
            combined_input= self.sequential_model((inputs), training=True)
            x = combined_input
            for dense_layer in self.dense_layers:
                x = dense_layer(x)
                x = self.dropout(x, training=training)
        elif train_combined:
            tab_out , seq_out = inputs
            combined_input = self.concatenate([tab_out, seq_out])
            x = combined_input
            for i, dense_layer in enumerate(self.dense_layers):
                x = dense_layer(x)
                if i < len(self.dense_layers) - 1:
                    x = self.dropout(x, training=training)
            # x = self.combined_dense(x)
        else:
            x = inputs
        if return_representation:
            return x ,combined_input
        return self.final_output(x)

    def compile(self, optimizer, loss, metrics):
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.tabular_loss_tracker = tf.keras.metrics.Mean(name='tabular_loss')
        self.sequential_loss_tracker = tf.keras.metrics.Mean(name='sequential_loss')
        self.total_loss_tracker = tf.keras.metrics.Mean(name='total_loss')

 
    def fit(self, train_data, val_data=None, epochs=1, initial_epoch=0, train_flags=None, callbacks=None):
        if train_flags is None:
            train_flags = {}

        if callbacks is None:
            callbacks = []

        self._callbacks = tf.keras.callbacks.CallbackList(callbacks)
        self._callbacks.set_model(self)
        self._callbacks.set_params({
            'epochs': epochs,
            'steps': len(train_data[0]) // self.batch_size,
            'verbose': 1,
            'metrics': [],
            'initial_epoch': initial_epoch,
        })

        self._callbacks.on_train_begin()
        history = {
            'total_loss': [],
            'val_total_loss': [],
            'tabular_loss': [],
            'val_tabular_loss': [],
            'sequential_loss': [],
            'val_sequential_loss': [],
            'accuracy': [],
            'val_accuracy': [],
            'f1_score': [],
            'val_f1_score': [],
            'precision': [],
            'val_precision': [],
            'recall': [],
            'val_recall': [],
            'auc_pr': [],
            'val_auc_pr': [],
            'avg_precision': [],
            'val_avg_precision': [],
            'auc_roc': [],
            'val_auc_roc': [],
    

        }
        for epoch in range(initial_epoch, epochs):
            starttime = time.time()
            self._callbacks.on_epoch_begin(epoch)
            # Shuffle operation
            # indices = np.arange(len(train_data[0][0]))
            # np.random.shuffle(indices)

            # # Rearrange each array in train_data using the shuffled indices
            # shuffled_train_data = ([array[indices] for array in train_data[0]], train_data[1][indices])
            shuffled_train_data = train_data
            epoch_loss = 0
            num_samples = len(shuffled_train_data[1])

            progress_bar = self._batch_generator(shuffled_train_data, self.batch_size, train_flags)
            avg_precision_score = 0
            for batch_index, (inputs, labels) in enumerate(progress_bar):
                self._callbacks.on_train_batch_begin(batch_index)
                if train_flags.get('train_combined'):
                    tabular_input, event_seq_code_input, time_step_input,event_failure_sys, event_failure_user,  mask, mask_final, mask_code,lengths = inputs
                    y_tab , y_seq = labels
                    with tf.GradientTape(persistent=True) as tape:
                        tape.watch(self.trainable_variables)
                        tabular_representation = self.tabular_model(tabular_input, training=True)
                        y_tab_pred = (
                            self.tabular_output(
                                tabular_representation
                                )
                        )
                        # getting sequential prediction
                        sequential_representation = self.sequential_model((event_seq_code_input, time_step_input,event_failure_sys, event_failure_user, mask, mask_final, mask_code,lengths) , training = True)
                        y_seq_pred = (
                            self.sequential_output(
                                sequential_representation
                                )
                        )

                        # getting combined prediction
                        inputs = (tabular_representation, sequential_representation)
                        y_combined_pred = self(inputs, training=True, **train_flags)
                        

                        y_pred = (y_tab_pred, y_seq_pred,y_combined_pred)
                        total_loss = self.total_loss(labels,y_pred) 
                        
                            # Compute gradients
                    gradients = tape.gradient(total_loss , self.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                    

                    # Update the loss trackers
                    self.tabular_loss_tracker.update_state(self.total_loss.tabular_loss)
                    self.sequential_loss_tracker.update_state(self.total_loss.sequence_loss)
                    self.total_loss_tracker.update_state(total_loss)
                    
                    # Update the metrics for the combined output
                    self.compiled_metrics.update_state(y_seq, y_combined_pred)
                    avg_precision_score += average_precision_score(y_seq, y_combined_pred)
                elif train_flags.get('train_seq'):
                    event_seq_code_input, time_step_input, event_failure_sys, event_failure_user, mask, mask_final, mask_code,lengths = inputs
                    y_seq = labels
                    with tf.GradientTape() as tape:

                        y_seq_pred = self((event_seq_code_input, time_step_input, event_failure_sys, event_failure_user, mask, mask_final, mask_code,lengths), training=True, **train_flags)
                        loss = self.bce_loss_seq(y_seq, y_seq_pred)
                    gradients = tape.gradient(loss, self.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                    self.sequential_loss_tracker.update_state(loss)
                    self.compiled_metrics.update_state(y_seq, y_seq_pred)
                    avg_precision_score += average_precision_score(y_seq, y_seq_pred)
                elif train_flags.get('train_tab'):
                    tabular_input = inputs
                    y_tab = labels
                    with tf.GradientTape() as tape:
                        y_tab_pred = self((tabular_input), training=True, **train_flags)
                        loss = self.bce_loss_tab(y_tab, y_tab_pred)
                    gradients = tape.gradient(loss, self.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                    self.tabular_loss_tracker.update_state(loss)
                    self.compiled_metrics.update_state(y_tab, y_tab_pred)
                    avg_precision_score += average_precision_score(y_tab, y_tab_pred)
                

                self._callbacks.on_train_batch_end(batch_index)

            train_metrics_result = {m.name: m.result().numpy() for m in self.metrics if not m.name.startswith('val_')}
            train_metrics_result.update({'avg_precision': avg_precision_score / (batch_index + 1)})
            history['total_loss'].append(train_metrics_result['total_loss'])
            history['tabular_loss'].append(train_metrics_result['tabular_loss'])
            history['sequential_loss'].append(train_metrics_result['sequential_loss'])
            history['accuracy'].append(train_metrics_result['accuracy'])
            history['f1_score'].append(train_metrics_result['f1_score'])    
            history['precision'].append(train_metrics_result['precision'])
            history['recall'].append(train_metrics_result['recall'])
            history['auc_pr'].append(train_metrics_result['auc_pr'])
            history['avg_precision'].append(avg_precision_score)  
            history['auc_roc'].append(train_metrics_result['auc_roc'])

            print(f'Epoch {epoch+1}/{epochs} ============================================>' )
            print(f'Training Metrics: {train_metrics_result}')

            logs = {**train_metrics_result}
            if val_data is not None:
                val_metrics_result = self.evaluate_model(val_data, self.batch_size, train_flags)
                val_metrics_result = {'val_' + key: value for key, value in val_metrics_result.items()}

                # Now val_metrics_result has keys like 'val_loss', 'val_accuracy', etc.
                history['val_total_loss'].append(val_metrics_result['val_total_loss'])
                history['val_tabular_loss'].append(val_metrics_result['val_tabular_loss'])
                history['val_sequential_loss'].append(val_metrics_result['val_sequential_loss'])
                history['val_accuracy'].append(val_metrics_result['val_accuracy'])
                history['val_f1_score'].append(val_metrics_result['val_f1_score'])
                history['val_precision'].append(val_metrics_result['val_precision'])
                history['val_recall'].append(val_metrics_result['val_recall'])
                history['val_auc_pr'].append(val_metrics_result['val_auc_pr'])
                history['val_avg_precision'].append(val_metrics_result['val_avg_precision'])
                history['val_auc_roc'].append(val_metrics_result['val_auc_roc'])


                print(f'Validation Metrics: {val_metrics_result}')
                logs = {**train_metrics_result, **val_metrics_result}
            print(f'time elapsed :{((time.time()-starttime)/60)}')
            for metric in self.metrics:
                metric.reset_states()

            self._callbacks.on_epoch_end(epoch, logs= logs)

        self._callbacks.on_train_end()

        return history

    def _batch_generator(self, data, batch_size, train_flags):
        x, y = data
        

        if train_flags.get('train_combined'):
            num_samples = x[0].shape[0]
            x_tabular,event_seq_code, time_step , event_failure_sys, event_failure_user = x
            y_tab ,y_seq = y
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                event_seq_code_sliced = event_seq_code[start_idx:end_idx]
                time_step_sliced = time_step[start_idx:end_idx]
                event_seq_code_sliced, mask, mask_final, mask_code = pad_matrix_new( event_seq_code_sliced, self.max_len , self.n_event_code)

                time_step_sliced = np.array((pad_time(time_step_sliced))) 
                event_failure_sys_sliced = event_failure_sys[start_idx:end_idx]
                event_failure_sys_sliced = np.array((pad_failure_bits(event_failure_sys_sliced))) 
                event_failure_user_sliced = event_failure_user[start_idx:end_idx]
                event_failure_user_sliced = np.array((pad_failure_bits(event_failure_user_sliced))) 
                lengths = np.array([len(seq) for seq in event_seq_code_sliced])

                event_seq_code_sliced = tf.convert_to_tensor(event_seq_code_sliced, dtype=tf.float32)
                time_step_sliced = tf.convert_to_tensor(time_step_sliced, dtype=tf.float32)
                event_failure_sys_sliced = tf.convert_to_tensor(event_failure_sys_sliced, dtype=tf.float32)
                event_failure_user_sliced = tf.convert_to_tensor(event_failure_user_sliced, dtype=tf.float32)
                mask = tf.convert_to_tensor(mask, dtype=tf.float32)
                mask_final = tf.convert_to_tensor(mask_final, dtype=tf.float32)
                mask_code = tf.convert_to_tensor(mask_code, dtype=tf.float32)
                lengths = tf.convert_to_tensor(lengths, dtype=tf.float32)


                yield (x_tabular[start_idx:end_idx],event_seq_code_sliced, time_step_sliced,event_failure_sys_sliced,event_failure_user_sliced , mask, mask_final, mask_code,lengths), (tf.convert_to_tensor(y_tab[start_idx:end_idx].astype(float)),tf.convert_to_tensor(y_seq[start_idx:end_idx].astype(float)))
        elif train_flags.get('train_seq'):
            num_samples = x[0].shape[0]
            event_seq_code, time_step, event_failure_sys, event_failure_user = x
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                event_seq_code_sliced = event_seq_code[start_idx:end_idx]
                time_step_sliced = time_step[start_idx:end_idx]
                event_seq_code_sliced, mask, mask_final, mask_code = pad_matrix_new( event_seq_code_sliced, self.max_len , self.n_event_code)
                time_step_sliced = np.array((pad_time(time_step_sliced))) 
                event_failure_sys_sliced = event_failure_sys[start_idx:end_idx]
                event_failure_sys_sliced = np.array((pad_failure_bits(event_failure_sys_sliced))) 
                event_failure_user_sliced = event_failure_user[start_idx:end_idx]
                event_failure_user_sliced = np.array((pad_failure_bits(event_failure_user_sliced))) 
                lengths = np.array([len(seq) for seq in event_seq_code_sliced])

                event_seq_code_sliced = tf.convert_to_tensor(event_seq_code_sliced, dtype=tf.float32)
                time_step_sliced = tf.convert_to_tensor(time_step_sliced, dtype=tf.float32)
                event_failure_sys_sliced = tf.convert_to_tensor(event_failure_sys_sliced, dtype=tf.float32)
                event_failure_user_sliced = tf.convert_to_tensor(event_failure_user_sliced, dtype=tf.float32)
                mask = tf.convert_to_tensor(mask, dtype=tf.float32)
                mask_final = tf.convert_to_tensor(mask_final, dtype=tf.float32)
                mask_code = tf.convert_to_tensor(mask_code, dtype=tf.float32)
                lengths = tf.convert_to_tensor(lengths, dtype=tf.float32)


                yield (event_seq_code_sliced, time_step_sliced,event_failure_sys_sliced,event_failure_user_sliced, mask, mask_final, mask_code,lengths), tf.convert_to_tensor(y[start_idx:end_idx].astype(float))
        elif train_flags.get('train_tab'):
            x_tabular = x
            num_samples = x.shape[0]
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                yield x_tabular[start_idx:end_idx], tf.convert_to_tensor(y[start_idx:end_idx].astype(float))
        

    def evaluate_model(self, data, batch_size, train_flags):
        # Reset states of the metrics at the start of the evaluation
        for metric in self.metrics:
            metric.reset_states()
        avg_precision_score = 0
        # Iterate over batches and update metric states
        batch_index = 0
        for inputs, labels in self._batch_generator(data, batch_size, train_flags):
            if train_flags.get('train_combined') :
                tabular_input, event_seq_code_input, time_step_input,event_failure_sys, event_failure_user,  mask, mask_final, mask_code,lengths = inputs
                y_tab , y_seq = labels
                # Forward pass
                # getting tabular prediction
                tabular_representation = self.tabular_model(tabular_input, training=True)
                y_tab_pred = (
                    self.tabular_output(
                        tabular_representation
                        )
                )
                # getting sequential prediction
                sequential_representation = self.sequential_model((event_seq_code_input, time_step_input,event_failure_sys, event_failure_user, mask, mask_final, mask_code,lengths) , training = True)
                y_seq_pred = (
                    self.sequential_output(
                        sequential_representation
                        )
                )

                # getting combined prediction
                inputs = (tabular_representation, sequential_representation)
                y_combined_pred = self(inputs, training=True, **train_flags)
                

                y_pred = (y_tab_pred, y_seq_pred,y_combined_pred)
                total_loss = self.total_loss(labels,y_pred) 
                # Update the loss trackers
                self.tabular_loss_tracker.update_state(self.total_loss.tabular_loss)
                self.sequential_loss_tracker.update_state(self.total_loss.sequence_loss)
                self.total_loss_tracker.update_state(total_loss)
                
                # Update the metrics for the combined output
                self.compiled_metrics.update_state(y_seq, y_combined_pred)
                avg_precision_score += average_precision_score(y_seq, y_combined_pred)
            elif train_flags.get('train_seq'):
                y_seq = labels
                event_seq_code_input, time_step_input, event_failure_sys, event_failure_user, mask, mask_final, mask_code,lengths = inputs
                y_seq_pred = self((event_seq_code_input, time_step_input, event_failure_sys, event_failure_user, mask, mask_final, mask_code,lengths), training=True, **train_flags)
                loss = self.bce_loss_seq(y_seq, y_seq_pred)
                self.sequential_loss_tracker.update_state(loss)
                self.compiled_metrics.update_state(y_seq, y_seq_pred)
                avg_precision_score += average_precision_score(y_seq, y_seq_pred)
            elif train_flags.get('train_tab'):
                tabular_input = inputs
                y_tab = labels
                y_tab_pred = self((tabular_input), training=True, **train_flags)
                loss = self.bce_loss_tab(y_tab, y_tab_pred)
                self.tabular_loss_tracker.update_state(loss)
                self.compiled_metrics.update_state(y_tab, y_tab_pred)
                avg_precision_score += average_precision_score(y_tab, y_tab_pred)
            batch_index+= 1

        # Gather the final results
        results = {m.name: m.result().numpy() for m in self.metrics}
        results['avg_precision'] = avg_precision_score / (batch_index+ 1)
        return results
    
    def get_representation_in_batches(self, data, batch_size, train_flags):
        final_layer_reps = []
        concat_layer_reps = []
        
        # Iterate over batches
        for inputs, _ in self._batch_generator(data, batch_size, train_flags):
            if train_flags.get('train_combined') :
                # Unpack inputs based on the training flags
                if train_flags.get('train_combined'):
                    tabular_input, event_seq_code_input, time_step_input,event_failure_sys, event_failure_user,  mask, mask_final, mask_code,lengths = inputs

                    tabular_representation = self.tabular_model(tabular_input, training=False)

                    # getting sequential prediction
                    sequential_representation = self.sequential_model((event_seq_code_input, time_step_input,event_failure_sys, event_failure_user, mask, mask_final, mask_code,lengths) , training = False)

                    # getting combined prediction
                    inputs = (tabular_representation, sequential_representation)
                    final_layer_rep ,concat_layer_rep = self(inputs, training=False,return_representation=True, **train_flags)
                        
            elif train_flags.get('train_seq'):
                event_seq_code_input, time_step_input, event_failure_sys, event_failure_user, mask, mask_final, mask_code,lengths = inputs
                final_layer_rep ,concat_layer_rep = self((event_seq_code_input, time_step_input, event_failure_sys, event_failure_user, mask, mask_final, mask_code,lengths), training=True,return_representation=True, **train_flags)

            elif train_flags.get('train_tab'):
                tabular_input = inputs
                final_layer_rep ,concat_layer_rep = self((tabular_input), training=True,return_representation=True, **train_flags)

            final_layer_reps.append(final_layer_rep)
            concat_layer_reps.append(concat_layer_rep)

        # Concatenate all batch predictions
        final_layer_reps = tf.concat(final_layer_reps, axis=0)
        concat_layer_reps = tf.concat(concat_layer_reps, axis=0)
        return final_layer_reps, concat_layer_reps
    
    def predict_in_batches(self, data, batch_size, train_flags):
        predictions = []

        
        # Iterate over batches
        for inputs, _ in self._batch_generator(data, batch_size, train_flags):
            if train_flags.get('train_combined') :
                # Unpack inputs based on the training flags
                if train_flags.get('train_combined'):
                    tabular_input, event_seq_code_input, time_step_input,event_failure_sys, event_failure_user,  mask, mask_final, mask_code,lengths = inputs

                    tabular_representation = self.tabular_model(tabular_input, training=False)

                    # getting sequential prediction
                    sequential_representation = self.sequential_model((event_seq_code_input, time_step_input,event_failure_sys, event_failure_user, mask, mask_final, mask_code,lengths) , training = False)

                    # getting combined prediction
                    inputs = (tabular_representation, sequential_representation)
                    prediction = self(inputs, training=False,return_representation=False, **train_flags)
                        
            elif train_flags.get('train_seq'):
                event_seq_code_input, time_step_input, event_failure_sys, event_failure_user, mask, mask_final, mask_code,lengths = inputs
                prediction  = self((event_seq_code_input, time_step_input, event_failure_sys, event_failure_user, mask, mask_final, mask_code,lengths), training=True,return_representation=False, **train_flags)

            elif train_flags.get('train_tab'):
                tabular_input = inputs
                prediction = self((tabular_input), training=True,return_representation=False, **train_flags)

            predictions.append(prediction)

        # Concatenate all batch predictions
        predictions = tf.concat(predictions, axis=0)

        return predictions