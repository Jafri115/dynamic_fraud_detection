import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
import pickle as cPickle

class TLSTM(tf.Module):
    def init_weights(self, input_dim, output_dim, name, std=0.1, reg=None):
        return tf.Variable(tf.random.normal([input_dim, output_dim], stddev=std), name=name)

    def init_bias(self, output_dim, name):
        return tf.Variable(tf.constant(1.0, shape=[output_dim]), name=name)

    def no_init_weights(self, input_dim, output_dim, name):
        return tf.Variable(tf.zeros([input_dim, output_dim]), name=name)

    def no_init_bias(self, output_dim, name):
        return tf.Variable(tf.zeros([output_dim]), name=name)

    def __init__(self, input_dim, output_dim, hidden_dim, fc_dim, train):
        super(TLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if train == 1:
            self.Wi = self.init_weights(self.input_dim, self.hidden_dim, name='Input_Hidden_weight')
            self.Ui = self.init_weights(self.hidden_dim, self.hidden_dim, name='Input_State_weight')
            self.bi = self.init_bias(self.hidden_dim, name='Input_Hidden_bias')

            self.Wf = self.init_weights(self.input_dim, self.hidden_dim, name='Forget_Hidden_weight')
            self.Uf = self.init_weights(self.hidden_dim, self.hidden_dim, name='Forget_State_weight')
            self.bf = self.init_bias(self.hidden_dim, name='Forget_Hidden_bias')

            self.Wog = self.init_weights(self.input_dim, self.hidden_dim, name='Output_Hidden_weight')
            self.Uog = self.init_weights(self.hidden_dim, self.hidden_dim, name='Output_State_weight')
            self.bog = self.init_bias(self.hidden_dim, name='Output_Hidden_bias')

            self.Wc = self.init_weights(self.input_dim, self.hidden_dim, name='Cell_Hidden_weight')
            self.Uc = self.init_weights(self.hidden_dim, self.hidden_dim, name='Cell_State_weight')
            self.bc = self.init_bias(self.hidden_dim, name='Cell_Hidden_bias')

            self.W_decomp = self.init_weights(self.hidden_dim, self.hidden_dim, name='Decomposition_Hidden_weight')
            self.b_decomp = self.init_bias(self.hidden_dim, name='Decomposition_Hidden_bias_enc')

            self.Wo = self.init_weights(self.hidden_dim, fc_dim, name='Fc_Layer_weight')
            self.bo = self.init_bias(fc_dim, name='Fc_Layer_bias')

            self.W_softmax = self.init_weights(fc_dim, output_dim, name='Output_Layer_weight')
            self.b_softmax = self.init_bias(output_dim, name='Output_Layer_bias')

        else:
            self.Wi = self.no_init_weights(self.input_dim, self.hidden_dim, name='Input_Hidden_weight')
            self.Ui = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Input_State_weight')
            self.bi = self.no_init_bias(self.hidden_dim, name='Input_Hidden_bias')

            self.Wf = self.no_init_weights(self.input_dim, self.hidden_dim, name='Forget_Hidden_weight')
            self.Uf = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Forget_State_weight')
            self.bf = self.no_init_bias(self.hidden_dim, name='Forget_Hidden_bias')

            self.Wog = self.no_init_weights(self.input_dim, self.hidden_dim, name='Output_Hidden_weight')
            self.Uog = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Output_State_weight')
            self.bog = self.no_init_bias(self.hidden_dim, name='Output_Hidden_bias')

            self.Wc = self.no_init_weights(self.input_dim, self.hidden_dim, name='Cell_Hidden_weight')
            self.Uc = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Cell_State_weight')
            self.bc = self.no_init_bias(self.hidden_dim, name='Cell_Hidden_bias')

            self.W_decomp = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Decomposition_Hidden_weight')
            self.b_decomp = self.no_init_bias(self.hidden_dim, name='Decomposition_Hidden_bias_enc')

            self.Wo = self.no_init_weights(self.hidden_dim, fc_dim, name='Fc_Layer_weight')
            self.bo = self.no_init_bias(fc_dim, name='Fc_Layer_bias')

            self.W_softmax = self.no_init_weights(fc_dim, output_dim, name='Output_Layer_weight')
            self.b_softmax = self.no_init_bias(output_dim, name='Output_Layer_bias')

    def TLSTM_Unit(self, prev_hidden_memory, concat_input):
        prev_hidden_state, prev_cell = tf.unstack(prev_hidden_memory)

        batch_size = tf.shape(concat_input)[0]
        x = tf.slice(concat_input, [0, 1], [batch_size, self.input_dim])
        t = tf.slice(concat_input, [0, 0], [batch_size, 1])

        # Dealing with time irregularity

        # Map elapse time in days or months
        T = self.map_elapse_time(t)

        # Decompose the previous cell if there is a elapse time
        C_ST = tf.nn.tanh(tf.matmul(prev_cell, self.W_decomp) + self.b_decomp)
        C_ST_dis = tf.multiply(T, C_ST)
        # if T is 0, then the weight is one
        prev_cell = prev_cell - C_ST + C_ST_dis

        # Input gate
        i = tf.sigmoid(tf.matmul(x, self.Wi) + tf.matmul(prev_hidden_state, self.Ui) + self.bi)

        # Forget Gate
        f = tf.sigmoid(tf.matmul(x, self.Wf) + tf.matmul(prev_hidden_state, self.Uf) + self.bf)

        # Output Gate
        o = tf.sigmoid(tf.matmul(x, self.Wog) + tf.matmul(prev_hidden_state, self.Uog) + self.bog)

        # Candidate Memory Cell
        C = tf.nn.tanh(tf.matmul(x, self.Wc) + tf.matmul(prev_hidden_state, self.Uc) + self.bc)

        # Current Memory cell
        Ct = f * prev_cell + i * C

        # Current Hidden state
        current_hidden_state = o * tf.nn.tanh(Ct)

        return tf.stack([current_hidden_state, Ct])

    def get_states(self, inputs, times):  # Returns all hidden states for the samples in a batch
        batch_size = tf.shape(inputs)[0]
        scan_input_ = tf.transpose(inputs, perm=[2, 0, 1])
        scan_input = tf.transpose(scan_input_)  # scan input is [seq_length x batch_size x input_dim]
        scan_time = tf.transpose(times)  # scan_time [seq_length x batch_size]
        scan_input = tf.cast(scan_input, dtype=tf.float32)
        scan_time = tf.cast(scan_time, dtype=tf.float32)
        initial_hidden = tf.zeros([batch_size, self.hidden_dim], tf.float32)
        ini_state_cell = tf.stack([initial_hidden, initial_hidden])
        # make scan_time [seq_length x batch_size x 1]
        scan_time = tf.reshape(scan_time, [tf.shape(scan_time)[0], tf.shape(scan_time)[1], 1])
        concat_input = tf.concat([scan_time, scan_input], 2)  # [seq_length x batch_size x input_dim+1]
        packed_hidden_states = tf.scan(self.TLSTM_Unit, concat_input, initializer=ini_state_cell, name='states')
        all_states = packed_hidden_states[:, 0, :, :]
        return all_states

    def get_output(self, state, keep_prob):
        output = tf.nn.relu(tf.matmul(state, self.Wo) + self.bo)
        output = tf.nn.dropout(output, rate=1 - keep_prob)
        output = tf.matmul(output, self.W_softmax) + self.b_softmax
        return output

    def get_outputs(self, inputs, times, keep_prob):  # Returns all the outputs
        all_states = self.get_states(inputs, times)
        all_outputs = tf.map_fn(lambda state: self.get_output(state, keep_prob), all_states)
        output = tf.reverse(all_outputs, [0])[0, :, :]  # Compatible with tensorflow 2.x
        return output

    def get_cost_acc(self, inputs, labels, times, keep_prob):
        logits = self.get_outputs(inputs, times, keep_prob)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        y_pred = tf.argmax(logits, 1)
        y = tf.argmax(labels, 1)
        return cross_entropy, y_pred, y, logits, labels

    def map_elapse_time(self, t):
        c1 = tf.constant(1, dtype=tf.float32)
        c2 = tf.constant(2.7183, dtype=tf.float32)
        T = tf.math.divide(c1, tf.math.log(t + c2), name='Log_elapse_time')
        Ones = tf.ones([1, self.hidden_dim], dtype=tf.float32)
        T = tf.matmul(T, Ones)
        return T

def generate_synthetic_data(num_samples, seq_len, input_dim, output_dim):
    data = np.random.rand(num_samples, seq_len, input_dim).astype(np.float32)
    elapsed_time = np.random.rand(num_samples, seq_len).astype(np.float32)
    labels = np.random.randint(0, 2, size=(num_samples, output_dim)).astype(np.float32)
    return data, elapsed_time, labels

def load_pkl(path):
    with open(path, 'rb') as f:
        obj = cPickle.load(f, encoding='bytes')
    return obj

def training(data, elapsed_time, labels, number_train_batches, learning_rate, training_epochs, train_dropout_prob, hidden_dim, fc_dim, key, model_path):
    input_dim = data[0].shape[2]
    output_dim = labels[0].shape[1]

    # Initialize your custom LSTM model here
    lstm = TLSTM(input_dim, output_dim, hidden_dim, fc_dim, key)

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def train_step(batch_xs, batch_ys, batch_ts):
        with tf.GradientTape() as tape:
            loss, y_pred, y_true, logits, labels = lstm.get_cost_acc(batch_xs, batch_ys, batch_ts, train_dropout_prob)
        gradients = tape.gradient(loss, lstm.trainable_variables)
        optimizer.apply_gradients(zip(gradients, lstm.trainable_variables))
        return loss, y_pred, y_true

    for epoch in range(training_epochs):
        total_cost = 0
        all_y_true = []
        all_y_pred = []
        for i in range(number_train_batches):
            batch_xs, batch_ys, batch_ts = data[i], labels[i], elapsed_time[i]
            batch_ts = np.reshape(batch_ts, [batch_ts.shape[0], batch_ts.shape[2]])
            loss, y_pred, y_true = train_step(batch_xs, batch_ys, batch_ts)
            total_cost += loss.numpy()
            all_y_true.extend(y_true.numpy())
            all_y_pred.extend(y_pred.numpy())
        # Calculate metrics
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)

        f1 = f1_score(all_y_true, all_y_pred, average='weighted')
        precision = precision_score(all_y_true, all_y_pred, average='weighted')
        recall = recall_score(all_y_true, all_y_pred, average='weighted')

        print(f"Epoch {epoch + 1}/{training_epochs} - Loss: {total_cost / number_train_batches} - F1 Score: {f1} - Precision: {precision} - Recall: {recall}")


    # Save the model
    checkpoint = tf.train.Checkpoint(model=lstm)
    checkpoint.save(file_prefix=model_path)

def testing(data, elapsed_time, labels, number_test_batches, hidden_dim, fc_dim, key, model_path):
    input_dim = data[0].shape[2]
    output_dim = labels[0].shape[1]

    # Initialize your custom LSTM model here
    lstm = TLSTM(input_dim, output_dim, hidden_dim, fc_dim, key)

    # Restore the model
    checkpoint = tf.train.Checkpoint(model=lstm)
    checkpoint.restore(tf.train.latest_checkpoint('./')).expect_partial()

    all_y_true = []
    all_y_pred = []
    logits_all = []
    for i in range(number_test_batches):
        batch_xs, batch_ys, batch_ts = data[i], labels[i], elapsed_time[i]
        batch_ts = np.reshape(batch_ts, [batch_ts.shape[0], batch_ts.shape[2]])
        logits = lstm.get_outputs(batch_xs, batch_ts, keep_prob=1.0)
        y_pred = tf.argmax(logits, axis=1).numpy()
        y_true = tf.argmax(batch_ys, axis=1).numpy()
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        logits_all.extend(logits.numpy())

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    accuracy = accuracy_score(all_y_true, all_y_pred)
    roc_auc = roc_auc_score(all_y_true, tf.sigmoid(logits_all).numpy()[:, 1])
    f1 = f1_score(all_y_true, all_y_pred)
    precision = precision_score(all_y_true, all_y_pred)
    recall = recall_score(all_y_true, all_y_pred)
    average_precision = average_precision_score(all_y_true, tf.sigmoid(logits_all).numpy()[:, 1])

    print(f"Accuracy: {accuracy}")
    print(f"ROC AUC: {roc_auc}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Average Precision: {average_precision}")

path = 'data/processed_data/Split0'
data_train_batches = load_pkl(path + '/data_train.pkl')
elapsed_train_batches = load_pkl(path + '/elapsed_train.pkl')
labels_train_batches = load_pkl(path + '/label_train.pkl')
number_train_batches = len(data_train_batches)

data_test_batches = load_pkl(path + '/data_test.pkl')
elapsed_test_batches = load_pkl(path + '/elapsed_test.pkl')
labels_test_batches = load_pkl(path + '/label_test.pkl')
number_test_batches = len(data_test_batches)
hidden_dim = 128
fc_dim = 64
learning_rate = 0.01
training_epochs = 2
dropout_prob = 0.5
model_path = "./saved_models/t_lstm"

training(data_train_batches, elapsed_train_batches, labels_train_batches, number_train_batches, learning_rate, training_epochs, dropout_prob, hidden_dim, fc_dim, 1, model_path)

testing(data_test_batches, elapsed_test_batches, labels_test_batches, number_test_batches, hidden_dim, fc_dim, 1, model_path)