# units.py
import pickle
import numpy as np
import tensorflow as tf
import copy
from datetime import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences
tf.config.run_functions_eagerly(False)
def load_data(training_file, validation_file, testing_file):

    # Print unique lengths
    print("Unique lengths of sublists:", 
    )
    train = (pickle.load(open(training_file, 'rb')))
    validate = (pickle.load(open(validation_file, 'rb')))
    test = (pickle.load(open(testing_file, 'rb')))
    return train, validate, test
def cut_data(training_file, validation_file, testing_file):

    cut_ratio = 80
    train = list(pickle.load(open(training_file, 'rb')))
    validate = list(pickle.load(open(validation_file, 'rb')))
    test = list(pickle.load(open(testing_file, 'rb')))
    for dataset in [train, validate, test]:
        dataset[0] = dataset[0][0: len(dataset[0]) // cut_ratio]
        dataset[1] = dataset[1][0: len(dataset[1]) // cut_ratio]
        dataset[2] = dataset[2][0: len(dataset[2]) // cut_ratio]
        dataset[3] = dataset[3][0: len(dataset[3]) // cut_ratio]
    return train, validate, test

def pad_time(seq_time_step, pad_value=0, max_len=50):
    """Pad or truncate time sequences to max_len with padding value."""
    seq_time_step_padded = np.full((len(seq_time_step), max_len), pad_value, dtype=np.float32)

    for bid, seq in enumerate(seq_time_step):
        for pid, subseq in enumerate(seq):
            if pid >= max_len:
                break
            seq_time_step_padded[bid, pid] = _to_timestamp(subseq)

    return seq_time_step_padded

def pad_failure_bits(failure_bit_seq, pad_value=1, max_len=50):
    failure_bit_seq_padded = np.full((len(failure_bit_seq), max_len), pad_value, dtype=np.float32)

    for bid, seq in enumerate(failure_bit_seq):
        for pid, subseq in enumerate(seq):
            if pid >= max_len:
                break
            failure_bit_seq_padded[bid, pid] = subseq

    return failure_bit_seq_padded

def _to_timestamp(ts):
    """Convert a time string to seconds since epoch if needed."""
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").timestamp()
        except ValueError:
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            except Exception:
                pass
    # Fallback to zero if parsing fails
    return 0.0

def pad_matrix(event_seq_code, maxlen, n_event_code, maxcode=50, pad_token=None):
    """Pad or truncate each patient's event sequence to shape [batch, maxlen, maxcode]."""
    n_samples = len(event_seq_code)
    
    # Use separate padding token (n_event_code) vs EOS token (n_event_code - 1)
    if pad_token is None:
        pad_token = n_event_code  # 273041 for padding

    # Get max number of codes per visit across all patients (cap optional)
    # maxcode = max(
    #     max((len(visit) for visit in seq[:maxlen]), default=0) 
    #     for seq in event_seq_code
    # )

    batch_diagnosis_codes = np.full((n_samples, maxlen, maxcode), pad_token, dtype=np.int64)
    batch_mask = np.zeros((n_samples, maxlen), dtype=np.float32)
    batch_mask_code = np.zeros((n_samples, maxlen, maxcode), dtype=np.float32)
    batch_mask_final = np.zeros((n_samples, maxlen), dtype=np.float32)

    for bid, seq in enumerate(event_seq_code):
        for pid, subseq in enumerate(seq[:maxlen]):  # truncate if too long
            for tid, code in enumerate(subseq):
                if tid < maxcode:
                    batch_diagnosis_codes[bid, pid, tid] = code
                    batch_mask_code[bid, pid, tid] = 1.0

        seq_len = min(len(seq), maxlen)
        if seq_len > 1:
            batch_mask[bid, :seq_len - 1] = 1.0
            batch_mask_final[bid, seq_len - 1] = 1.0
        elif seq_len == 1:
            batch_mask_final[bid, 0] = 1.0  # if only one visit

    return batch_diagnosis_codes, batch_mask, batch_mask_final, batch_mask_code


def calculate_cost_tran(model, data, options, max_len, loss_function=tf.keras.losses.SparseCategoricalCrossentropy()):

    batch_size = options['batch_size']
    n_batches = int(np.ceil(float(len(data[0])) / float(batch_size)))
    cost_sum = 0.0

    for index in range(n_batches):
        event_seq_code = data[0][batch_size * index: batch_size * (index + 1)]
        batch_time_step = data[2][batch_size * index: batch_size * (index + 1)]
        event_seq_code, batch_time_step = adjust_input(event_seq_code, batch_time_step, max_len, options['n_event_code'])
        batch_labels = data[1][batch_size * index: batch_size * (index + 1)]
        lengths = np.array([len(seq) for seq in event_seq_code])
        maxlen = np.max(lengths)

        event_seq_code = pad_sequences(event_seq_code, maxlen=max_len, padding='post', value=0)
        batch_time_step = pad_sequences(batch_time_step, maxlen=max_len, padding='post', value=0)

        event_seq_code = tf.convert_to_tensor(event_seq_code, dtype=tf.int32)
        batch_time_step = tf.convert_to_tensor(batch_time_step, dtype=tf.float32)
        batch_labels = tf.convert_to_tensor(batch_labels, dtype=tf.int32)

        logit, labels, self_attention = model(event_seq_code, batch_time_step, batch_labels, options, maxlen, training=False)

        loss = loss_function(logit,labels)
        cost_sum += loss.numpy()  # Directly get the numpy value


    return cost_sum / n_batches



def adjust_input(event_seq_code, batch_time_step,event_failure_sys,event_failure_user, max_len, n_event_code):
    '''
    Appending EOS token (n_event_code-1) and last timestep (0)
    EOS token: n_event_code-1 (273040 - no_cat value)
    PAD token: n_event_code (273041 - for padding sequences)
    '''
    batch_time_step = copy.deepcopy(batch_time_step)
    event_seq_code = copy.deepcopy(event_seq_code)
    event_failure_sys = copy.deepcopy(event_failure_sys)
    event_failure_user = copy.deepcopy(event_failure_user)
    
    # EOS token is n_event_code - 1 (273040), PAD token is n_event_code (273041)
    eos_token = n_event_code - 1

    for ind in range(len(event_seq_code)):
        if len(event_seq_code[ind]) > max_len:
            event_seq_code[ind] = event_seq_code[ind][-(max_len):]
            batch_time_step[ind] = batch_time_step[ind][-(max_len):]
            event_failure_sys[ind] = event_failure_sys[ind][-(max_len):]
            event_failure_user[ind] = event_failure_user[ind][-(max_len):]

        batch_time_step[ind].append(0) # should append greater value
        event_seq_code[ind].append([eos_token])  # Use EOS token (273040), not PAD token
        event_failure_sys[ind].append(1)
        event_failure_user[ind].append(1)

    return event_seq_code, batch_time_step , event_failure_sys, event_failure_user
