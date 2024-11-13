import pickle
import numpy as np
import tensorflow as tf
import copy
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

def pad_time(seq_time_step, pad_value=0):
    '''
    Pad time with 10000 to sequences less than max length
    :param seq_time_step: list of list of time step
    :param options: options
    :return: padded time step
    '''
    # Calculate the maximum length of sequences
    maxlen = max(len(seq) for seq in seq_time_step)
    seq_time_step_padded = np.zeros((len(seq_time_step), maxlen), dtype=np.float32) + pad_value
    # Pad each sequence to the maximum length



    for bid, seq in enumerate(seq_time_step):
        for pid, subseq in enumerate(seq):
                seq_time_step_padded[bid, pid] = subseq # putting actual codes
   
    return seq_time_step_padded

def pad_failure_bits(failure_bit_seq, pad_value=1):
    '''
    Pad time with 10000 to sequences less than max length
    :param seq_time_step: list of list of time step
    :param options: options
    :return: padded time step
    '''
    # Calculate the maximum length of sequences
    maxlen = max(len(seq) for seq in failure_bit_seq)
    failure_bit_seq_padded = np.zeros((len(failure_bit_seq), maxlen), dtype=np.float32) + pad_value
    # Pad each sequence to the maximum length



    for bid, seq in enumerate(failure_bit_seq):
        for pid, subseq in enumerate(seq):
                failure_bit_seq_padded[bid, pid] = subseq # putting actual codes
   
    return failure_bit_seq_padded

def pad_matrix_new(event_seq_code, maxlen,n_event_code):
    lengths = np.array([len(seq) for seq in event_seq_code]) 
    n_samples = len(event_seq_code)
    maxlen = np.max(lengths) 
    lengths_code = []
    for seq in event_seq_code:
        for code_set in seq: # each visit in sequence( have multiple codes in one each vist)
            lengths_code.append(len(code_set))
    lengths_code = np.array(lengths_code) # codes in each visit for all patient
    maxcode = np.max(lengths_code) # 

    batch_diagnosis_codes = np.zeros((n_samples, maxlen, maxcode), dtype=np.int64) + n_event_code
    batch_mask = np.zeros((n_samples, maxlen), dtype=np.float32)
    batch_mask_code = np.zeros((n_samples, maxlen, maxcode), dtype=np.float32)
    batch_mask_final = np.zeros((n_samples, maxlen), dtype=np.float32)

    for bid, seq in enumerate(event_seq_code):
        for pid, subseq in enumerate(seq):
            for tid, code in enumerate(subseq):
                batch_diagnosis_codes[bid, pid, tid] = code # putting actual codes
                batch_mask_code[bid, pid, tid] = 1 # putting 1 where actual value exist

    for i in range(n_samples):
        batch_mask[i, 0:lengths[i]-1] = 1 # putting 1 every where we have actual visit
        max_visit = lengths[i] - 1
        batch_mask_final[i, max_visit] = 1

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
    Appending last event code(code not present for any event ) and last timestep (0)
    '''
    batch_time_step = copy.deepcopy(batch_time_step)
    event_seq_code = copy.deepcopy(event_seq_code)
    event_failure_sys = copy.deepcopy(event_failure_sys)
    event_failure_user = copy.deepcopy(event_failure_user)

    for ind in range(len(event_seq_code)):
        if len(event_seq_code[ind]) > max_len:
            event_seq_code[ind] = event_seq_code[ind][-(max_len):]
            batch_time_step[ind] = batch_time_step[ind][-(max_len):]
            event_failure_sys[ind] = event_failure_sys[ind][-(max_len):]
            event_failure_user[ind] = event_failure_user[ind][-(max_len):]

        batch_time_step[ind].append(0) # should append greater value
        event_seq_code[ind].append([n_event_code-1])
        event_failure_sys[ind].append(1)
        event_failure_user[ind].append(1)

    return event_seq_code, batch_time_step , event_failure_sys, event_failure_user
