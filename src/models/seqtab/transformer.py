# transformer.py
import math
import tensorflow as tf
import numpy as np

import tensorflow as tf
from tensorflow.keras import regularizers

class Embedding(tf.keras.layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.embeddings = self.add_weight(shape=(self.num_embeddings, self.embedding_dim),
                                          initializer=tf.keras.initializers.he_uniform(),
                                          trainable=True)

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.embeddings, inputs)


class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, attention_dropout=0.0, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(attention_dropout)
        self.softmax = tf.keras.layers.Softmax(axis=2)

    def call(self, q, k, v, scale=None, attn_mask=None,training=False):
        attention = tf.linalg.matmul(q, k, transpose_b=True)
        if scale is not None:
            attention *= scale
        if attn_mask is not None:
            tf.where(attn_mask, -np.inf, attention)
            # attention += (attn_mask * -1e9)
        attention = self.softmax(attention)
        attention = self.dropout(attention,training= training)
        context = tf.linalg.matmul(attention, v)
        return context, attention



class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_seq_len, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        # Precomputed positional encodings
        position_encoding = np.array([
            [pos / np.power(10000, (2 * (j // 2)) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])  # sine for even indices
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])  # cosine for odd indices
        position_encoding = np.concatenate((np.zeros([1, d_model]), position_encoding), axis=0)
        self.position_encoding = tf.constant(position_encoding, dtype=tf.float32)

    def call(self, input_len):
        """Return positional encodings up to the maximum sequence length.

        The incoming ``input_len`` tensor contains the real lengths for each
        sequence in the batch.  We always broadcast the positional encodings to
        ``max_seq_len`` (the size used when the layer was constructed) so that
        the returned tensor can be safely added to padded sequence inputs.  A
        mask based on ``input_len`` ensures padded positions remain zero.
        """

        # Ensure ``input_len`` is an int32 tensor
        input_len = tf.cast(input_len, dtype=tf.int32)

        # ``self.position_encoding`` includes an extra row at index 0 for padding
        max_len = tf.shape(self.position_encoding)[0] - 1

        # Generate positional indices for the full maximum length
        input_pos = tf.range(start=1, limit=max_len + 1, delta=1)
        input_pos = tf.broadcast_to(input_pos, [tf.shape(input_len)[0], max_len])

        # Mask out positions beyond each sequence's actual length
        mask = tf.sequence_mask(lengths=input_len, maxlen=max_len)
        input_pos = tf.where(mask, input_pos, 0)

        # Gather the corresponding encodings
        positions = tf.gather(self.position_encoding, input_pos)

        return positions, input_pos


class PositionalWiseFeedForward_old(tf.keras.layers.Layer):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0, **kwargs): # 256, 1024, 0.0
        super(PositionalWiseFeedForward, self).__init__(**kwargs)
        self.w1 = tf.keras.layers.Conv1D(filters= ffn_dim, input_shape=(50, model_dim,50) , kernel_size=1)
        self.w2 = tf.keras.layers.Conv1D(filters= 50, input_shape=(50, ffn_dim,50)  ,kernel_size=1)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        output = tf.transpose(x, [0, 2, 1]) # 50, 256, 50
        output = self.w2(tf.nn.relu(self.w1(output)))
        output = self.dropout(tf.transpose(output, [0, 2, 1]))
        output = self.layer_norm(x + output)
        return output
    
        output = tf.transpose(x, [0, 2, 1])  # Transpose to shape [50, 256, 50]

        # Apply a two-layer network; make sure the final layer outputs the same last dimension as the input
        output = self.w2(tf.nn.relu(self.w1(output)))  # Shape should now be [50, 256, 50]

        output = self.dropout(tf.transpose(output, [0, 2, 1]))  # Transpose back to [50, 50, 256]

        output = self.layer_norm(x + output) 

class ScaledDotProductAttention_2(tf.keras.layers.Layer):
    def __init__(self, attention_dropout=0.0, **kwargs):
        super(ScaledDotProductAttention_2, self).__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(attention_dropout)
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, q, k, v, scale=None, attn_mask=None, training=False):
        attention = tf.linalg.matmul(q, k, transpose_b=True)
        if scale:
            attention *= scale
        if attn_mask is not None:
            # Ensure the mask is a boolean tensor, then apply -inf to masked positions.
            attention += (tf.cast(attn_mask, tf.float32) * -1e9)
        attention = self.softmax(attention)
        attention = self.dropout(attention, training=training)
        context = tf.linalg.matmul(attention, v)
        return context, attention
    
class PositionalWiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0, **kwargs):
        super(PositionalWiseFeedForward, self).__init__(**kwargs)
        # First fully connected layer
        self.dense_1 = tf.keras.layers.Dense(ffn_dim, activation='relu')
        # Second fully connected layer
        self.dense_2 = tf.keras.layers.Dense(model_dim)
        # Dropout layer
        self.dropout = tf.keras.layers.Dropout(dropout)
        # Layer normalization
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        # Applying dense layers with dropout in between
        output = self.dense_1(x)
        output = self.dense_2(output)
        output = self.dropout(output)
        
        # Adding residual connection and applying layer normalization
        output = self.layer_norm(x + output)
        return output

class MultiHeadAttention_2(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, dropout=0.0, l2_lambda=0.01):
        super(MultiHeadAttention_2, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.dim_per_head = model_dim // num_heads

        # Define layers for linear transformations to Query, Key, and Value
        self.linear_k = tf.keras.layers.Dense(model_dim)
        self.linear_v = tf.keras.layers.Dense(model_dim)
        self.linear_q = tf.keras.layers.Dense(model_dim)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = tf.keras.layers.Dense(model_dim, kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, key, value, query, attn_mask=None, training=False):
        batch_size = tf.shape(query)[0]

        # Apply linear transformation and split into num_heads
        key = tf.reshape(self.linear_k(key), [batch_size, -1, self.num_heads, self.dim_per_head])
        value = tf.reshape(self.linear_v(value), [batch_size, -1, self.num_heads, self.dim_per_head])
        query = tf.reshape(self.linear_q(query), [batch_size, -1, self.num_heads, self.dim_per_head])

        # Transpose for attention dot product: [batch_size, num_heads, seq_len, depth]
        key = tf.transpose(key, perm=[0, 2, 1, 3])
        value = tf.transpose(value, perm=[0, 2, 1, 3])
        query = tf.transpose(query, perm=[0, 2, 1, 3])

        # Call ScaledDotProductAttention
        context, attention = self.dot_product_attention(query, key, value, attn_mask, training=training)

        # Transpose back and reshape
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, [batch_size, -1, self.model_dim])

        # Apply final linear layer and dropout
        output = self.linear_final(context)
        output = self.dropout(output, training=training)

        # Add and normalize
        output = self.layer_norm(query + output)

        return output, attention


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0,l2_lambda =0.01):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = tf.keras.layers.Dense(self.dim_per_head * num_heads)
        self.linear_v = tf.keras.layers.Dense(self.dim_per_head * num_heads)
        self.linear_q = tf.keras.layers.Dense(self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = tf.keras.layers.Dense(model_dim, kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, key, value, query, attn_mask=None,training = False):
        residual = query

        batch_size = tf.shape(key)[0]

        # Linear projection
        key = self.linear_k(key) # 207, 96, 256
        value = self.linear_v(value) # 207, 96, 256
        query = self.linear_q(query) # 207, 96, 256

        # Split by heads
        key = tf.reshape(key, [batch_size * self.num_heads, -1 , self.dim_per_head]) # 828, 96, 64
        value = tf.reshape(value, [batch_size * self.num_heads, -1 , self.dim_per_head]) # 828, 96, 64
        query = tf.reshape(query, [batch_size * self.num_heads, -1 , self.dim_per_head]) # 828, 96, 64



        if attn_mask is not None:
            attn_mask = tf.repeat(attn_mask, self.num_heads, axis=0)

        # Scaled Dot Product Attention
        # ``key`` has already been reshaped so that its last dimension is
        # ``dim_per_head``.  Using integer division here can yield zero when the
        # model dimension is small, causing ``0.0**-0.5`` errors.  Instead, use
        # the known head dimension directly.
        scale = tf.cast(self.dim_per_head, tf.float32) ** -0.5
        context, attention = self.dot_product_attention(query, key, value, scale, attn_mask,training = training)

        # Concat heads
        context = tf.reshape(context, [batch_size, -1, self.dim_per_head * self.num_heads])

        # Final linear projection
        output = self.linear_final(context)

        # Dropout
        output = self.dropout(output)

        # Add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention

class TimeEncoder(tf.keras.Model):
    """Time Encoder module: Globle level Comprehensive Analysis
    input: seq_time_step
    input: final_queries (equation 5) q = ReLU(Wqh∗ + bq), query vector
    output: time_weight
    
    """
    def __init__(self, batch_size,l2_lambda=0.01):
        super(TimeEncoder, self).__init__()
        self.batch_size = batch_size
        self.selection_layer = tf.keras.layers.Dense(64, kernel_regularizer=regularizers.l2(l2_lambda))
        self.weight_layer = tf.keras.layers.Dense(64, kernel_regularizer=regularizers.l2(l2_lambda))

    def call(self, seq_time_step, final_queries , mask):
        # Processing the sequence time steps
        seq_time_step = tf.cast(tf.expand_dims(seq_time_step, axis=-1), dtype=tf.float32) / 7200

        # Applying the selection layer and the activation functions
        selection_feature = 1 - tf.math.tanh(tf.pow(self.selection_layer(seq_time_step), 2))  # ot equation 6 part 1, selection layer multiplies (δt/180) by Wo
        selection_feature = tf.keras.activations.relu(self.weight_layer(selection_feature)) # Kt equation 6 part 2, weight_layer layer multiplies (ot) by Wk

        input_dim = 8 # change according to the input dimension
        # Applying the weights to the final queries
        selection_feature = tf.reduce_sum(selection_feature * final_queries, axis=2, keepdims=True) / input_dim # equation 7, q⊤*kt/√s. qt was input

        # Applying the mask
        expanded_mask = tf.cast(mask,tf.bool)
        # print(mask.shape)
        selection_feature = tf.where(expanded_mask, -np.inf, selection_feature)
        # selection_feature = tf.where(mask, selection_feature, tf.fill(tf.shape(selection_feature), -np.inf))

        # Applying softmax
        return tf.nn.softmax(selection_feature, axis=1)  # equation 8, β = Softmax(ϕ), Globle attention weight
def padding_mask(seq_k, seq_q):
    # Get the length of the sequences
    len_q = tf.shape(seq_q)[1]

    # Create a mask for padding values
    pad_mask = tf.equal(seq_k, 0)

    # Expand dimensions to add the required axis for broadcasting
    pad_mask = tf.expand_dims(pad_mask, 1)

    # Leverage broadcasting instead of manual expansion
    pad_mask = tf.broadcast_to(pad_mask, [tf.shape(pad_mask)[0], len_q, tf.shape(pad_mask)[2]])

    return pad_mask

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2018, dropout=0.0,l2_lambda=0.01):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout, l2_lambda=l2_lambda)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def call(self, inputs, attn_mask=None,training = False):
        # Self attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask,training = training)

        # Feed forward network
        output = self.feed_forward(context)

        return output, attention
class EncoderNew(tf.keras.Model):
    def __init__(self, vocab_size, max_seq_len, num_layers=1, model_dim=256, num_heads=4, ffn_dim=1024, dropout=0.0, is_public_dataset=False, l2_lambda=0.01):
        super(EncoderNew, self).__init__()
        self.is_public_dataset = is_public_dataset
        # Encoder Layers with L2 regularization
        self.encoder_layers = [
            EncoderLayer(model_dim + 2, num_heads, ffn_dim, dropout, l2_lambda) 
            for _ in range(num_layers)
        ] if not is_public_dataset else [
            EncoderLayer(model_dim, num_heads, ffn_dim, dropout, l2_lambda) 
            for _ in range(num_layers)
        ]

        # Pre Embedding with L2 regularization
        self.pre_embedding = tf.keras.layers.Embedding(vocab_size, model_dim, embeddings_regularizer=regularizers.l2(l2_lambda))

        # Bias Embedding remains unchanged as it doesn't typically require regularization
        bound = 1 / math.sqrt(vocab_size)

        self.bias_embedding = tf.Variable(
            tf.random.uniform(shape=[model_dim], minval=-bound, maxval=bound),
            trainable=True
        ) if not is_public_dataset else  tf.Variable(
            tf.random.uniform(shape=[model_dim], minval=-bound, maxval=bound),
            trainable=True
        )
        # Positional Encoding remains unchanged
        self.pos_embedding = PositionalEncoding(model_dim + 2, max_seq_len) if not is_public_dataset else PositionalEncoding(model_dim, max_seq_len)

        # Time Layer with L2 regularization
        self.time_layer = tf.keras.layers.Dense(model_dim + 2, kernel_regularizer=regularizers.l2(l2_lambda)) if not is_public_dataset else tf.keras.layers.Dense(model_dim, kernel_regularizer=regularizers.l2(l2_lambda))

        # Selection Layer with L2 regularization
        self.selection_layer = tf.keras.layers.Dense(64, kernel_regularizer=regularizers.l2(l2_lambda))


        # Activation Functions
        self.relu = tf.keras.activations.relu
        self.tanh = tf.keras.activations.tanh

    def call(self, event_code, mask, mask_code, seq_time_step,event_failure_sys, event_failure_user, input_len ,training = False):
        # Time Feature
        
        seq_time_step = tf.expand_dims(tf.convert_to_tensor(seq_time_step, dtype=tf.float32), -1) / 7200
        time_feature = 1 - self.tanh(tf.pow(tf.cast(self.selection_layer(seq_time_step),tf.float32), 2))
        time_feature = self.time_layer(time_feature)

        event_failure_sys = tf.expand_dims(event_failure_sys, axis=-1)
        event_failure_sys = tf.expand_dims(event_failure_sys, axis=-1)
        event_failure_sys = tf.broadcast_to(event_failure_sys,
                                            tf.concat([tf.shape(event_code), [1]], axis=0))

        event_failure_user = tf.expand_dims(event_failure_user, axis=-1)
        event_failure_user = tf.expand_dims(event_failure_user, axis=-1)
        event_failure_user = tf.broadcast_to(event_failure_user,
                                             tf.concat([tf.shape(event_code), [1]], axis=0))
        

        # Output Embeddingtf.cast(mask, tf.bool))
        act_emb = tf.cast(self.pre_embedding(tf.cast(event_code,tf.int32)), tf.float32)
        if not self.is_public_dataset:
            # event_failure_sys_broadcasted = tf.broadcast_to(event_failure_sys, act_emb.shape[:-1] + [1])
            # event_failure_user_broadcasted = tf.broadcast_to(event_failure_user, act_emb.shape[:-1] + [1])
            fail_bit_emb = tf.concat([act_emb, event_failure_sys,event_failure_user ], axis=3)
            output = fail_bit_emb  *  tf.cast(mask_code, tf.float32) 
        else:
            # event_failure_sys_broadcasted = tf.broadcast_to(event_failure_sys, act_emb.shape[:-1] + [1])
            # event_failure_sys_broadcasted = tf.broadcast_to(event_failure_sys, act_emb.shape[:-1] + [1])
            # event_failure_user_broadcasted = tf.broadcast_to(event_failure_user, act_emb.shape[:-1] + [1])
            # fail_bit_emb = tf.concat([act_emb, event_failure_user_broadcasted ], axis=3)
            output = act_emb  *  tf.cast(mask_code, tf.float32)
        output = tf.reduce_sum(output, axis=2)  + self.bias_embedding
        output += time_feature

        # Positional Encoding
        output_pos, ind_pos = self.pos_embedding(input_len)
        
        output += output_pos

        # Padding Mask
        self_attention_mask = padding_mask(ind_pos, ind_pos)

        attentions = []
        outputs = []
        # Encoder Layers
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask,training = training)
            attentions.append(attention)
            outputs.append(output)

        return output


class DecoderNew(tf.keras.Model):
    def __init__(self, num_layers,vocab_size, model_dim, num_heads, ffn_dim, dropout=0.0, l2_lambda=0.01, max_seq_len=100):
        super(DecoderNew, self).__init__()
        self.decoder_layers = [
            DecoderLayer(model_dim, num_heads, ffn_dim, dropout, l2_lambda)
            for _ in range(num_layers)
        ]
        self.final_layer = tf.keras.layers.Dense(model_dim)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        self.pre_embedding = tf.keras.layers.Embedding(vocab_size, model_dim, embeddings_regularizer=regularizers.l2(l2_lambda))

    def call(self, x, enc_output,mask_code, input_len, training=False):
        # Calculate positional encodings
        act_emb = tf.cast(self.pre_embedding(tf.cast(tf.expand_dims(x,-1),tf.int32)), tf.float32)
        output = act_emb *  tf.cast(mask_code, tf.float32)
        output = tf.reduce_sum(output, axis=2)  
        positional_encodings, ind_pos = self.pos_embedding(input_len)
        output += positional_encodings

        # Calculate look-ahead mask based on input lengths
        look_ahead_mask = create_look_ahead_mask(tf.shape(x)[1])
        seq_padding_mask = padding_mask(ind_pos, ind_pos)  # Renamed to seq_padding_mask to avoid confusion

        attention_weights = {}
        for i, layer in enumerate(self.decoder_layers):
            x, block1, block2 = layer(x, enc_output, look_ahead_mask, seq_padding_mask, training=training)
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        final_output = self.final_layer(x)

        return final_output
    
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def create_padding_mask(seq, input_len):
    # Assume seq is the tensor and input_len are the actual lengths of the sequences
    mask = tf.sequence_mask(input_len, maxlen=tf.shape(seq)[1])
    return 1 - tf.cast(mask, dtype=tf.float32)  # Convert to float and invert
# import tensorflow as tf

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, ffn_dim, dropout_rate, l2_lambda=0.01):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(model_dim, num_heads, dropout_rate, l2_lambda=l2_lambda)
        self.enc_dec_attention = MultiHeadAttention(model_dim, num_heads, dropout_rate, l2_lambda=l2_lambda)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout_rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, look_ahead_mask, padding_mask, training):
        # Self attention
        combined_mask = tf.maximum(padding_mask, look_ahead_mask)
        attn1, attn_weights_block1 = self.self_attention(x, x, x, combined_mask, training=training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # Encoder-Decoder attention
        attn2, attn_weights_block2 = self.enc_dec_attention(out1, enc_output, enc_output, padding_mask, training=training)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        # Feed forward
        ffn_output = self.feed_forward(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2

class Decoder(tf.keras.Model):
    def __init__(self, num_layers, model_dim, num_heads, ffn_dim, vocab_size, max_seq_len, dropout_rate, l2_lambda=0.01):
        super(Decoder, self).__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_dim)
        self.pos_encoding = PositionalEncoding(model_dim, max_seq_len)

        self.dec_layers = [DecoderLayer(model_dim, num_heads, ffn_dim, dropout_rate, l2_lambda=l2_lambda) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, padding_mask,input_len, training):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x) # (256, 77, 1, 64)
        x *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        positional_encodings, ind_pos = self.pos_embedding(input_len)
        x += positional_encodings
        look_ahead_mask = create_look_ahead_mask(tf.shape(x)[1])
        seq_padding_mask = padding_mask(ind_pos, ind_pos)  # Renamed to seq_padding_mask to avoid confusion
        x = self.dropout(x, training=training)

        for i, dec_layer in enumerate(self.dec_layers):
            x, block1, block2 = dec_layer(x, enc_output, look_ahead_mask, seq_padding_mask, training)

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        return x, attention_weights


class TransformerTime(tf.keras.Model):
    def __init__(self, n_event_code, batch_size,max_len, layer, dropout_rate, is_public_dataset=False, l2_lambda=0.01,model_dim = 256,name=None):
        super(TransformerTime, self).__init__(name=name)
        self.n_event_code = n_event_code
        self.time_encoder = TimeEncoder(batch_size,l2_lambda=l2_lambda)
        # ``pad_matrix`` already pads sequences to ``max_len``. The +1 used
        # during data preparation only affects the unpadded lengths, so the
        # encoder should expect inputs of length ``max_len``.
        self.feature_encoder = EncoderNew(
            n_event_code + 1,
            max_len,
            num_layers=layer,
            model_dim=model_dim,
            is_public_dataset=is_public_dataset,
            dropout=dropout_rate,
            l2_lambda=l2_lambda,
        )
        self.self_layer = tf.keras.layers.Dense(1, kernel_regularizer=regularizers.l2(l2_lambda))
        self.quiry_layer = tf.keras.layers.Dense(64, kernel_regularizer=regularizers.l2(l2_lambda))
        self.quiry_weight_layer = tf.keras.layers.Dense(2, kernel_regularizer=regularizers.l2(l2_lambda))
        self.relu = tf.keras.activations.relu
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.batch_size = batch_size

    def get_self_attention(self, features, query, mask):
        attention_logits = self.self_layer(features)
        expanded_mask = tf.cast(mask, tf.bool)
        attention_logits = tf.where(expanded_mask, -np.inf, attention_logits)
        attention = tf.nn.softmax(attention_logits, axis=1)
        return attention
    
    def remove_padding(self, padded_sequences, original_lengths):
        removed_padded_sequences = []
        for i in range(len(padded_sequences)):
            if len(padded_sequences[i]) > int(original_lengths[i].numpy()):
                removed_padded_sequences.append(padded_sequences[i][:int(original_lengths[i].numpy())].numpy())
            else:
                removed_padded_sequences.append(padded_sequences[i].numpy())

        return removed_padded_sequences
    def call(self, seq_input , training = False):
        # Unpack inputs
        event_code, seq_time_step,event_failure_sys, event_failure_user,  mask, mask_final, mask_code,lengths = seq_input


        # Convert to TensorFlow tensors
        event_code = tf.convert_to_tensor(event_code)
        mask_mult = tf.expand_dims(tf.cast(tf.math.logical_not(tf.cast(mask, tf.bool)), tf.float32), axis=2)

        mask_final = tf.expand_dims(mask_final, -1) # shape (50,62,130) one for end of sequence, padded values are zero, all other values are zero
        mask_code = tf.expand_dims(mask_code, -1)

        # Forward pass through the feature encoder
        features = self.feature_encoder(event_code, mask_mult, mask_code, seq_time_step,event_failure_sys, event_failure_user, lengths,training =training) # shape (50,62,130) = (batch, seq_len, model_dim)

        # Compute final statuses
        final_statuses = tf.cast(features,tf.float32) * tf.cast(mask_final,tf.float32) # mask will put zero every where except for special event e*
        final_statuses = tf.reduce_sum(final_statuses, axis=1, keepdims=True) ## basically getting feature embedding for the special event e*

        # Queries and weights
        queries = self.relu(self.quiry_layer(final_statuses))

        # Self-attention weights
        self_attention = self.get_self_attention(features, queries, mask_mult)

        # Time weights # betas
        time_weight = self.time_encoder(seq_time_step, queries, mask_mult)

        # Attention weights # z equation 9
        attention_weight = tf.nn.softmax(self.quiry_weight_layer(final_statuses), axis=2)

        # Total weight calculation
        total_weight = tf.concat([time_weight, self_attention], axis=2)
        total_weight = tf.reduce_sum(total_weight * attention_weight, axis=2, keepdims=True)
        total_weight = total_weight / (tf.reduce_sum(total_weight, axis=1, keepdims=True) + 1e-5)

        # final_output = self.decoder( tf.cast(tf.squeeze(event_code),dtype=tf.int32), features,mask_code,  lengths, training=True)
        # Weighted features and dropout
        weighted_features = features * total_weight
        averaged_features = tf.reduce_sum(weighted_features, axis=1)
        averaged_features = self.dropout(averaged_features,training = training)
#
        return averaged_features

