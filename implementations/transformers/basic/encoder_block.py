#  FlowTransformer 2023 by liamdm / liam@riftcs.com

import warnings
try:
    from tensorflow._api.v2.v2 import keras
except ImportError:
    from tensorflow import keras
import tensorflow as tf
import keras.layers as layers
from keras.layers import Dense, Conv1D

class GPT3Attention(layers.Layer):
    def __init__(self, input_dimension, inner_dimension, n_heads, dropout_rate=0.1, use_conv=True, prefix: str = "", attn_implementation: str="vanilla"):
            super().__init__(name=prefix + "transformer_encoder")
            self.input_dimension = input_dimension
            self.inner_dimension = inner_dimension
            self.n_heads = n_heads
            self.dropout_rate = dropout_rate
            self.use_conv = use_conv
            self.attn_implementation = attn_implementation

            # The rest of the layers should be defined here
            self.attention = MultiHeadAttention(self.n_heads, self.input_dimension, implementation=self.attn_implementation, prefix=self.name + "_attn_")
            self.dropout_1 = Dropout(self.dropout_rate)
            self.add_1 = Add()
            self.layer_norm_1 = LayerNormalization()

            if self.use_conv:
                self.conv1 = Conv1D(filters=self.inner_dimension, kernel_size=1, activation="relu")
                self.conv2 = Conv1D(filters=self.input_dimension, kernel_size=1)
            else:
                self.dense1 = Dense(self.inner_dimension, activation="relu")
                self.dense2 = Dense(self.input_dimension)

            self.dropout_2 = Dropout(self.dropout_rate)
            self.add_2 = Add()
            self.layer_norm_2 = LayerNormalization()
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # noinspection PyMethodOverriding
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled Dot-Product Attention
        scaled_attention_logits = tf.matmul(q, k, transpose_b=True)
        scaled_attention_logits = scaled_attention_logits / tf.math.sqrt(tf.cast(self.depth, tf.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights)

        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))

        output = self.dense(output)
        output = self.dropout(output)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_heads": self.n_heads,
            "d_model": self.d_model,
            "dropout_rate": self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class MultiHeadAttentionImplementation:
    Keras = 0
    GPT3 = 1

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, **kwargs):
        # Pop custom arguments from kwargs, providing defaults for the problematic first block
        input_dimension = kwargs.pop("input_dimension", 289)
        inner_dimension = kwargs.pop("inner_dimension", 128)
        num_heads = kwargs.pop("num_heads", 2)
        dropout_rate = kwargs.pop("dropout_rate", 0.1)
        use_conv = kwargs.pop("use_conv", False)
        attn_implementation = kwargs.pop("attn_implementation", MultiHeadAttentionImplementation.Keras)

        # The remaining kwargs (e.g., 'name', 'trainable') are passed to the parent constructor
        super().__init__(**kwargs)

        if inner_dimension < input_dimension:
            warnings.warn(f"Typically inner_dimension should be greater than or equal to the input_dimension!")

        # Store parameters as instance attributes for get_config()
        self.input_dimension = input_dimension
        self.inner_dimension = inner_dimension
        self.num_heads = num_heads
        self.use_conv = use_conv
        self.attn_implementation = attn_implementation
        self.dropout_rate = dropout_rate
        self.attention = \
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=inner_dimension) \
                if attn_implementation == MultiHeadAttentionImplementation.Keras else\
                GPT3Attention(num_heads, inner_dimension, dropout_rate=0.0)

        layer_norm = 1e-6

        self.attention_dropout = layers.Dropout(dropout_rate)
        self.attention_layer_norm = layers.LayerNormalization(epsilon=layer_norm)

        self.feed_forward_0 = Conv1D(filters=inner_dimension, kernel_size=1, activation="relu") \
            if use_conv else Dense(inner_dimension, activation="relu")
        self.feed_forward_1 = Conv1D(filters=input_dimension, kernel_size=1, activation="relu") \
            if use_conv else Dense(input_dimension, activation="relu")

        self.dense_1 = Conv1D(filters=inner_dimension, kernel_size=1, activation='relu') if use_conv else Dense(inner_dimension, activation='relu')
        self.dense_2 = Conv1D(filters=input_dimension, kernel_size=1) if use_conv else Dense(input_dimension)

        self.norm_1 = layers.LayerNormalization()
        self.norm_2 = layers.LayerNormalization()
        self.add_1 = layers.Add()
        self.add_2 = layers.Add()
        self.dropout_1 = layers.Dropout(dropout_rate)
        self.dropout_2 = layers.Dropout(dropout_rate)
        self.activation = layers.ReLU()

    # noinspection PyMethodOverriding
    def call(self, inputs, training=None):
        x = inputs
        x_att = self.attention(x, x)

        x = self.add_1([x, x_att])
        x = self.norm_1(x)

        x_dense = self.dense_1(x)
        x_dense = self.activation(x_dense)
        x_dense = self.dropout_1(x_dense, training=training)
        x_dense = self.dense_2(x_dense)
        x_dense = self.dropout_2(x_dense, training=training)

        x = self.add_2([x, x_dense])
        x = self.norm_2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dimension": self.input_dimension,
            "inner_dimension": self.inner_dimension,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "use_conv": self.use_conv,
            "attn_implementation": self.attn_implementation
        })
        return config

    @classmethod
    def from_config(cls, config):
        custom_config = {
            "input_dimension": config.pop("input_dimension"),
            "inner_dimension": config.pop("inner_dimension"),
            "num_heads": config.pop("num_heads"),
            "dropout_rate": config.pop("dropout_rate"),
            "use_conv": config.pop("use_conv"),
            "attn_implementation": config.pop("attn_implementation"),
        }
        return cls(**custom_config, **config)
