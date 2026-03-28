import tensorflow as tf
from tensorflow.keras import layers, models

from config import (
    BILSTM_UNITS_1,
    BILSTM_UNITS_2,
    BILSTM_DROPOUT,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_NUM_HEADS,
    TRANSFORMER_FF_DIM,
    TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_DROPOUT,
)


@tf.keras.utils.register_keras_serializable()
class ReduceMeanLayer(layers.Layer):
    def __init__(self, axis, keepdims=False, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=self.axis, keepdims=self.keepdims)

    def get_config(self):
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "keepdims": self.keepdims,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class ReduceSumLayer(layers.Layer):
    def __init__(self, axis, keepdims=False, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis, keepdims=self.keepdims)

    def get_config(self):
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "keepdims": self.keepdims,
        })
        return config
# =========================================================
# Common Attention Layer for BiLSTM
# ========================================================
#.utils.register_keras_serializable()
@tf.keras.utils.register_keras_serializable()
class TemporalAttention(layers.Layer):
    def __init__(self, attn_units=64, **kwargs):
        super().__init__(**kwargs)
        self.attn_units = attn_units

    def build(self, input_shape):
        hidden_dim = input_shape[-1]

        self.attn_dense = layers.Dense(self.attn_units, activation="tanh")
        self.score_dense = layers.Dense(1)

        super().build(input_shape)

    def call(self, inputs):
        scores = self.score_dense(self.attn_dense(inputs))   # (B, T, 1)
        weights = tf.nn.softmax(scores, axis=1)
        context = tf.reduce_sum(weights * inputs, axis=1)
        return context, weights

#class TemporalAttention(layers.Layer):
 #   def __init__(self, attn_units=64, **kwargs):
 #       super().__init__(**kwargs)
 #       self.attn_dense = layers.Dense(attn_units, activation="tanh")
 #       self.score_dense = layers.Dense(1)
 #
 #   def call(self, inputs):
 #       """
 #       inputs: (B, T, H)
 #       returns:
 #           context: (B, H)
 #           weights: (B, T, 1)
 #       """
 #       scores = self.score_dense(self.attn_dense(inputs))   # (B, T, 1)
 #       weights = tf.nn.softmax(scores, axis=1)              # (B, T, 1)
 #       context = tf.reduce_sum(weights * inputs, axis=1)    # (B, H)
 #       return context, weights


# =========================================================
# Positional Encoding for Transformer
# =========================================================
def positional_encoding(length, d_model):
    pos = tf.range(length, dtype=tf.float32)[:, tf.newaxis]      # (T, 1)
    i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]       # (1, D)

    angle_rates = 1.0 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    angle_rads = pos * angle_rates

    sines = tf.sin(angle_rads[:, 0::2])
    cosines = tf.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)          # (T, D)
    pos_encoding = pos_encoding[tf.newaxis, ...]                 # (1, T, D)

    return tf.cast(pos_encoding, tf.float32)


# =========================================================
# Transformer Encoder Block
# ========================================================







@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
        )

        self.dropout1 = layers.Dropout(dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        self.ffn_dense1 = layers.Dense(ff_dim, activation="relu")
        self.ffn_dropout = layers.Dropout(dropout)
        self.ffn_dense2 = layers.Dense(d_model)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.mha.build(input_shape, input_shape)
        self.dropout1.build(input_shape)
        self.norm1.build(input_shape)

        self.ffn_dense1.build(input_shape)
        ffn_hidden_shape = list(input_shape)
        ffn_hidden_shape[-1] = self.ff_dim
        self.ffn_dropout.build(tuple(ffn_hidden_shape))
        self.ffn_dense2.build(tuple(ffn_hidden_shape))
        self.norm2.build(input_shape)

        super().build(input_shape)

    def call(self, x, training=False, return_attention=False):
        attn_output, attn_scores = self.mha(
            x,
            x,
            return_attention_scores=True,
            training=training
        )

        x1 = self.norm1(x + self.dropout1(attn_output, training=training))

        ff = self.ffn_dense1(x1)
        ff = self.ffn_dropout(ff, training=training)
        ff = self.ffn_dense2(ff)

        out = self.norm2(x1 + ff)

        if return_attention:
            return out, attn_scores
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
        })
        return config
# =========================================================
# Plain BiLSTM Baseline
# =========================================================
def build_bilstm_model(input_shape=(128, 3)):
    inputs = layers.Input(shape=input_shape, name="input")

    x = layers.Bidirectional(
        layers.LSTM(BILSTM_UNITS_1, return_sequences=True),
        name="bilstm_1"
    )(inputs)
    x = layers.Dropout(BILSTM_DROPOUT, name="dropout_1")(x)

    x = layers.Bidirectional(
        layers.LSTM(BILSTM_UNITS_2, return_sequences=False),
        name="bilstm_2"
    )(x)
    x = layers.Dropout(BILSTM_DROPOUT, name="dropout_2")(x)

    x = layers.Dense(32, activation="relu", name="dense_1")(x)
    x = layers.Dropout(BILSTM_DROPOUT, name="dropout_3")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="BiLSTM_Baseline")
    return model


# =========================================================
# Plain Transformer Baseline
# =========================================================
def build_transformer_model(
    input_shape=(128, 3),
    d_model=TRANSFORMER_D_MODEL,
    num_heads=TRANSFORMER_NUM_HEADS,
    ff_dim=TRANSFORMER_FF_DIM,
    num_layers=TRANSFORMER_NUM_LAYERS,
    dropout=TRANSFORMER_DROPOUT,
):
    inputs = layers.Input(shape=input_shape, name="input")

    x = layers.Dense(d_model, name="input_projection")(inputs)
    x = x + positional_encoding(input_shape[0], d_model)

    for i in range(num_layers):
        block = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            name=f"transformer_block_{i+1}"
        )
        x = block(x)

    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    x = layers.Dense(64, activation="relu", name="dense_1")(x)
    x = layers.Dropout(0.3, name="dropout_final")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Transformer_Baseline")
    return model


# =========================================================
# Attention BiLSTM Supervised Model
# Used for pretrained RL later
# =========================================================
def build_attn_bilstm_supervised_model(input_shape=(128, 3)):
    inputs = layers.Input(shape=input_shape, name="input")

    x = layers.Bidirectional(
        layers.LSTM(BILSTM_UNITS_1, return_sequences=True),
        name="bilstm_1"
    )(inputs)
    x = layers.Dropout(BILSTM_DROPOUT, name="dropout_1")(x)

    x = layers.Bidirectional(
        layers.LSTM(BILSTM_UNITS_2, return_sequences=True),
        name="bilstm_2"
    )(x)
    x = layers.Dropout(BILSTM_DROPOUT, name="dropout_2")(x)

    context, attn_weights = TemporalAttention(name="temporal_attention")(x)

    cls = layers.Dense(32, activation="relu", name="classifier_dense")(context)
    outputs = layers.Dense(1, activation="sigmoid", name="classifier_output")(cls)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="Attn_BiLSTM_Supervised"
    )
    return model


# =========================================================
# Attention Transformer Supervised Model
# Used for pretrained RL later
# =========================================================

def build_attn_transformer_supervised_model(
    input_shape=(128, 3),
    d_model=TRANSFORMER_D_MODEL,
    num_heads=TRANSFORMER_NUM_HEADS,
    ff_dim=TRANSFORMER_FF_DIM,
    num_layers=TRANSFORMER_NUM_LAYERS,
    dropout=TRANSFORMER_DROPOUT,
):
    inputs = layers.Input(shape=input_shape, name="input")

    x = layers.Dense(d_model, name="input_projection")(inputs)
    x = x + positional_encoding(input_shape[0], d_model)

    last_attn_scores = None
    for i in range(num_layers):
        block = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            name=f"transformer_block_{i+1}"
        )
        x, last_attn_scores = block(x, return_attention=True)

    # last_attn_scores: (B, num_heads, T, T)
    attn_mean_heads = ReduceMeanLayer(axis=1, name="attn_mean_heads")(last_attn_scores)   # (B, T, T)

    token_importance = ReduceMeanLayer(axis=1, name="token_importance_mean")(attn_mean_heads)  # (B, T)

    token_importance = layers.Reshape((input_shape[0], 1), name="token_importance_reshape")(token_importance)

    token_importance = layers.Softmax(axis=1, name="token_importance_softmax")(token_importance)

    weighted_x = layers.Multiply(name="token_weighted_features")([token_importance, x])

    context = ReduceSumLayer(axis=1, name="context_sum")(weighted_x)   # (B, D)

    cls = layers.Dense(32, activation="relu", name="classifier_dense")(context)
    outputs = layers.Dense(1, activation="sigmoid", name="classifier_output")(cls)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="Attn_Transformer_Supervised"
    )
    return model
    


# =========================================================
# Unified Model Builder
# =========================================================
def build_model(model_name, input_shape):
    if model_name == "bilstm":
        return build_bilstm_model(input_shape=input_shape)

    if model_name == "transformer":
        return build_transformer_model(input_shape=input_shape)

    if model_name == "attn_bilstm":
        return build_attn_bilstm_supervised_model(input_shape=input_shape)

    if model_name == "attn_transformer":
        return build_attn_transformer_supervised_model(input_shape=input_shape)

    raise ValueError(f"Unsupported model_name: {model_name}")
