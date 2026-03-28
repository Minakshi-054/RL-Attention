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
# Temporal Attention for BiLSTM
# =========================================================
class TemporalAttention(layers.Layer):
    def __init__(self, attn_units=64, **kwargs):
        super().__init__(**kwargs)
        self.attn_dense = layers.Dense(attn_units, activation="tanh")
        self.score_dense = layers.Dense(1)

    def call(self, inputs):
        """
        inputs: (B, T, H)
        returns:
            context: (B, H)
            weights: (B, T, 1)
        """
        scores = self.score_dense(self.attn_dense(inputs))   # (B, T, 1)
        weights = tf.nn.softmax(scores, axis=1)              # (B, T, 1)
        context = tf.reduce_sum(weights * inputs, axis=1)    # (B, H)
        return context, weights


# =========================================================
# Attention BiLSTM Backbone
# =========================================================
def build_attention_bilstm_backbone(
    input_shape=(128, 3),
    lstm1=BILSTM_UNITS_1,
    lstm2=BILSTM_UNITS_2,
    dropout=BILSTM_DROPOUT,
):
    """
    Returns a backbone that outputs:
        hidden_seq:   (B, T, H)
        context:      (B, H)
        attn_weights: (B, T, 1)
    """
    inputs = layers.Input(shape=input_shape, name="input")

    x = layers.Bidirectional(
        layers.LSTM(lstm1, return_sequences=True),
        name="bilstm_1"
    )(inputs)
    x = layers.Dropout(dropout, name="dropout_1")(x)

    x = layers.Bidirectional(
        layers.LSTM(lstm2, return_sequences=True),
        name="bilstm_2"
    )(x)
    x = layers.Dropout(dropout, name="dropout_2")(x)

    context, attn_weights = TemporalAttention(name="temporal_attention")(x)

    backbone = models.Model(
        inputs=inputs,
        outputs=[x, context, attn_weights],
        name="attention_bilstm_backbone"
    )
    return backbone


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
# ======================================================









#.utils.register_keras_serializable()
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
# Attention Transformer Backbone
# =========================================================
def build_attention_transformer_backbone(
    input_shape=(128, 3),
    d_model=TRANSFORMER_D_MODEL,
    num_heads=TRANSFORMER_NUM_HEADS,
    ff_dim=TRANSFORMER_FF_DIM,
    num_layers=TRANSFORMER_NUM_LAYERS,
    dropout=TRANSFORMER_DROPOUT,
):
    """
    Returns a backbone that outputs:
        hidden_seq:   (B, T, D)
        context:      (B, D)
        attn_weights: (B, T, 1)
    """
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
    attn_mean_heads = ReduceMeanLayer(axis=1, name="attn_mean_heads")(last_attn_scores)

    token_importance = ReduceMeanLayer(axis=1, name="token_importance_mean")(attn_mean_heads)

    token_importance = layers.Reshape((input_shape[0], 1), name="token_importance_reshape")(token_importance)

    token_importance = layers.Softmax(axis=1, name="token_importance_softmax")(token_importance)

    weighted_x = layers.Multiply(name="token_weighted_features")([token_importance, x])

    context = ReduceSumLayer(axis=1, name="context_sum")(weighted_x)

    backbone = models.Model(
        inputs=inputs,
        outputs=[x, context, token_importance],
        name="attention_transformer_backbone"
    )
    return backbone


# =========================================================
# Policy Head
# =========================================================
def build_policy_head(hidden_dim):
    """
    Input:
        hidden_seq: (B, T, H)
    Output:
        policy_probs: (B, T, 1)
    """
    inputs = layers.Input(shape=(None, hidden_dim), name="policy_input")

    x = layers.Dense(64, activation="relu", name="policy_dense_1")(inputs)
    logits = layers.Dense(1, name="policy_logits")(x)
    probs = layers.Softmax(axis=1, name="policy_attention")(logits)

    model = models.Model(inputs=inputs, outputs=probs, name="policy_head")
    return model


# =========================================================
# Classifier Head
# =========================================================
def build_classifier_head(context_dim):
    """
    Input:
        context: (B, H)
    Output:
        pred: (B, 1)
    """
    inputs = layers.Input(shape=(context_dim,), name="classifier_input")

    x = layers.Dense(32, activation="relu", name="classifier_dense")(inputs)
    outputs = layers.Dense(1, activation="sigmoid", name="classifier_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="classifier_head")
    return model


# =========================================================
# Action Sampling Helper
# =========================================================
def sample_one_hot_action(policy_probs):
    """
    policy_probs: (B, T, 1)

    returns:
        one_hot_action: (B, T, 1)
        sampled_idx:    (B, 1)
    """
    probs = tf.squeeze(policy_probs, axis=-1)          # (B, T)
    log_probs = tf.math.log(probs + 1e-8)              # (B, T)

    sampled_idx = tf.random.categorical(log_probs, num_samples=1)   # (B, 1)
    one_hot = tf.one_hot(
        tf.squeeze(sampled_idx, axis=-1),
        depth=tf.shape(probs)[1]
    )                                                  # (B, T)
    one_hot = tf.expand_dims(one_hot, axis=-1)         # (B, T, 1)

    return one_hot, sampled_idx


# =========================================================
# RL BiLSTM Modules
# =========================================================
def build_rl_bilstm_modules(input_shape=(128, 3)):
    backbone = build_attention_bilstm_backbone(input_shape=input_shape)

    # BiLSTM second layer is bidirectional, so hidden dim = 2 * BILSTM_UNITS_2
    hidden_dim = 2 * BILSTM_UNITS_2

    policy_head = build_policy_head(hidden_dim)
    classifier_head = build_classifier_head(hidden_dim)

    return backbone, policy_head, classifier_head


# =========================================================
# RL Transformer Modules
# =========================================================
def build_rl_transformer_modules(input_shape=(128, 3)):
    backbone = build_attention_transformer_backbone(input_shape=input_shape)

    hidden_dim = TRANSFORMER_D_MODEL

    policy_head = build_policy_head(hidden_dim)
    classifier_head = build_classifier_head(hidden_dim)

    return backbone, policy_head, classifier_head
