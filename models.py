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


def build_bilstm_model(input_shape=(128, 3)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Bidirectional(
        layers.LSTM(BILSTM_UNITS_1, return_sequences=True)
    )(inputs)
    x = layers.Dropout(BILSTM_DROPOUT)(x)

    x = layers.Bidirectional(
        layers.LSTM(BILSTM_UNITS_2, return_sequences=False)
    )(x)
    x = layers.Dropout(BILSTM_DROPOUT)(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(BILSTM_DROPOUT)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="BiLSTM_Baseline")
    return model


def positional_encoding(length, d_model):
    pos = tf.range(length, dtype=tf.float32)[:, tf.newaxis]
    i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]

    angle_rates = 1.0 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    angle_rads = pos * angle_rates

    sines = tf.sin(angle_rads[:, 0::2])
    cosines = tf.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]

    return tf.cast(pos_encoding, tf.float32)


def transformer_encoder(inputs, d_model, num_heads, ff_dim, dropout=0.1):
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout
    )(inputs, inputs)

    x = layers.Add()([inputs, attn_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(d_model)(ff)

    x = layers.Add()([x, ff])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    return x


def build_transformer_model(
    input_shape=(128, 3),
    d_model=TRANSFORMER_D_MODEL,
    num_heads=TRANSFORMER_NUM_HEADS,
    ff_dim=TRANSFORMER_FF_DIM,
    num_layers=TRANSFORMER_NUM_LAYERS,
    dropout=TRANSFORMER_DROPOUT,
):
    inputs = layers.Input(shape=input_shape)

    x = layers.Dense(d_model)(inputs)
    x = x + positional_encoding(input_shape[0], d_model)

    for _ in range(num_layers):
        x = transformer_encoder(x, d_model, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Transformer_Baseline")
    return model


def build_model(model_name, input_shape):
    if model_name == "bilstm":
        return build_bilstm_model(input_shape=input_shape)

    if model_name == "transformer":
        return build_transformer_model(input_shape=input_shape)

    raise ValueError(f"Unsupported model_name: {model_name}")
