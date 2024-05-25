import random

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from keras.src.optimizers.adam import Adam

from src import Utils, HammingCode

def decode(model, full_sample):

    decoded = []
    for input_index in range(0, len(full_sample), 7):
        input = Utils.toInt(full_sample[input_index:input_index + 7])
        input = tf.expand_dims(input, axis=0)
        decoded_bits = model.predict(input)[0]
        decoded.extend(Utils.roundToBits(decoded_bits))
    return decoded
def create_and_train_auto_encoder(noisy, original, epoches):

    autoencoder = create_auto_encoder()

    noisy_data_reshaped = []
    for noisy_sample in noisy:
        for index in range(0, len(noisy_sample), 7):
            noisy_data_reshaped.append(Utils.toInt(noisy_sample[index:index + 7]))

    noisy_data_reshaped = np.array(noisy_data_reshaped)

    original_data_reshaped = []
    for original_sample in original:
        for index in range(0, len(original_sample), 4):
            original_data_reshaped.append(np.array(Utils.toInt(list(original_sample)[index:index + 4]), dtype=int))

    original_data_reshaped = np.array(original_data_reshaped)

    autoencoder.fit(
        noisy_data_reshaped,
        original_data_reshaped,
        epochs=epoches,
        batch_size=32,
        shuffle=True,
        validation_data=(noisy_data_reshaped, original_data_reshaped)
    )

    loss = autoencoder.evaluate(noisy_data_reshaped, original_data_reshaped)

    print(f'Test Loss: {loss:.4f}')
    return autoencoder

def create_auto_encoder():

    encoding_dim = 3
    input = tfk.layers.Input(shape=(7,))

    # encoder
    encoder = tfk.layers.Dense(encoding_dim, activation='relu')(input)

    # decoder with encoder as input
    decoder = tfk.layers.Dense(4, activation='sigmoid')(encoder)

    # model
    autoencoder = tfk.models.Model(input, decoder)

    autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')
    return autoencoder

